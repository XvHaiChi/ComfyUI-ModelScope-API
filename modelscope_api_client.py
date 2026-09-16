# -*- coding: utf-8 -*-
"""
ModelScope API-Inference 统一客户端
=====================================

本模块把魔搭 ModelScope 官方 API-Inference 的调用协议收敛到一处，供所有节点复用。

官方协议要点（https://www.modelscope.cn/docs/model-service/API-Inference/intro）：

1. 提交任务（异步模式）
   POST https://api-inference.modelscope.cn/v1/images/generations
   headers:
       Authorization: Bearer <token>
       Content-Type: application/json
       X-ModelScope-Async-Mode: true          <-- 只有提交任务时才带
   body:
       {
         "model": "<org/model>",              # 必填
         "prompt": "...",                     # 必填
         "negative_prompt": "...",            # 可选
         "size": "1024x1024",                 # 可选
         "seed": 12345,                       # 可选
         "steps": 30,                         # 可选
         "guidance": 3.5,                     # 可选
         "image_url": "<url 或 base64 data URI>",  # 图像编辑/图生图时使用
         "loras": "<lora-id>"                 # 单个 LoRA
             或  {"<id1>": 0.6, "<id2>": 0.4} # 多个 LoRA（最多 6 个，权重和为 1.0）
       }
   返回: {"task_id": "..."}

2. 轮询任务
   GET https://api-inference.modelscope.cn/v1/tasks/{task_id}
   headers:
       Authorization: Bearer <token>
       X-ModelScope-Task-Type: image_generation   <-- 轮询时才带
   返回: {"task_status": "SUCCEED"|"FAILED"|..., "output_images": ["<url>", ...]}

3. 同步模式（不带 X-ModelScope-Async-Mode）会直接返回 {"images": [{"url": ...}]}。

设计原则
--------
* 本模块只依赖 requests / Pillow / numpy，**不导入 torch**，因此可以脱离 ComfyUI 单独做单元测试。
* 文本类接口（chat/completions）由各节点直接使用 openai SDK，不在此模块内。
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

# ------------------------------------------------------------------ 常量

API_BASE_URL = "https://api-inference.modelscope.cn/"
IMAGE_GENERATION_ENDPOINT = API_BASE_URL + "v1/images/generations"
TASKS_ENDPOINT = API_BASE_URL + "v1/tasks/{task_id}"
MODELS_ENDPOINT = API_BASE_URL + "v1/models"

#: 轮询任务时必须携带的 X-ModelScope-Task-Type
DEFAULT_TASK_TYPE = "image_generation"

#: 官方限制：最多 6 个 LoRA
MAX_LORAS = 6

#: 任务终态
_SUCCESS_STATES = {"SUCCEED", "SUCCESS", "SUCCEEDED"}
_FAILED_STATES = {"FAILED", "FAIL", "CANCELED", "CANCELLED", "TIMEOUT", "ERROR"}

#: 默认超时（秒）
DEFAULT_REQUEST_TIMEOUT = 60
DEFAULT_TASK_TIMEOUT = 720
DEFAULT_DOWNLOAD_TIMEOUT = 60
DEFAULT_POLL_INTERVAL = 5

#: 当模型不支持某个可选参数时，精简重试会剔除这些字段
OPTIONAL_PARAM_KEYS = (
    "negative_prompt",
    "size",
    "steps",
    "guidance",
    "seed",
    "loras",
)


class ModelScopeAPIError(RuntimeError):
    """API-Inference 调用失败时抛出。"""


# ------------------------------------------------------------------ 请求头


def build_headers(
    token: str,
    async_mode: bool = False,
    task_type: Optional[str] = None,
) -> Dict[str, str]:
    """构造 API-Inference 请求头。

    :param token: 魔搭访问令牌
    :param async_mode: 提交任务时置 True，会带上 ``X-ModelScope-Async-Mode: true``
    :param task_type: 轮询任务时传入，例如 ``image_generation``
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if async_mode:
        headers["X-ModelScope-Async-Mode"] = "true"
    if task_type:
        headers["X-ModelScope-Task-Type"] = task_type
    return headers


# ------------------------------------------------------------------ LoRA


def prepare_lora_pairs(
    lora_ids: Sequence[str],
    lora_weights: Sequence[float],
) -> List[Tuple[str, float]]:
    """把并行的 id / weight 列表整理成 ``[(id, weight), ...]``，过滤空 id。"""
    pairs: List[Tuple[str, float]] = []
    for lora_id, weight in zip(lora_ids, lora_weights):
        if lora_id is None:
            continue
        clean_id = str(lora_id).strip()
        if not clean_id:
            continue
        try:
            clean_weight = float(weight)
        except (TypeError, ValueError):
            clean_weight = 1.0
        pairs.append((clean_id, clean_weight))
    return pairs


def normalize_loras(
    pairs: Sequence[Tuple[str, float]],
    auto_normalize: bool = True,
    single_as_string: bool = True,
) -> Optional[Any]:
    """把 LoRA 列表转换成 API-Inference 需要的 ``loras`` 字段。

    官方规则：
      * 单个 LoRA  -> ``"loras": "<lora-repo-id>"``
      * 多个 LoRA  -> ``"loras": {"<id1>": 0.6, "<id2>": 0.4}``，最多 6 个，权重和为 1.0

    :param auto_normalize: 权重和不为 1.0 时自动归一化（避免 API 直接报错）
    :param single_as_string: 单个 LoRA 时返回字符串形式（官方推荐写法）
    :returns: ``None`` / ``str`` / ``dict``
    """
    if not pairs:
        return None

    # 去重（同一个 LoRA 只保留一次，后出现的权重覆盖先出现的）
    merged: Dict[str, float] = {}
    for lora_id, weight in pairs:
        merged[lora_id] = float(weight)

    # 官方限制最多 6 个
    items = list(merged.items())[:MAX_LORAS]

    if len(items) == 1:
        lora_id, weight = items[0]
        if single_as_string:
            # 单个 LoRA 官方写法就是字符串，权重不参与
            return lora_id
        return {lora_id: float(weight)}

    weights = [w for _, w in items]
    total = sum(weights)
    if auto_normalize and total > 0 and abs(total - 1.0) > 1e-6:
        items = [(lora_id, w / total) for lora_id, w in items]

    # 四舍五入到 4 位小数，避免浮点尾差导致总和不是 1.0
    rounded = [(lora_id, round(w, 4)) for lora_id, w in items]
    diff = round(1.0 - sum(w for _, w in rounded), 4)
    if auto_normalize and rounded and abs(diff) > 0:
        last_id, last_w = rounded[-1]
        rounded[-1] = (last_id, round(last_w + diff, 4))

    return {lora_id: w for lora_id, w in rounded}


# ------------------------------------------------------------------ 图像编解码


def pil_to_data_uri(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """PIL 图像 -> ``data:image/jpeg;base64,...``"""
    buffer = io.BytesIO()
    if fmt.upper() == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(buffer, format=fmt, quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def ndarray_to_data_uri(array: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> str:
    """numpy 图像数组（HWC，取值 0-255 或 0-1）-> data URI。"""
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return pil_to_data_uri(Image.fromarray(array), fmt=fmt, quality=quality)


def download_image(
    url: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Image.Image:
    """下载图片并返回 RGB 模式的 PIL Image。"""
    getter = session.get if session is not None else requests.get
    response = getter(url, timeout=timeout)
    if response.status_code != 200:
        raise ModelScopeAPIError(
            f"图片下载失败: HTTP {response.status_code} ({url[:80]})"
        )
    image = Image.open(io.BytesIO(response.content))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def image_to_tensor(image: Image.Image):
    """PIL Image -> ComfyUI 使用的 IMAGE 张量（延迟导入 torch）。"""
    import torch  # 延迟导入，保证本模块可脱离 ComfyUI 测试

    if image.mode != "RGB":
        image = image.convert("RGB")
    array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(array)[None,]


# ------------------------------------------------------------------ 任务提交 / 轮询


def _extract_error(data: Dict[str, Any]) -> str:
    """从任务返回体里尽力提取可读的错误信息。"""
    errors = data.get("errors")
    if isinstance(errors, dict):
        code = errors.get("code")
        message = errors.get("message")
        parts = [str(p) for p in (code, message) if p]
        if parts:
            return " / ".join(parts)
    if isinstance(errors, list) and errors:
        return "; ".join(str(e) for e in errors)
    for key in ("message", "error", "error_message", "detail"):
        if data.get(key):
            return str(data[key])
    return json.dumps(data, ensure_ascii=False)[:500]


def submit_image_task(
    token: str,
    payload: Dict[str, Any],
    async_mode: bool = True,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """提交图像生成/编辑任务，返回原始 JSON 响应。

    异步模式下返回 ``{"task_id": ...}``；同步模式下返回 ``{"images": [...]}``。
    """
    poster = session.post if session is not None else requests.post
    headers = build_headers(token, async_mode=async_mode)
    response = poster(
        IMAGE_GENERATION_ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        timeout=timeout,
    )

    if response.status_code != 200:
        raise ModelScopeAPIError(
            f"提交任务失败: HTTP {response.status_code} - {response.text[:500]}"
        )

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - 极端情况
        raise ModelScopeAPIError(f"接口返回非 JSON 内容: {response.text[:300]}") from exc


def poll_image_task(
    token: str,
    task_id: str,
    task_type: str = DEFAULT_TASK_TYPE,
    task_timeout: float = DEFAULT_TASK_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    session: Optional[requests.Session] = None,
    logger=print,
) -> List[str]:
    """轮询任务直到成功，返回图片 URL 列表。"""
    getter = session.get if session is not None else requests.get
    headers = build_headers(token, task_type=task_type)
    url = TASKS_ENDPOINT.format(task_id=task_id)

    started = time.time()
    while True:
        response = getter(url, headers=headers, timeout=request_timeout)
        if response.status_code != 200:
            raise ModelScopeAPIError(
                f"任务查询失败: HTTP {response.status_code} - {response.text[:300]}"
            )

        data = response.json()
        status = str(data.get("task_status", "")).upper()
        elapsed = int(time.time() - started)
        logger(f"⌛ 任务状态: {status or '未知'} (已等待 {elapsed} 秒)")

        if status in _SUCCESS_STATES:
            images = data.get("output_images") or []
            if not images:
                raise ModelScopeAPIError("任务成功但未返回图片地址")
            return list(images)

        if status in _FAILED_STATES:
            raise ModelScopeAPIError(f"任务失败: {_extract_error(data)}")

        if elapsed > task_timeout:
            raise ModelScopeAPIError(
                f"任务轮询超时（{int(task_timeout)} 秒），请稍后重试或降低并发"
            )

        time.sleep(poll_interval)


def strip_optional_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    """剔除可选调参字段，只保留模型与必填内容，用于兼容性降级重试。"""
    return {k: v for k, v in payload.items() if k not in OPTIONAL_PARAM_KEYS}


def run_image_generation(
    token: str,
    payload: Dict[str, Any],
    async_mode: bool = True,
    task_type: str = DEFAULT_TASK_TYPE,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    task_timeout: float = DEFAULT_TASK_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    allow_minimal_retry: bool = True,
    session: Optional[requests.Session] = None,
    logger=print,
) -> List[str]:
    """一次完整的「提交 -> 轮询」流程，返回图片 URL 列表。

    同时兼容异步（task_id）与同步（images）两种返回格式。
    若接口因不接受某些可选参数而返回 400，会自动降级为精简参数重试一次。
    """
    try:
        data = submit_image_task(
            token, payload, async_mode=async_mode, timeout=request_timeout, session=session
        )
    except ModelScopeAPIError as exc:
        if not allow_minimal_retry or "400" not in str(exc):
            raise
        minimal = strip_optional_params(payload)
        if minimal == payload:
            raise
        logger("⚠️ 完整参数被拒绝（HTTP 400），自动使用精简参数重试...")
        data = submit_image_task(
            token, minimal, async_mode=async_mode, timeout=request_timeout, session=session
        )

    if data.get("task_id"):
        logger(f"📌 任务已提交，task_id = {data['task_id']}")
        return poll_image_task(
            token,
            data["task_id"],
            task_type=task_type,
            task_timeout=task_timeout,
            poll_interval=poll_interval,
            request_timeout=request_timeout,
            session=session,
            logger=logger,
        )

    images = data.get("output_images") or data.get("images") or []
    urls: List[str] = []
    for item in images:
        if isinstance(item, str):
            urls.append(item)
        elif isinstance(item, dict) and item.get("url"):
            urls.append(item["url"])
    if urls:
        logger("✅ 接口同步返回图片地址")
        return urls

    raise ModelScopeAPIError(f"未识别的接口返回格式: {json.dumps(data, ensure_ascii=False)[:500]}")


# ------------------------------------------------------------------ 多 Token 轮询


def generate_with_token_rotation(
    tokens: Sequence[str],
    build_payload,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    task_timeout: float = DEFAULT_TASK_TIMEOUT,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    logger=print,
) -> Tuple[List[str], str]:
    """依次尝试每个 Token，返回 ``(图片URL列表, 成功的token)``。

    :param build_payload: 接收 token 返回 payload 字典的回调（便于按 Token 调整参数）
    """
    if not tokens:
        raise ModelScopeAPIError("请提供至少一个有效的 API Token")

    session = requests.Session()
    last_error: Optional[Exception] = None
    try:
        for index, token in enumerate(tokens):
            try:
                logger(f"🔄 尝试使用第 {index + 1}/{len(tokens)} 个 Token...")
                payload = build_payload(token)
                urls = run_image_generation(
                    token,
                    payload,
                    request_timeout=request_timeout,
                    task_timeout=task_timeout,
                    poll_interval=poll_interval,
                    session=session,
                    logger=logger,
                )
                return urls, token
            except Exception as exc:  # noqa: BLE001 - 需要逐个 Token 兜底
                last_error = exc
                logger(f"❌ 第 {index + 1} 个 Token 调用失败: {exc}")
                if index < len(tokens) - 1:
                    logger("⏳ 准备尝试下一个 Token...")
    finally:
        session.close()

    raise ModelScopeAPIError(
        f"所有 {len(tokens)} 个 API Token 都失败了。最后的错误: {last_error}"
    )


# ------------------------------------------------------------------ 模型列表


def fetch_available_models(
    token: Optional[str] = None,
    timeout: float = 30,
    logger=print,
) -> List[str]:
    """从 ``/v1/models`` 拉取当前账号可用的模型 ID 列表。

    该接口在未鉴权时也能返回一份公共模型清单。
    """
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(MODELS_ENDPOINT, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger(f"⚠️ 获取模型列表失败: {exc}")
        return []

    items = data.get("data") or []
    model_ids = sorted({str(m.get("id")) for m in items if isinstance(m, dict) and m.get("id")})
    logger(f"✅ 获取到 {len(model_ids)} 个可用模型")
    return model_ids


def classify_models(model_ids: Iterable[str]) -> Dict[str, List[str]]:
    """按用途粗略归类模型 ID，供节点下拉框使用。"""
    buckets: Dict[str, List[str]] = {
        "image_models": [],
        "image_edit_models": [],
        "text_models": [],
        "vision_models": [],
    }
    edit_keywords = ("edit", "kontext", "fire-red", "firered")
    image_keywords = (
        "image", "flux", "z-image", "seedream", "kolors", "stable-diffusion",
        "sd3", "sdxl", "hunyuanimage", "janus",
    )
    vision_keywords = ("vl", "vision", "internvl", "step3", "step-3", "qwen-vl", "omni")

    for model_id in model_ids:
        low = model_id.lower()
        if any(k in low for k in edit_keywords):
            buckets["image_edit_models"].append(model_id)
        elif any(k in low for k in image_keywords):
            buckets["image_models"].append(model_id)
        elif any(k in low for k in vision_keywords):
            buckets["vision_models"].append(model_id)
        else:
            buckets["text_models"].append(model_id)
    return buckets
