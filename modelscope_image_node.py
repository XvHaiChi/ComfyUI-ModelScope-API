# -*- coding: utf-8 -*-
"""
ModelScope API-Inference 图像生成 / 图像编辑节点
================================================

全部节点统一走魔搭官方 API-Inference 接口：

    POST https://api-inference.modelscope.cn/v1/images/generations
    headers: Authorization / Content-Type / X-ModelScope-Async-Mode: true
    GET  https://api-inference.modelscope.cn/v1/tasks/{task_id}
    headers: Authorization / X-ModelScope-Task-Type: image_generation

「文生图」与「图像编辑/图生图」使用同一个端点，区别仅在于：
  * 图像编辑会在 payload 里带上 ``image_url``（URL 或 base64 data URI）
  * 使用不同的模型（编辑模型 vs 生成模型）

兼容任意模型：
  每个节点都提供 ``custom_model`` 自定义模型输入，填写后优先于下拉框选择，
  因此无需改代码即可调用魔搭上的新模型。

> 重要：新增控件一律追加到可选输入的**末尾**。
> ComfyUI 保存的工作流按位置匹配 widgets_values，中间插入会导致旧工作流错位。
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from io import BytesIO

import numpy as np
import requests
from PIL import Image

try:
    from . import modelscope_api_client as client
except ImportError:  # 允许作为顶层模块单独导入（测试 / 直接运行）
    import modelscope_api_client as client

# ------------------------------------------------------------------ 默认配置

DEFAULT_CONFIG = {
    "default_model": "Qwen/Qwen-Image",
    "timeout": 720,
    "image_download_timeout": 60,
    "request_timeout": 60,
    "poll_interval": 5,
    "default_prompt": "A beautiful landscape",
    "default_negative_prompt": "",
    "default_width": 1024,
    "default_height": 1024,
    "default_seed": -1,
    "default_steps": 30,
    "default_guidance": 7.5,
    "default_lora_weight": 0.8,
    "auto_normalize_lora": True,
    "image_input_mode": "base64（推荐，无需图床）",
    "image_models": [
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
        "MusePublic/Qwen-image",
        "MusePublic/489_ckpt_FLUX_1",
        "MusePublic/flux-high-res",
        "black-forest-labs/FLUX.1-Krea-dev",
        "MAILAND/majicflus_v1",
        "MoYouuu/MYHuman-QWen",
        "Tongyi-MAI/Z-Image-Turbo",
    ],
    "image_edit_models": [
        "Qwen/Qwen-Image-Edit",
        "Qwen/Qwen-Image-Edit-2511",
        "MusePublic/Qwen-Image-Edit",
        "MusePublic/FLUX.1-Kontext-Dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "FireRedTeam/FireRed-Image-Edit-1.1",
    ],
    "lora_presets": [
        {"name": "无LoRA", "model_id": "", "weight": 0.8}
    ],
    "api_tokens": [],
}

IMAGE_INPUT_MODES = ["base64（推荐，无需图床）", "图床URL（ai.kefan.cn）"]

#: image_url 的写法。官方两个示例不一致：FireRed 单图用字符串，
#: Qwen-Image-Edit-2509 多图用数组，因此这里提供显式开关。
IMAGE_URL_FORMATS = [
    "自动（1张=字符串，多张=数组）",
    "始终数组",
    "始终字符串",
]


# ------------------------------------------------------------------ 配置管理


def load_config() -> dict:
    """从 modelscope_config.json 加载配置，缺失字段自动补默认值。"""
    config_path = os.path.join(os.path.dirname(__file__), "modelscope_config.json")
    config = dict(DEFAULT_CONFIG)
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            config.update(loaded)
    except Exception as exc:  # noqa: BLE001
        print(f"读取配置文件失败，使用默认配置: {exc}")
    # 嵌套结构兜底
    for key in ("lora_presets", "image_models", "image_edit_models", "api_tokens"):
        if not isinstance(config.get(key), list):
            config[key] = DEFAULT_CONFIG[key]
    return config


def save_config(config: dict) -> bool:
    """保存配置到 modelscope_config.json。"""
    config_path = os.path.join(os.path.dirname(__file__), "modelscope_config.json")
    try:
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"保存配置文件失败: {exc}")
        return False


# ------------------------------------------------------------------ API Token


def load_api_tokens():
    try:
        tokens = load_config().get("api_tokens", [])
        if isinstance(tokens, list):
            return [str(t).strip() for t in tokens if str(t).strip()]
    except Exception as exc:  # noqa: BLE001
        print(f"加载API tokens失败: {exc}")
    return []


def save_api_tokens(tokens) -> bool:
    try:
        config = load_config()
        config["api_tokens"] = list(tokens)
        return save_config(config)
    except Exception as exc:  # noqa: BLE001
        print(f"保存API tokens失败: {exc}")
        return False


def parse_api_tokens(token_input):
    """解析用户输入的 Token，支持逗号 / 分号 / 换行分隔。"""
    if not token_input or token_input.strip() in ("", f"***已保存{len(load_api_tokens())}个Token***"):
        return load_api_tokens()
    return [t.strip() for t in re.split(r"[,;\n]+", token_input) if t.strip()]


def _sync_tokens(api_tokens) -> list:
    """解析并在 Token 发生变化时写入配置文件。"""
    tokens = parse_api_tokens(api_tokens)
    if not tokens:
        raise Exception("请提供至少一个有效的 API Token")
    raw = (api_tokens or "").strip()
    if raw and raw != f"***已保存{len(load_api_tokens())}个Token***":
        if save_api_tokens(tokens):
            print(f"✅ 已保存 {len(tokens)} 个 API Token")
        else:
            print("⚠️ API Token 保存失败，但不影响当前使用")
    return tokens


# ------------------------------------------------------------------ 图像工具


def tensor_to_base64_url(image_tensor) -> str:
    """ComfyUI IMAGE 张量 -> base64 data URI（保留旧接口名，供其他节点使用）。"""
    return client.ndarray_to_data_uri(np.asarray(image_tensor))


def tensor_to_pil(image_tensor) -> Image.Image:
    array = np.asarray(image_tensor)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def upload_image_to_host(image_tensor, timeout: int = 30):
    """把图像上传到公共图床换取 URL（失败返回 None）。

    base64 为首选方案，图床仅作为兜底/兼容选项保留。
    """
    temp_path = None
    try:
        pil_image = tensor_to_pil(image_tensor)
        temp_path = os.path.join(
            os.path.dirname(__file__), f".ms_upload_{int(time.time() * 1000)}.jpg"
        )
        pil_image.save(temp_path, format="JPEG", quality=90)
        with open(temp_path, "rb") as handle:
            response = requests.post(
                "https://ai.kefan.cn/api/upload/local",
                files={"file": handle},
                timeout=timeout,
            )
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("data"):
                print(f"📤 图像上传成功: {str(data['data'])[:60]}...")
                return data["data"]
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 图像上传失败: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
    return None


def split_batch(image_tensor, max_images: int = 3):
    """把 ComfyUI 的 IMAGE 张量拆成单帧数组列表（最多 max_images 帧）。

    官方 ``Qwen/Qwen-Image-Edit-2509`` 支持 1-3 张图的多图编辑
    （例如「图一的狗去追图二的飞盘」），所以 batch > 1 时不能只取第一张。
    """
    array = np.asarray(image_tensor)
    if array.ndim == 4:
        return [array[i] for i in range(min(len(array), max_images))]
    return [array]


def _shape_image_url(uris, url_format: str):
    """按官方两种写法整理 image_url：单图字符串 / 多图数组。"""
    if url_format and url_format.startswith("始终数组"):
        return uris
    if url_format and url_format.startswith("始终字符串"):
        return uris[0]
    # 自动：1 张 -> 字符串（与官方 FireRed 示例一致），多张 -> 数组
    return uris[0] if len(uris) == 1 else uris


def resolve_image_urls(image_tensor, mode: str, url_format: str = IMAGE_URL_FORMATS[0]):
    """把输入图像转换成 API 可接受的 ``image_url`` 值（字符串或数组）。"""
    frames = split_batch(image_tensor)
    if len(frames) > 1:
        print(f"🖼️ 检测到 {len(frames)} 张图像，将按多图编辑提交")

    if mode and mode.startswith("图床"):
        urls = []
        for frame in frames:
            url = upload_image_to_host(frame)
            if not url:
                urls = None
                break
            urls.append(url)
        if urls:
            return _shape_image_url(urls, url_format)
        print("🔄 图床不可用，回退到 base64 编码")

    return _shape_image_url(
        [client.ndarray_to_data_uri(frame) for frame in frames], url_format
    )


def resolve_image_url(image_tensor, mode: str) -> str:
    """兼容旧调用：返回单个字符串形式的 image_url。"""
    return resolve_image_urls(image_tensor, mode, IMAGE_URL_FORMATS[2])


# ------------------------------------------------------------------ Payload 构造


def build_image_payload(
    model: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = None,
    height: int = None,
    steps: int = None,
    guidance: float = None,
    seed: int = None,
    image_url: str = None,
    lora_ids=None,
    lora_weights=None,
    auto_normalize_lora: bool = True,
) -> dict:
    """构造 API-Inference 的图像请求体。

    ``image_url`` 存在即代表「图像编辑 / 图生图」，与官方示例一致。
    """
    payload = {
        "model": model,
        "prompt": prompt,
    }
    if negative_prompt and str(negative_prompt).strip():
        payload["negative_prompt"] = negative_prompt
    if width and height:
        payload["size"] = f"{int(width)}x{int(height)}"
    if steps is not None:
        payload["steps"] = int(steps)
    if guidance is not None:
        payload["guidance"] = float(guidance)
    if seed is not None:
        payload["seed"] = int(seed)
    if image_url:
        payload["image_url"] = image_url

    loras = client.normalize_loras(
        client.prepare_lora_pairs(lora_ids or [], lora_weights or []),
        auto_normalize=auto_normalize_lora,
    )
    if loras:
        payload["loras"] = loras
    return payload


def resolve_model_name(selected_model: str, custom_model: str = "") -> str:
    """``custom_model`` 非空时优先使用，从而支持任意魔搭模型。"""
    custom = (custom_model or "").strip()
    if custom:
        if not selected_model or custom != selected_model:
            print(f"✏️ 使用自定义模型: {custom}")
        return custom
    return selected_model


def describe_loras(lora_ids, lora_weights) -> str:
    pairs = client.prepare_lora_pairs(lora_ids, lora_weights)
    if not pairs:
        return "🔧 未使用 LoRA"
    text = "🔧 LoRA 配置: " + ", ".join(f"{i}:{w}" for i, w in pairs)
    if len(pairs) == 1:
        text += "（单个 LoRA 按官方写法只传 ID，权重不生效）"
    return text


# ------------------------------------------------------------------ LoRA 预设管理节点


class ModelScopeLoraPresetNode:
    """管理 modelscope_config.json 中的 LoRA 预设。"""

    @classmethod
    def INPUT_TYPES(cls):
        presets = load_config().get("lora_presets", [])
        names = [preset.get("name", "无LoRA") for preset in presets]

        return {
            "required": {
                "action": (["查看预设", "添加预设", "删除预设", "保存预设"], {"default": "查看预设"}),
            },
            "optional": {
                "preset_name": ("STRING", {
                    "default": "自定义LoRA",
                    "label": "预设名称",
                }),
                "lora_model_id": ("STRING", {
                    "default": "",
                    "label": "LoRA模型ID",
                    "placeholder": "例如：qiyuanai/TikTok_Xiaohongshu_career_line_beauty_v1",
                }),
                "default_weight": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "默认权重",
                }),
                "target_preset": (names if names else ["无LoRA"], {
                    "default": names[0] if names else "无LoRA",
                    "label": "目标预设",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_model_id", "lora_weight", "preset_info")
    FUNCTION = "manage_lora_presets"
    CATEGORY = "ModelScopeAPI/LoRA"

    def manage_lora_presets(
        self,
        action,
        preset_name="",
        lora_model_id="",
        default_weight=0.8,
        target_preset="",
    ):
        config = load_config()
        presets = config.get("lora_presets", [])
        preset_info = f"当前共有 {len(presets)} 个LoRA预设"

        if action == "查看预设":
            lines = ["=== LoRA预设列表 ==="]
            for index, preset in enumerate(presets):
                lines.append(
                    f"{index + 1}. {preset.get('name')} | ID: {preset.get('model_id')} | 权重: {preset.get('weight')}"
                )
            selected = next((p for p in presets if p.get("name") == target_preset),
                            {"model_id": "", "weight": 0.8})
            return (selected.get("model_id"), selected.get("weight"), "\n".join(lines))

        if action == "添加预设":
            if not preset_name or not preset_name.strip():
                raise Exception("预设名称不能为空")
            if any(p.get("name") == preset_name for p in presets):
                raise Exception(f"已存在名为 {preset_name} 的预设")
            presets.append({
                "name": preset_name.strip(),
                "model_id": lora_model_id.strip(),
                "weight": float(default_weight),
            })
            config["lora_presets"] = presets
            save_config(config)
            return (lora_model_id, default_weight, f"成功添加预设: {preset_name} | ID: {lora_model_id}")

        if action == "删除预设":
            if target_preset == "无LoRA":
                raise Exception("不能删除默认的无LoRA预设")
            remaining = [p for p in presets if p.get("name") != target_preset]
            if len(remaining) == len(presets):
                raise Exception(f"未找到预设: {target_preset}")
            config["lora_presets"] = remaining
            save_config(config)
            return ("", 0.8, f"成功删除预设: {target_preset}")

        if action == "保存预设":
            updated = False
            for index, preset in enumerate(presets):
                if preset.get("name") == target_preset:
                    presets[index]["model_id"] = lora_model_id.strip()
                    presets[index]["weight"] = float(default_weight)
                    updated = True
                    break
            if not updated:
                raise Exception(f"未找到预设: {target_preset}")
            config["lora_presets"] = presets
            save_config(config)
            return (lora_model_id, default_weight,
                    f"成功更新预设: {target_preset} | 新ID: {lora_model_id} | 新权重: {default_weight}")

        return ("", 0.8, preset_info)


class ModelScopeSingleLoraLoaderNode:
    """从预设中选择单个 LoRA。"""

    @classmethod
    def INPUT_TYPES(cls):
        presets = load_config().get("lora_presets", [])
        options = [preset.get("name", "无LoRA") for preset in presets] or ["无LoRA"]
        return {
            "required": {
                "lora_preset": (options, {"default": options[0], "label": "LoRA预设"}),
            },
            "optional": {
                "lora_weight": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "自定义权重",
                }),
                "use_custom_weight": ("BOOLEAN", {
                    "default": False,
                    "label_on": "使用自定义权重",
                    "label_off": "使用预设权重",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("lora_id", "lora_weight")
    FUNCTION = "load_single_lora"
    CATEGORY = "ModelScopeAPI/LoRA"

    def load_single_lora(self, lora_preset, lora_weight=0.8, use_custom_weight=False):
        presets = load_config().get("lora_presets", [])
        selected = next(
            (p for p in presets if p.get("name") == lora_preset),
            {"model_id": "", "weight": 0.8},
        )
        return (selected.get("model_id", ""),
                lora_weight if use_custom_weight else selected.get("weight", 0.8))


class ModelScopeMultiLoraLoaderNode:
    """从预设中选择 3 个 LoRA（保持原有输出结构，避免破坏旧工作流）。"""

    @classmethod
    def INPUT_TYPES(cls):
        presets = load_config().get("lora_presets", [])
        options = [preset.get("name", "无LoRA") for preset in presets] or ["无LoRA"]
        return {
            "required": {
                "lora1_preset": (options, {"default": options[0], "label": "LoRA 1 预设"}),
                "lora2_preset": (options, {"default": options[0], "label": "LoRA 2 预设"}),
                "lora3_preset": (options, {"default": options[0], "label": "LoRA 3 预设"}),
            },
            "optional": {
                "lora1_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 1 权重"}),
                "lora2_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 2 权重"}),
                "lora3_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA 3 权重"}),
                "lora1_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA1用自定义权重", "label_off": "用预设权重"}),
                "lora2_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA2用自定义权重", "label_off": "用预设权重"}),
                "lora3_use_custom": ("BOOLEAN", {"default": False, "label_on": "LoRA3用自定义权重", "label_off": "用预设权重"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("lora1_id", "lora2_id", "lora3_id", "lora1_w", "lora2_w", "lora3_w")
    FUNCTION = "load_multi_lora"
    CATEGORY = "ModelScopeAPI/LoRA"

    def _resolve(self, preset_name, custom_weight, use_custom):
        presets = load_config().get("lora_presets", [])
        preset = next((p for p in presets if p.get("name") == preset_name),
                      {"model_id": "", "weight": 0.8})
        return preset.get("model_id", ""), (custom_weight if use_custom else preset.get("weight", 0.8))

    def load_multi_lora(
        self,
        lora1_preset, lora2_preset, lora3_preset,
        lora1_weight=0.8, lora2_weight=0.8, lora3_weight=0.8,
        lora1_use_custom=False, lora2_use_custom=False, lora3_use_custom=False,
    ):
        id1, w1 = self._resolve(lora1_preset, lora1_weight, lora1_use_custom)
        id2, w2 = self._resolve(lora2_preset, lora2_weight, lora2_use_custom)
        id3, w3 = self._resolve(lora3_preset, lora3_weight, lora3_use_custom)
        return (id1, id2, id3, w1, w2, w3)


class ModelScopeSixLoraLoaderNode:
    """从预设中选择最多 6 个 LoRA（官方上限即 6 个）。"""

    @classmethod
    def INPUT_TYPES(cls):
        presets = load_config().get("lora_presets", [])
        options = [preset.get("name", "无LoRA") for preset in presets] or ["无LoRA"]
        required = {
            f"lora{i}_preset": (options, {"default": options[0], "label": f"LoRA {i} 预设"})
            for i in range(1, 7)
        }
        optional = {
            f"lora{i}_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": f"LoRA {i} 权重"})
            for i in range(1, 7)
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING",) * 6 + ("FLOAT",) * 6
    RETURN_NAMES = tuple([f"lora{i}_id" for i in range(1, 7)] + [f"lora{i}_w" for i in range(1, 7)])
    FUNCTION = "load_six_lora"
    CATEGORY = "ModelScopeAPI/LoRA"

    def load_six_lora(self, **kwargs):
        presets = load_config().get("lora_presets", [])
        ids, weights = [], []
        for index in range(1, 7):
            preset_name = kwargs.get(f"lora{index}_preset", "无LoRA")
            preset = next((p for p in presets if p.get("name") == preset_name),
                          {"model_id": "", "weight": 0.8})
            ids.append(preset.get("model_id", ""))
            weights.append(float(kwargs.get(f"lora{index}_weight", preset.get("weight", 0.8))))
        return tuple(ids + weights)


class ModelScopeModelListRefreshNode:
    """从 /v1/models 拉取当前可用模型，并按需写回配置文件。"""

    @classmethod
    def INPUT_TYPES(cls):
        saved_tokens = load_api_tokens()
        return {
            "required": {
                "action": (["仅查看", "查看并写入配置文件"], {"default": "仅查看"}),
            },
            "optional": {
                "api_tokens": ("STRING", {
                    "default": f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else "",
                    "placeholder": "留空使用已保存的 Token",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_models", "image_edit_models", "model_report")
    FUNCTION = "refresh_models"
    CATEGORY = "ModelScopeAPI/LoRA"

    def refresh_models(self, action, api_tokens=""):
        tokens = parse_api_tokens(api_tokens)
        models = client.fetch_available_models(tokens[0] if tokens else None)
        if not models:
            return ([], [], "⚠️ 未获取到模型列表，请检查网络或 Token")

        buckets = client.classify_models(models)
        image_models = sorted(set(load_config().get("image_models", []) + buckets["image_models"]))
        edit_models = sorted(set(load_config().get("image_edit_models", []) + buckets["image_edit_models"]))

        report = [
            f"=== 共获取 {len(models)} 个模型 ===",
            f"生图类: {len(buckets['image_models'])}",
            f"图像编辑类: {len(buckets['image_edit_models'])}",
            f"视觉理解类: {len(buckets['vision_models'])}",
            f"文本类: {len(buckets['text_models'])}",
        ]

        if action == "查看并写入配置文件":
            config = load_config()
            config["image_models"] = image_models
            config["image_edit_models"] = edit_models
            if save_config(config):
                report.append("✅ 已写入 modelscope_config.json，重启 ComfyUI 后下拉框生效")
            else:
                report.append("❌ 写入配置文件失败")
        else:
            report.append("ℹ️ 仅查看模式，未写入配置文件")

        return ("\n".join(image_models), "\n".join(edit_models), "\n".join(report))


# ------------------------------------------------------------------ 文生图节点


class ModelScopeImageNode:
    """ModelScope-Image 生图节点（API-Inference 异步模式）。"""

    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        token_hint = f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else ""

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "A beautiful landscape"),
                }),
                "api_tokens": ("STRING", {
                    "default": token_hint,
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）" if not saved_tokens else "留空使用已保存的Token",
                    "multiline": True,
                }),
            },
            "optional": {
                "model": (config.get("image_models", []), {
                    "default": config.get("default_model", "Qwen/Qwen-Image"),
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_negative_prompt", ""),
                }),
                "width": ("INT", {
                    "default": config.get("default_width", 1024),
                    "min": 64, "max": 2048, "step": 64,
                }),
                "height": ("INT", {
                    "default": config.get("default_height", 1024),
                    "min": 64, "max": 2048, "step": 64,
                }),
                "seed": ("INT", {
                    "default": config.get("default_seed", -1),
                    "min": -1, "max": 2147483647,
                }),
                "steps": ("INT", {
                    "default": config.get("default_steps", 30),
                    "min": 1, "max": 100,
                }),
                "guidance": ("FLOAT", {
                    "default": config.get("default_guidance", 7.5),
                    "min": 1.5, "max": 20.0, "step": 0.1,
                }),
                "lora1_id": ("STRING", {"default": "", "label": "LoRA1 模型ID"}),
                "lora1_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA1 权重"}),
                "lora2_id": ("STRING", {"default": "", "label": "LoRA2 模型ID"}),
                "lora2_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA2 权重"}),
                "lora3_id": ("STRING", {"default": "", "label": "LoRA3 模型ID"}),
                "lora3_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA3 权重"}),
                # ↓↓↓ 以下是新增控件，必须排在末尾以保证旧工作流兼容 ↓↓↓
                "lora4_id": ("STRING", {"default": "", "label": "LoRA4 模型ID"}),
                "lora4_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA4 权重"}),
                "lora5_id": ("STRING", {"default": "", "label": "LoRA5 模型ID"}),
                "lora5_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA5 权重"}),
                "lora6_id": ("STRING", {"default": "", "label": "LoRA6 模型ID"}),
                "lora6_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA6 权重"}),
                "custom_model": ("STRING", {
                    "default": "",
                    "label": "自定义模型ID（填了优先于上面的下拉框）",
                    "placeholder": "例如：Tongyi-MAI/Z-Image-Turbo",
                }),
                "auto_normalize_lora": ("BOOLEAN", {
                    "default": config.get("auto_normalize_lora", True),
                    "label_on": "LoRA权重自动归一化到1.0",
                    "label_off": "LoRA权重原样提交",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "ModelScopeAPI"

    def generate_image(
        self,
        prompt,
        api_tokens,
        model="Qwen/Qwen-Image",
        negative_prompt="",
        width=1024,
        height=1024,
        seed=-1,
        steps=30,
        guidance=7.5,
        lora1_id="", lora1_w=0.8,
        lora2_id="", lora2_w=0.8,
        lora3_id="", lora3_w=0.8,
        lora4_id="", lora4_w=0.8,
        lora5_id="", lora5_w=0.8,
        lora6_id="", lora6_w=0.8,
        custom_model="",
        auto_normalize_lora=True,
    ):
        config = load_config()
        tokens = _sync_tokens(api_tokens)
        model_name = resolve_model_name(model, custom_model)

        actual_seed = int(seed) if seed and int(seed) != -1 else random.randint(0, 2147483647)

        print("🔍 开始生成图像...")
        print(f"📝 提示词: {prompt}")
        print(f"❌ 反向提示词: {negative_prompt if negative_prompt else '无'}")
        print(f"🤖 模型: {model_name}")
        print(f"🔑 可用Token数量: {len(tokens)}")
        print(f"📐 尺寸: {width}x{height}")
        print(f"🔄 步数: {steps}")
        print(f"🧭 引导系数: {guidance}")
        print(f"🔢 种子: {actual_seed}{'（随机）' if seed == -1 else ''}")
        print(describe_loras(
            [lora1_id, lora2_id, lora3_id, lora4_id, lora5_id, lora6_id],
            [lora1_w, lora2_w, lora3_w, lora4_w, lora5_w, lora6_w],
        ))

        def build_payload(_token):
            return build_image_payload(
                model=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=actual_seed,
                lora_ids=[lora1_id, lora2_id, lora3_id, lora4_id, lora5_id, lora6_id],
                lora_weights=[lora1_w, lora2_w, lora3_w, lora4_w, lora5_w, lora6_w],
                auto_normalize_lora=auto_normalize_lora,
            )

        urls, used_token = client.generate_with_token_rotation(
            tokens,
            build_payload,
            request_timeout=config.get("request_timeout", 60),
            task_timeout=config.get("timeout", 720),
            poll_interval=config.get("poll_interval", 5),
        )

        print(f"📥 下载图片 ({len(urls)} 张)...")
        pil_image = client.download_image(
            urls[0], timeout=config.get("image_download_timeout", 60)
        )
        print(f"✅ 图像生成完成! (使用 Token 尾号 ...{str(used_token)[-4:]})")
        return (client.image_to_tensor(pil_image),)


# ------------------------------------------------------------------ 图像编辑 / 图生图节点


class ModelScopeImageEditNode:
    """ModelScope-Image 图像编辑 / 图生图节点（统一端点 + image_url）。"""

    @classmethod
    def INPUT_TYPES(cls):
        config = load_config()
        saved_tokens = load_api_tokens()
        token_hint = f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else ""
        edit_models = config.get("image_edit_models", [])
        gen_models = config.get("image_models", [])

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "修改图片中的内容"}),
                "api_tokens": ("STRING", {
                    "default": token_hint,
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）" if not saved_tokens else "留空使用已保存的Token",
                    "multiline": True,
                }),
                "image_gen_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "图生图模式",
                    "label_off": "图像编辑模式",
                }),
            },
            "optional": {
                "gen_model": (gen_models, {
                    "default": gen_models[0] if gen_models else "Qwen/Qwen-Image",
                }),
                "edit_model": (edit_models, {
                    "default": edit_models[0] if edit_models else "Qwen/Qwen-Image-Edit",
                }),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 1664, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 1664, "step": 8}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 1.5, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "lora1_id": ("STRING", {"default": "", "label": "LoRA1 模型ID"}),
                "lora1_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA1 权重"}),
                "lora2_id": ("STRING", {"default": "", "label": "LoRA2 模型ID"}),
                "lora2_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA2 权重"}),
                "lora3_id": ("STRING", {"default": "", "label": "LoRA3 模型ID"}),
                "lora3_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA3 权重"}),
                # ↓↓↓ 新增控件，排在末尾以保证旧工作流兼容 ↓↓↓
                "lora4_id": ("STRING", {"default": "", "label": "LoRA4 模型ID"}),
                "lora4_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA4 权重"}),
                "lora5_id": ("STRING", {"default": "", "label": "LoRA5 模型ID"}),
                "lora5_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA5 权重"}),
                "lora6_id": ("STRING", {"default": "", "label": "LoRA6 模型ID"}),
                "lora6_w": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.1, "label": "LoRA6 权重"}),
                "custom_model": ("STRING", {
                    "default": "",
                    "label": "自定义模型ID（填了优先于上面的下拉框）",
                    "placeholder": "例如：FireRedTeam/FireRed-Image-Edit-1.1",
                }),
                "auto_normalize_lora": ("BOOLEAN", {
                    "default": config.get("auto_normalize_lora", True),
                    "label_on": "LoRA权重自动归一化到1.0",
                    "label_off": "LoRA权重原样提交",
                }),
                "image_input_mode": (IMAGE_INPUT_MODES, {
                    "default": config.get("image_input_mode", IMAGE_INPUT_MODES[0]),
                }),
                "image_url_format": (IMAGE_URL_FORMATS, {
                    "default": config.get("image_url_format", IMAGE_URL_FORMATS[0]),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "ModelScopeAPI"

    def edit_image(
        self,
        image,
        prompt,
        api_tokens,
        image_gen_mode=False,
        gen_model="Qwen/Qwen-Image",
        edit_model="Qwen/Qwen-Image-Edit",
        negative_prompt="",
        width=512,
        height=512,
        steps=30,
        guidance=3.5,
        seed=-1,
        lora1_id="", lora1_w=0.8,
        lora2_id="", lora2_w=0.8,
        lora3_id="", lora3_w=0.8,
        lora4_id="", lora4_w=0.8,
        lora5_id="", lora5_w=0.8,
        lora6_id="", lora6_w=0.8,
        custom_model="",
        auto_normalize_lora=True,
        image_input_mode=None,
        image_url_format=None,
    ):
        config = load_config()
        tokens = _sync_tokens(api_tokens)

        selected_model = gen_model if image_gen_mode else edit_model
        model_name = resolve_model_name(selected_model, custom_model)
        mode_label = "图生图模式" if image_gen_mode else "图像编辑模式"
        image_mode = image_input_mode or config.get("image_input_mode", IMAGE_INPUT_MODES[0])
        url_format = image_url_format or config.get("image_url_format", IMAGE_URL_FORMATS[0])
        actual_seed = int(seed) if seed and int(seed) != -1 else random.randint(0, 2147483647)

        print("🔍 开始图像编辑...")
        print(f"📝 提示词: {prompt}")
        print(f"❌ 反向提示词: {negative_prompt if negative_prompt else '无'}")
        print(f"🤖 模型: {model_name} ({mode_label})")
        print(f"🖼️ 图像输入方式: {image_mode} | image_url 写法: {url_format}")
        print(f"🔑 可用Token数量: {len(tokens)}")
        print(f"📐 尺寸: {width}x{height}")
        print(f"🔄 步数: {steps}")
        print(f"🧭 引导系数: {guidance}")
        print(f"🔢 种子: {actual_seed}{'（随机）' if seed == -1 else ''}")
        print(describe_loras(
            [lora1_id, lora2_id, lora3_id, lora4_id, lora5_id, lora6_id],
            [lora1_w, lora2_w, lora3_w, lora4_w, lora5_w, lora6_w],
        ))

        print("🔄 转换输入图像...")
        image_url = resolve_image_urls(image, image_mode, url_format)

        def build_payload(_token):
            return build_image_payload(
                model=model_name,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                seed=actual_seed,
                image_url=image_url,
                lora_ids=[lora1_id, lora2_id, lora3_id, lora4_id, lora5_id, lora6_id],
                lora_weights=[lora1_w, lora2_w, lora3_w, lora4_w, lora5_w, lora6_w],
                auto_normalize_lora=auto_normalize_lora,
            )

        urls, used_token = client.generate_with_token_rotation(
            tokens,
            build_payload,
            request_timeout=config.get("request_timeout", 60),
            task_timeout=config.get("timeout", 720),
            poll_interval=config.get("poll_interval", 5),
        )

        print(f"📥 下载结果图片 ({len(urls)} 张)...")
        pil_image = client.download_image(
            urls[0], timeout=config.get("image_download_timeout", 60)
        )
        print(f"✅ 图像处理完成! (使用 Token 尾号 ...{str(used_token)[-4:]})")
        return (client.image_to_tensor(pil_image),)


# ------------------------------------------------------------------ 节点映射

NODE_CLASS_MAPPINGS = {
    "ModelScopeImageNode": ModelScopeImageNode,
    "ModelScopeImageEditNode": ModelScopeImageEditNode,
    "ModelScopeLoraPresetNode": ModelScopeLoraPresetNode,
    "ModelScopeSingleLoraLoaderNode": ModelScopeSingleLoraLoaderNode,
    "ModelScopeMultiLoraLoaderNode": ModelScopeMultiLoraLoaderNode,
    "ModelScopeSixLoraLoaderNode": ModelScopeSixLoraLoaderNode,
    "ModelScopeModelListRefreshNode": ModelScopeModelListRefreshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageNode": "ModelScope-Image 生图节点",
    "ModelScopeImageEditNode": "ModelScope-Image 图像编辑节点",
    "ModelScopeLoraPresetNode": "ModelScope-LoRA 预设管理",
    "ModelScopeSingleLoraLoaderNode": "ModelScope-LoRA 单LoRA加载",
    "ModelScopeMultiLoraLoaderNode": "ModelScope-LoRA 多LoRA加载(3)",
    "ModelScopeSixLoraLoaderNode": "ModelScope-LoRA 多LoRA加载(6)",
    "ModelScopeModelListRefreshNode": "ModelScope 模型列表刷新",
}
