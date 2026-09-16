#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_model_ids.py —— 批量核验 ModelScope 模型 ID 是否真实存在、是否可用于 API-Inference。

为什么要用这个脚本：
    https://www.modelscope.cn/models/<org>/<name> 对**不存在的模型也返回 HTTP 200**
    （纯前端 SPA），所以状态码完全不可靠。旧办法是抓页面 <title>，能判断存在性
    但拿不到任何结构化信息。

更好的办法（本脚本采用）：
    GET https://www.modelscope.cn/api/v1/models/<org>/<name>
    —— **无需登录**即可返回完整结构化元数据（AigcType / SupportInference /
    AigcAttributes / Tasks / widgets / Downloads / Visibility ...）。

用法：
    python check_model_ids.py                     # 核验 modelscope_config.json 里的全部模型 ID
    python check_model_ids.py krea/Krea-2-Turbo   # 核验指定 ID
    python check_model_ids.py --quiet a/b c/d     # 只输出结论行

退出码：0 = 全部可用；1 = 存在死 ID 或不可用的模型。

⚠️ 关于 SupportApiInference 字段：**不要用它判断能否 API 调用**。
   实测连 Qwen/Qwen-Image、FireRedTeam/FireRed-Image-Edit-1.1 这类官方示例主力模型
   该字段也是 false。可靠的判断依据是下面两项**稳定**字段：
     - SupportInference        非空（"txt2img" / "img2img"）
     - Tasks[].Name            非空（"text-to-image-synthesis" / "image-to-image"）
   再参考 widgets（平台可视化推理界面），但 ⚠️ **widgets 会随时间抖动**
   （同一模型先后返回过非空与 0），所以只作提示、不作判据。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 requests，请先执行：pip install requests")

DETAIL_API = "https://www.modelscope.cn/api/v1/models/{model_id}"
MODEL_PAGE = "https://www.modelscope.cn/models/{model_id}"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# 元数据里 Tasks[].Name -> 人类可读的任务名
TASK_CN = {
    "text-to-image-synthesis": "文生图",
    "image-to-image": "图像编辑",
}

# 已知的「图像编辑」类任务；用于把模型归到 image_edit_models 还是 image_models
EDIT_TASKS = {"image-to-image"}
GEN_TASKS = {"text-to-image-synthesis"}


def _subvision(raw) -> str:
    """AigcAttributes 有时是 dict，有时是 JSON 字符串，有时是空。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return raw or ""
    if isinstance(raw, dict):
        return raw.get("SubVisionFoundation") or ""
    return ""


def fetch_detail(model_id: str, timeout: int = 30) -> dict:
    """拉取单个模型的元数据。返回 dict，失败时带 error 字段。"""
    url = DETAIL_API.format(model_id=model_id)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return {"id": model_id, "exists": None, "error": f"{type(exc).__name__}: {exc}"}

    if resp.status_code != 200:
        return {"id": model_id, "exists": False, "error": f"HTTP {resp.status_code}"}

    try:
        payload = resp.json() or {}
    except Exception:  # noqa: BLE001
        return {"id": model_id, "exists": False, "error": "响应不是 JSON"}

    data = payload.get("Data")
    if not data:
        # Data 为空 = 模型不存在（接口仍返回 200 + Success:false）
        return {"id": model_id, "exists": False,
                "error": payload.get("Message") or "模型不存在"}

    tasks = [t.get("Name") for t in (data.get("Tasks") or []) if t.get("Name")]
    widgets = data.get("widgets")
    return {
        "id": model_id,
        "exists": True,
        "name": data.get("Name") or "",
        "chinese_name": data.get("ChineseName") or "",
        "aigc_type": data.get("AigcType") or "",
        "sub_vision": _subvision(data.get("AigcAttributes")),
        "support_inference": data.get("SupportInference") or "",
        "tasks": tasks,
        "widgets": len(widgets) if isinstance(widgets, list) else 0,
        "downloads": data.get("Downloads") or 0,
        "stars": data.get("Stars") or 0,
        "visibility": data.get("Visibility"),
        "base_model": data.get("BaseModel"),
        "error": None,
    }


# 走 OpenAI 兼容接口（/v1/chat/completions）的类别。这些模型的 AIGC 专区元数据
# （SupportInference / widgets）通常为空，但那不代表不能调用，所以不套用 AIGC 判据。
LLM_SOURCES = {
    "text_models", "vision_models", "default_text_model", "default_vision_model",
}
AIGC_SOURCES = {
    "image_models", "image_edit_models", "default_model", "lora_presets",
}


def judge(info: dict, source: str = "") -> tuple[str, str]:
    """返回 (等级, 说明)。等级：OK / WARN / BAD"""
    if info.get("exists") is False:
        return "BAD", f"模型不存在（{info.get('error')}）"
    if info.get("exists") is None:
        return "BAD", f"请求失败（{info.get('error')}）"

    # —— 文本 / 视觉模型：只校验存在性 ——
    # 它们的可调用性取决于账号权限，只能带 Token 实测，不能靠模型页元数据判断。
    if source in LLM_SOURCES:
        return "OK", "模型存在（LLM/VLM，走 OpenAI 兼容接口；可调用性需带 Token 实测）"

    has_infer = bool(info["support_inference"])
    has_task = bool(info["tasks"])
    kind = (info["aigc_type"] or "").upper()

    if info["visibility"] == 1:
        return "WARN", "非公开模型（Visibility=1），仅作者本人可用"

    if kind == "LORA":
        base = info["base_model"] or []
        base_s = "、".join(base) if base else "未登记"
        if has_task:
            return "OK", f"LoRA，底座 {base_s}；可当 model 传，也可作为 loras 传给底座"
        return "WARN", f"LoRA，底座 {base_s}；未登记推理任务"

    # 判据只用 SupportInference + Tasks 这两个**稳定**字段。
    # ⚠️ 不要拿 widgets 当判据：实测该字段会随时间抖动
    #（同一模型先后返回过非空与 0），会导致判定结果不稳定。
    if has_infer and has_task:
        cn = "、".join(TASK_CN.get(t, t) for t in info["tasks"])
        note = "，且配置了可视化推理界面" if info["widgets"] > 0 else ""
        return "OK", f"推理已登记（{cn}）{note}，可直接调用"

    if has_infer or has_task:
        got = []
        if has_infer:
            got.append(f"SupportInference={info['support_inference']}")
        if has_task:
            got.append("Tasks=" + "、".join(TASK_CN.get(t, t) for t in info["tasks"]))
        return "WARN", f"仅部分登记（{'；'.join(got)}），建议先用 custom_model 实测"

    if kind:
        return "WARN", (f"AigcType={kind}，但未登记任何推理任务"
                        "（SupportInference/Tasks 均空）。"
                        "可能是视频等其它模态，或元数据尚未补齐")
    return "WARN", ("未登记任何推理任务（SupportInference/Tasks 均空）。"
                    "可能是视频/其它模态模型，或元数据尚未补齐")


LEVEL_TAG = {"OK": "[可用]", "WARN": "[待验证]", "BAD": "[不可用]"}


def load_config_ids(config_path: str) -> list[tuple[str, str]]:
    """从 modelscope_config.json 收集 (模型ID, 来源字段)。"""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    pairs: list[tuple[str, str]] = []
    for key in ("image_models", "image_edit_models", "text_models", "vision_models"):
        for mid in cfg.get(key) or []:
            pairs.append((mid, key))
    for preset in cfg.get("lora_presets") or []:
        mid = (preset or {}).get("model_id")
        if mid:
            pairs.append((mid, "lora_presets"))
    for key in ("default_model", "default_text_model", "default_vision_model"):
        mid = cfg.get(key)
        if mid:
            pairs.append((mid, key))
    # 去重，保留首次出现
    seen, out = set(), []
    for mid, src in pairs:
        if mid not in seen:
            seen.add(mid)
            out.append((mid, src))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="核验 ModelScope 模型 ID")
    parser.add_argument("model_ids", nargs="*", help="要核验的模型 ID（不填则读配置）")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "modelscope_config.json"))
    parser.add_argument("--quiet", action="store_true", help="只输出结论行")
    parser.add_argument("--workers", type=int, default=8, help="并发数（默认 8）")
    args = parser.parse_args()

    if args.model_ids:
        targets = [(mid, "cli") for mid in args.model_ids]
    else:
        if not os.path.isfile(args.config):
            sys.exit(f"找不到配置文件：{args.config}")
        targets = load_config_ids(args.config)
        print(f"从 {os.path.basename(args.config)} 读取到 {len(targets)} 个模型 ID\n")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        details = list(pool.map(lambda p: fetch_detail(p[0]), targets))

    source_of = {mid: src for mid, src in targets}

    counts = {"OK": 0, "WARN": 0, "BAD": 0}
    rows = []
    for info in details:
        src = source_of.get(info["id"], "")
        level, reason = judge(info, src)
        counts[level] += 1
        rows.append((info, level, reason, src))

    for info, level, reason, src in rows:
        tag = LEVEL_TAG[level]
        head = f"{tag} {info['id']}"
        if not args.quiet:
            if info.get("exists"):
                extra = (f"AigcType={info['aigc_type'] or '-'}  "
                         f"SubVision={info['sub_vision'] or '-'}  "
                         f"Downloads={info['downloads']}  Visibility={info['visibility']}")
                print(head)
                print(f"        {extra}")
                print(f"        {reason}")
            else:
                print(head)
                print(f"        {reason}")
            if src:
                print(f"        来源: {src}")
            print()
        else:
            print(f"{head}  —— {reason}")

    print("=" * 72)
    print(f"合计 {len(rows)} 个：可用 {counts['OK']} / 待验证 {counts['WARN']} / 不可用 {counts['BAD']}")

    warn_aigc = [i["id"] for i, lv, _, src in rows
                 if lv == "WARN" and src not in LLM_SOURCES]
    if warn_aigc:
        print(f"\n图像相关但元数据不完整（{len(warn_aigc)} 个）——"
              "不一定是坏的，建议先用 custom_model 实测：")
        for mid in warn_aigc:
            print(f"  - {mid}")

    bad = [i["id"] for i, lv, _, _ in rows if lv == "BAD"]
    if bad:
        print("\n必须处理（死 ID，建议从配置中移除或修正）：")
        for mid in bad:
            print(f"  - {mid}")
            print(f"    页面: {MODEL_PAGE.format(model_id=mid)}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
