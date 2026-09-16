#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search_models.py —— 免登录搜索 / 枚举魔搭全站模型，并统计出「有哪些图像模型可用」。

核心接口（关键：**只接受 PUT**，且**完全不需要登录**）：
    PUT https://modelscope.cn/api/v1/dolphin/models
    Body: {"PageSize":100,"PageNumber":1,
           "Criterion":[{"category":"tags","predicate":"contains","values":["text-to-image"]}]}
    Resp: Data.Model.Models[] (每条 103 个字段) + Data.Model.TotalCount + Data.FiledAgg

⚠️ 踩坑记录（别再重复）：
  - 这个路由**只吃 PUT**。用 GET / POST 都会 404 或报错，
    很容易误判成「路由不存在」而放弃 —— 实际它是公开的。
  - Criterion 的键名是**全小写**：`category` / `predicate` / `values`。
  - `predicate` 实测只有 `contains` 有效；`eq` / `in` 会被忽略（返回全量）。
  - 域名 `modelscope.cn` 和 `www.modelscope.cn` 都可用。

可用的 category（值取自 `--facets` 的 FiledAgg）：
    tags, aigc_type, sub_vision_foundation, vision_foundation,
    model_type, libraries, language, license, nexa_catalog

用法：
    python search_models.py                      # 总览统计（全站 / AIGC / 图像）
    python search_models.py --facets             # 列出所有可筛选维度及取值
    python search_models.py --image-catalog      # 图像模型按底座分布
    python search_models.py --tag text-to-image  # 按标签筛
    python search_models.py --foundation KREA_2_TURBO --limit 30
    python search_models.py --aigc-type Checkpoint --limit 30
    python search_models.py --query "qwen image" --limit 20
    python search_models.py --export all_image.json
    python search_models.py --config-diff        # 对比 modelscope_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 requests，请先执行：pip install requests")

ENDPOINTS = [
    "https://modelscope.cn/api/v1/dolphin/models",
    "https://www.modelscope.cn/api/v1/dolphin/models",
]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Origin": "https://modelscope.cn",
    "Referer": "https://modelscope.cn/aigc/models",
}

MAX_PAGE_SIZE = 100


# ---------------------------------------------------------------- 底层调用

def _put(body: dict, timeout: int = 60) -> dict:
    """调用搜索接口，自动尝试两个域名。"""
    last_err = ""
    for url in ENDPOINTS:
        try:
            resp = requests.put(url, headers=HEADERS, json=body, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            continue
        if resp.status_code != 200:
            last_err = f"HTTP {resp.status_code}: {resp.text[:120]}"
            continue
        try:
            return resp.json() or {}
        except Exception as exc:  # noqa: BLE001
            last_err = f"响应不是 JSON：{exc}"
    raise RuntimeError(f"搜索接口调用失败 —— {last_err}")


def _unpack(payload: dict) -> tuple[list[dict], int, dict]:
    data = payload.get("Data") or {}
    block = data.get("Model") or {}
    return (block.get("Models") or [], block.get("TotalCount") or 0,
            data.get("FiledAgg") or {})


def count(criterion: list[dict] | None = None, name: str = "") -> int:
    """只取数量，不拉列表。"""
    body: dict = {"PageSize": 1, "PageNumber": 1}
    if criterion:
        body["Criterion"] = criterion
    if name:
        body["Name"] = name
    return _unpack(_put(body))[1]


def search(criterion: list[dict] | None = None, name: str = "",
           limit: int = 20, page_size: int = MAX_PAGE_SIZE) -> tuple[list[dict], int]:
    """拉取模型列表（自动翻页到 limit 条）。"""
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    out: list[dict] = []
    page = 1
    total = 0
    while len(out) < limit:
        body: dict = {"PageSize": page_size, "PageNumber": page}
        if criterion:
            body["Criterion"] = criterion
        if name:
            body["Name"] = name
        models, total, _ = _unpack(_put(body))
        if not models:
            break
        out.extend(models)
        if len(out) >= total:
            break
        page += 1
        if page > 200:  # 安全阀
            break
    return out[:limit], total


def crit(category: str, value: str, predicate: str = "contains") -> list[dict]:
    return [{"category": category, "predicate": predicate, "values": [value]}]


def facets() -> dict:
    """列出所有可筛选维度及其取值。"""
    _, _, agg = _unpack(_put({"PageSize": 1, "PageNumber": 1}))
    return agg


# ---------------------------------------------------------------- 展示

def model_id(m: dict) -> str:
    return f"{m.get('Path', '')}/{m.get('Name', '')}"


def subvision(m: dict) -> str:
    raw = m.get("AigcAttributes")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return raw or ""
    if isinstance(raw, dict):
        return raw.get("SubVisionFoundation") or ""
    return ""


def print_models(models: list[dict], total: int, show_fields: bool = False) -> None:
    print(f"（共 {total:,} 条，显示 {len(models)} 条）")
    for m in models:
        mid = model_id(m)
        kind = m.get("AigcType") or "-"
        sv = subvision(m) or "-"
        dl = m.get("Downloads") or 0
        print(f"  {mid}")
        print(f"      AigcType={kind:<12} SubVision={sv:<24} "
              f"Downloads={dl:<8} License={m.get('License') or '-'}")
        if show_fields:
            tasks = [t.get("Name") for t in (m.get("Tasks") or []) if t.get("Name")]
            print(f"      SupportInference={m.get('SupportInference')!r}  Tasks={tasks}")
            base = m.get("BaseModel") or []
            if base:
                print(f"      BaseModel={base}")


# ---------------------------------------------------------------- 统计

# 图像模型常见的 SubVisionFoundation 底座（用于分组统计）
IMAGE_FOUNDATIONS = [
    "QWEN_IMAGE_20_B", "QWEN_IMAGE_2512", "QWEN_IMAGE_EDIT_2509",
    "QWEN_IMAGE_EDIT_2511", "QWEN_IMAGE_EDIT", "KREA_2_TURBO", "KREA_2_RAW",
    "Z_IMAGE", "SD_XL", "SD_15", "FLUX", "HIDREAM_O1_IMAGE", "MAJICFLUS",
    "KOLORS", "HUNYUAN_IMAGE", "WAN", "MINIMAX_H3",
]


def show_stats() -> None:
    base = count()
    print("=" * 74)
    print(f"魔搭全站模型总数：{base:,}")
    print("=" * 74)

    print("\n【按 AigcType 分类】（AIGC 专区模型才有该字段）")
    aigc_total = 0
    rows = [("LoRA", "LoRA（微调/风格模型）"), ("Checkpoint", "Checkpoint（基座）"),
            ("VAE", "VAE"), ("ControlNet", "ControlNet"), ("Upscaler", "Upscaler")]
    for value, label in rows:
        n = count(crit("aigc_type", value))
        if n:
            aigc_total += n
            print(f"  {label:<28} {n:>9,}")
    print(f"  {'AIGC 类合计':<28} {aigc_total:>9,}")

    print("\n【按图像相关标签】")
    for tag in ["text-to-image", "image-to-image", "stable-diffusion",
                "flux", "diffusers", "krea-2"]:
        print(f"  tags ∋ {tag:<22} {count(crit('tags', tag)):>9,}")

    print("\n【非图像模态（供排除）】")
    for tag in ["text-to-video", "image-to-video", "text-generation",
                "automatic-speech-recognition"]:
        print(f"  tags ∋ {tag:<22} {count(crit('tags', tag)):>9,}")


def show_image_catalog() -> None:
    print("=" * 74)
    print("图像模型按底座（SubVisionFoundation）分布")
    print("=" * 74)
    print(f"{'底座':<28}{'LoRA':>10}{'Checkpoint':>12}{'合计':>10}")
    print("-" * 74)

    def one(f):
        return (f,
                count(crit("sub_vision_foundation", f) + crit("aigc_type", "LoRA")),
                count(crit("sub_vision_foundation", f) + crit("aigc_type", "Checkpoint")))

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(one, IMAGE_FOUNDATIONS))

    tot = 0
    for f, lora, ckpt in results:
        s = lora + ckpt
        tot += s
        if s:
            print(f"{f:<28}{lora:>10,}{ckpt:>12,}{s:>10,}")
    print("-" * 74)
    print(f"{'合计':<28}{'':>10}{'':>12}{tot:>10,}")


def show_facets() -> None:
    agg = facets()
    for cat, items in agg.items():
        if not isinstance(items, list):
            continue
        print(f"\n### {cat}  （{len(items)} 个取值）")
        for it in items:
            print(f"   {str(it.get('Value')):<34} {it.get('Count'):>9,}")


# ---------------------------------------------------------------- 可 API 调用的图像模型

# 图像生成/编辑类任务名
IMAGE_TASKS = {"text-to-image-synthesis", "image-to-image"}


def is_api_image_model(m: dict) -> bool:
    """判断该 Checkpoint 是否可用于 API-Inference 的图像生成/编辑。

    判据：SupportInference 是 txt2img/img2img，或 Tasks 里有图像类任务。
    ⚠️ 不要看 SupportApiInference（对 Qwen/Qwen-Image 这种主力模型也是 false），
       也不要看 widgets（该字段会随时间抖动）。
    """
    si = (m.get("SupportInference") or "").lower()
    if si in ("txt2img", "img2img"):
        return True
    tasks = {t.get("Name") for t in (m.get("Tasks") or []) if t.get("Name")}
    return bool(tasks & IMAGE_TASKS)


def show_api_image_models(as_json: str | None = None) -> None:
    """枚举全部 Checkpoint，筛出可 API 调用的图像模型。"""
    print("正在枚举全部 Checkpoint（可能需要十几秒）...\n")
    models, total = search(crit("aigc_type", "Checkpoint"),
                           limit=2000, page_size=MAX_PAGE_SIZE)

    usable, skipped = [], []
    for m in models:
        (usable if is_api_image_model(m) else skipped).append(m)

    # 统计被排除的都是些什么模态
    def why(m: dict) -> str:
        tasks = [t.get("Name") for t in (m.get("Tasks") or []) if t.get("Name")]
        if tasks:
            return tasks[0]
        return "无推理任务登记"

    from collections import Counter
    reasons = Counter(why(m) for m in skipped)

    print("=" * 78)
    print(f"Checkpoint 总数 {total:,}  →  可 API 调用的图像模型 {len(usable)} 个")
    print("=" * 78)

    usable.sort(key=lambda m: m.get("Downloads") or 0, reverse=True)
    print(f"\n{'模型 ID':<50}{'类型':<10}{'下载':>9}")
    print("-" * 78)
    for m in usable:
        kind = "编辑" if (m.get("SupportInference") or "").lower() == "img2img" else "文生图"
        print(f"{model_id(m):<50}{kind:<10}{m.get('Downloads') or 0:>9,}")

    print("\n" + "-" * 78)
    print(f"被排除的 {len(skipped)} 个（不是图像生成模型）：")
    for reason, n in reasons.most_common():
        print(f"   {reason:<34} {n:>5} 个")

    if as_json:
        with open(as_json, "w", encoding="utf-8") as fh:
            json.dump({"checkpoint_total": total,
                       "api_image_models": [model_id(m) for m in usable],
                       "excluded": [{"id": model_id(m), "reason": why(m)} for m in skipped]},
                      fh, ensure_ascii=False, indent=2)
        print(f"\n✓ 已导出到 {as_json}")


# ---------------------------------------------------------------- 配置对比

def config_diff(config_path: str) -> None:
    if not os.path.isfile(config_path):
        print(f"找不到配置文件：{config_path}")
        return
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    known: set[str] = set()
    for key in ("image_models", "image_edit_models", "text_models", "vision_models"):
        known.update(cfg.get(key) or [])
    known.update(p.get("model_id") for p in (cfg.get("lora_presets") or [])
                 if p.get("model_id"))

    print(f"配置里已有 {len(known)} 个模型 ID")
    print("\n【下载量最高的 Checkpoint 基座（可作为 image_models 候选）】")
    models, total = search(crit("aigc_type", "Checkpoint"), limit=60)
    models.sort(key=lambda m: m.get("Downloads") or 0, reverse=True)
    for m in models[:40]:
        mid = model_id(m)
        mark = "  ✓已有" if mid in known else ""
        print(f"  {mid:<52} ↓{m.get('Downloads') or 0:<8}"
              f" {subvision(m) or '-':<20}{mark}")


# ---------------------------------------------------------------- main

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="免登录搜索/枚举魔搭模型")
    ap.add_argument("--stats", action="store_true", help="总览统计（默认）")
    ap.add_argument("--facets", action="store_true", help="列出所有可筛选维度")
    ap.add_argument("--image-catalog", action="store_true", help="图像模型按底座分布")
    ap.add_argument("--api-image", action="store_true",
                    help="枚举全部 Checkpoint，筛出可 API 调用的图像模型")
    ap.add_argument("--tag", help="按 tags 筛选")
    ap.add_argument("--foundation", help="按 sub_vision_foundation 筛选")
    ap.add_argument("--aigc-type", dest="aigc_type", help="按 aigc_type 筛选（LoRA/Checkpoint/...）")
    ap.add_argument("--query", help="按名称搜索")
    ap.add_argument("--limit", type=int, default=20, help="列出条数（默认 20）")
    ap.add_argument("--show-fields", action="store_true", help="显示推理相关字段")
    ap.add_argument("--export", help="把结果导出为 JSON")
    ap.add_argument("--config", default=os.path.join(here, "modelscope_config.json"))
    ap.add_argument("--config-diff", action="store_true",
                    help="对比配置，列出可新增的候选模型")
    args = ap.parse_args()

    try:
        if args.facets:
            show_facets()
            return 0

        if args.image_catalog:
            show_image_catalog()
            return 0

        if args.api_image:
            show_api_image_models(args.export)
            return 0

        if args.config_diff:
            config_diff(args.config)
            return 0

        criterion: list[dict] = []
        if args.tag:
            criterion += crit("tags", args.tag)
        if args.foundation:
            criterion += crit("sub_vision_foundation", args.foundation)
        if args.aigc_type:
            criterion += crit("aigc_type", args.aigc_type)

        if criterion or args.query or args.export:
            models, total = search(criterion or None, name=args.query or "",
                                   limit=max(args.limit, 1))
            label = []
            if args.tag:
                label.append(f"tag={args.tag}")
            if args.foundation:
                label.append(f"foundation={args.foundation}")
            if args.aigc_type:
                label.append(f"aigc_type={args.aigc_type}")
            if args.query:
                label.append(f"name={args.query}")
            print(f"筛选条件：{', '.join(label) or '（无）'}")
            print_models(models, total, show_fields=args.show_fields)

            if args.export:
                with open(args.export, "w", encoding="utf-8") as fh:
                    json.dump({"criterion": criterion, "total": total,
                               "models": [model_id(m) for m in models],
                               "raw": models}, fh, ensure_ascii=False, indent=2)
                print(f"\n✓ 已导出到 {args.export}")
            return 0

        show_stats()
        return 0

    except RuntimeError as exc:
        print(f"✗ {exc}")
        print("\n提示：该接口必须用 PUT；如果返回 404/405，检查是否被网络策略拦截。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
