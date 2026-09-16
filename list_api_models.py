#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
list_api_models.py —— 拉取「你的 ModelScope 账号到底能用哪些模型」的完整清单。

背景（为什么要用 Token）：
    这里查的是「**你的账号**能调用哪些模型」，走的是推理侧的 `api-inference/v1/models`，
    它需要 Token 鉴权。
    ⚠️ 注意区分两件事：
      - 「**全站有哪些模型**」→ 免登录即可，用 `search_models.py`
        （`PUT https://modelscope.cn/api/v1/dolphin/models`，只吃 PUT）。
        实测全站 254,310 个模型，可 API 调用的图像模型 431 个。
      - 「**我的账号能用哪些**」→ 需要 Token，就是本脚本。
    ⚠️ `GET https://api-inference.modelscope.cn/v1/models` 在**未鉴权或 Token 无效**时，
    返回的是**固定 37 个公共样例模型**，既不报错也不返回空，极易误判成「账号有 37 个模型」。

用法：
    # 1) Token 来源（优先级从高到低）
    python list_api_models.py --token ms-xxxxxxxx
    set MODELSCOPE_API_TOKEN=ms-xxxxxxxx && python list_api_models.py
    # 或者先写进 modelscope_config.json 的 api_token / api_tokens 字段

    # 2) 常用参数
    python list_api_models.py                 # 拉取并打印统计 + 清单
    python list_api_models.py --diff          # 额外列出「可用但配置里没有」的模型（候选新增）
    python list_api_models.py --write         # 把新发现的图像模型合并进 modelscope_config.json
    python list_api_models.py --json out.json # 导出完整清单
    python list_api_models.py --source hub    # 只查 AIGC 专区清单（muse/models）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

try:
    import requests
except ImportError:  # pragma: no cover
    sys.exit("缺少依赖 requests，请先执行：pip install requests")

INFERENCE_MODELS = "https://api-inference.modelscope.cn/v1/models"
HUB_AIGC_MODELS = "https://www.modelscope.cn/api/v1/muse/models"
HUB_HOMEPAGE = "https://www.modelscope.cn/api/v1/dolphin/agg/homepage"
CONFIG_NAME = "modelscope_config.json"

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


# ---------------------------------------------------------------- Token 解析

def resolve_token(cli_token: str | None, config_path: str) -> str:
    """按 CLI → 环境变量 → 配置文件的顺序找 Token。"""
    if cli_token:
        return cli_token.strip()

    for env_key in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "MS_TOKEN"):
        val = os.environ.get(env_key)
        if val:
            print(f"[i] 使用环境变量 {env_key} 中的 Token")
            return val.strip()

    if os.path.isfile(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except Exception:  # noqa: BLE001
            cfg = {}
        single = (cfg.get("api_token") or "").strip()
        if single:
            print(f"[i] 使用 {CONFIG_NAME} 中 api_token 字段的 Token")
            return single
        many = cfg.get("api_tokens") or []
        if many and isinstance(many[0], str) and many[0].strip():
            print(f"[i] 使用 {CONFIG_NAME} 中 api_tokens[0] 的 Token")
            return many[0].strip()

    return ""


# ---------------------------------------------------------------- 拉取

def _extract_ids(payload) -> list[str]:
    """从各种可能的响应结构里把模型 ID 抠出来。"""
    if not isinstance(payload, (dict, list)):
        return []

    # OpenAI 风格 {"data":[{"id":...}]}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        out = []
        for it in payload["data"]:
            if isinstance(it, dict) and it.get("id"):
                out.append(str(it["id"]))
            elif isinstance(it, str):
                out.append(it)
        if out:
            return out

    # 魔搭风格 {"Data":{"Models":[...]}}
    ids: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            for key in ("Id", "ModelId", "ModelID", "Name", "Path", "ModelName"):
                v = node.get(key)
                if isinstance(v, str) and "/" in v:
                    ids.append(v)
                    return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    return ids


def fetch_inference_models(token: str, timeout: int = 60) -> tuple[list[str], str]:
    """GET api-inference /v1/models —— 权威的「我能调哪些模型」。"""
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(INFERENCE_MODELS, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        return [], f"请求失败：{type(exc).__name__}: {exc}"
    if resp.status_code == 401:
        return [], "401 未授权 —— Token 无效或已过期"
    if resp.status_code != 200:
        return [], f"HTTP {resp.status_code}: {resp.text[:150]}"
    try:
        return _extract_ids(resp.json()), ""
    except Exception as exc:  # noqa: BLE001
        return [], f"响应不是 JSON：{exc}"


def fetch_hub_aigc_models(token: str, page_size: int = 100,
                          max_pages: int = 30, timeout: int = 60) -> tuple[list[str], str]:
    """GET hub /api/v1/muse/models —— AIGC 专区清单（需登录）。"""
    headers = dict(UA)
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Cookie"] = f"m_session_id={token}"
    headers["Referer"] = "https://www.modelscope.cn/aigc/models"

    all_ids: list[str] = []
    for page in range(1, max_pages + 1):
        params = {"PageSize": page_size, "PageNumber": page}
        try:
            resp = requests.get(HUB_AIGC_MODELS, headers=headers,
                                params=params, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return all_ids, f"请求失败：{type(exc).__name__}: {exc}"
        if resp.status_code == 401:
            return all_ids, "401 用户未登录 —— 需要登录态（Token 或 m_session_id Cookie）"
        if resp.status_code != 200:
            return all_ids, f"HTTP {resp.status_code}: {resp.text[:150]}"
        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return all_ids, f"响应不是 JSON：{exc}"

        batch = _extract_ids(payload)
        if not batch:
            break
        new = [i for i in batch if i not in all_ids]
        all_ids.extend(new)
        if len(new) == 0:
            break
    return all_ids, ""


def fetch_platform_total(timeout: int = 30) -> int | None:
    """免登录读全站模型总数（首页聚合接口）。"""
    try:
        resp = requests.get(HUB_HOMEPAGE, headers=UA, timeout=timeout)
        payload = resp.json()
        return (((payload.get("Data") or {}).get("Data") or {})
                .get("Model") or {}).get("TotalCount")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- 分类

EDIT_HINT = re.compile(r"edit|kontext|inpaint|\\bimg2img\\b", re.I)
GEN_HINT = re.compile(r"image|flux|qwen-image|z-image|krea|hidream|sdxl|sd3|"
                      r"stable-diffusion|kolors|hunyuan.*image|wan", re.I)


def classify(ids: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"image_edit": [], "image_gen": [], "other": []}
    for mid in ids:
        name = mid.split("/")[-1]
        if EDIT_HINT.search(name):
            buckets["image_edit"].append(mid)
        elif GEN_HINT.search(name):
            buckets["image_gen"].append(mid)
        else:
            buckets["other"].append(mid)
    return buckets


# ---------------------------------------------------------------- 配置

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def config_ids(cfg: dict) -> set[str]:
    out: set[str] = set()
    for key in ("image_models", "image_edit_models", "text_models", "vision_models"):
        out.update(cfg.get(key) or [])
    out.update(p.get("model_id") for p in (cfg.get("lora_presets") or []) if p.get("model_id"))
    return {x for x in out if x}


def write_config(path: str, cfg: dict, add_gen: list[str], add_edit: list[str]) -> None:
    cfg.setdefault("image_models", []).extend(add_gen)
    cfg.setdefault("image_edit_models", []).extend(add_edit)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


# ---------------------------------------------------------------- main

def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="拉取账号可用的 ModelScope 模型清单")
    ap.add_argument("--token", help="ModelScope 访问令牌（不填则读环境变量或配置）")
    ap.add_argument("--config", default=os.path.join(here, CONFIG_NAME))
    ap.add_argument("--source", choices=["all", "inference", "hub"], default="all",
                    help="all=两边都查（默认）；inference=只查 API-Inference；hub=只查 AIGC 专区")
    ap.add_argument("--diff", action="store_true", help="列出「可用但配置里没有」的模型")
    ap.add_argument("--write", action="store_true",
                    help="把新发现的图像模型合并进配置（仅追加，不动既有项）")
    ap.add_argument("--json", dest="json_out", help="把完整清单导出为 JSON 文件")
    ap.add_argument("--max-pages", type=int, default=30, help="AIGC 清单最多翻多少页")
    args = ap.parse_args()

    total = fetch_platform_total()
    if total is not None:
        print(f"[i] 魔搭全站模型总数（免登录可读）：{total:,}")
    print()

    token = resolve_token(args.token, args.config)
    if not token:
        print("⚠️  没有找到 Token。")
        print("   列表接口必须登录才能访问，所以只能告诉你全站总数。")
        print("   请用以下任一方式提供 Token 后重跑：")
        print("     python list_api_models.py --token ms-xxxxxxxx")
        print("     设置环境变量 MODELSCOPE_API_TOKEN=ms-xxxxxxxx")
        print(f"     或写入 {CONFIG_NAME} 的 api_token 字段")
        print()
        print("   Token 获取：https://www.modelscope.cn/my/myaccesstoken")
        return 2

    collected: list[str] = []
    problems: list[str] = []

    if args.source in ("all", "inference"):
        ids, err = fetch_inference_models(token)
        if err:
            problems.append(f"API-Inference /v1/models → {err}")
            print(f"✗ API-Inference 清单拉取失败：{err}")
        else:
            print(f"✓ API-Inference 可用模型：{len(ids)} 个")
            collected.extend(ids)

    if args.source in ("all", "hub"):
        ids, err = fetch_hub_aigc_models(token, max_pages=args.max_pages)
        if err:
            problems.append(f"AIGC 专区 muse/models → {err}")
            print(f"✗ AIGC 专区清单拉取失败：{err}")
        else:
            print(f"✓ AIGC 专区模型：{len(ids)} 个")
            collected.extend(ids)

    uniq = sorted(set(collected))
    print()
    print("=" * 72)
    print(f"去重后共 {len(uniq)} 个模型")
    if problems:
        print("\n未成功的来源：")
        for p in problems:
            print(f"  - {p}")

    if not uniq:
        return 1

    buckets = classify(uniq)
    print(f"\n粗分类（按名字猜测，仅供筛选参考）：")
    print(f"  图像编辑  : {len(buckets['image_edit'])}")
    print(f"  图像生成  : {len(buckets['image_gen'])}")
    print(f"  其它      : {len(buckets['other'])}")

    if args.diff or args.write:
        if not os.path.isfile(args.config):
            print(f"\n✗ 找不到配置文件：{args.config}")
            return 1
        cfg = load_config(args.config)
        known = config_ids(cfg)
        new_gen = [m for m in buckets["image_gen"] if m not in known]
        new_edit = [m for m in buckets["image_edit"] if m not in known]

        print(f"\n配置里已有 {len(known)} 个模型 ID")
        print(f"可新增：图像生成 {len(new_gen)} 个 / 图像编辑 {len(new_edit)} 个")

        if args.diff:
            if new_gen:
                print("\n【候选新增 · 图像生成】")
                for m in new_gen[:60]:
                    print(f"  + {m}")
                if len(new_gen) > 60:
                    print(f"  ...（还有 {len(new_gen) - 60} 个）")
            if new_edit:
                print("\n【候选新增 · 图像编辑】")
                for m in new_edit[:60]:
                    print(f"  + {m}")
                if len(new_edit) > 60:
                    print(f"  ...（还有 {len(new_edit) - 60} 个）")

        if args.write:
            if not new_gen and not new_edit:
                print("\n没有需要新增的模型。")
            else:
                write_config(args.config, cfg, new_gen, new_edit)
                print(f"\n✓ 已写回 {os.path.basename(args.config)}"
                      f"（image_models +{len(new_gen)}，image_edit_models +{len(new_edit)}）")
                print("  建议接着跑：python check_model_ids.py   # 核验新增 ID")
                print("            python test_workflow_compat.py  # 确认旧工作流仍对齐")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"total_platform": total, "count": len(uniq),
                       "models": uniq, "buckets": buckets}, fh,
                      ensure_ascii=False, indent=2)
        print(f"\n✓ 完整清单已导出到 {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
