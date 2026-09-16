#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_secrets.py —— 推送前的凭据体检。发现疑似密钥就以退出码 1 退出，可用于 pre-push 钩子。

为什么要专门做这个：
    本插件会把用户填的 ModelScope Token **写进 `modelscope_config.json`**
    （见 `modelscope_image_node.py` 的 `save_api_tokens()`），
    而 `modelscope_config.json` 是仓库的一部分（含默认模型列表），无法整体 gitignore。
    ⇒ 一旦在 ComfyUI 里填过 Token 再提交，Token 就进公开仓库了。

用法：
    python check_secrets.py              # 扫描工作区
    python check_secrets.py --history    # 额外扫描全部 git 历史
    python check_secrets.py --staged     # 只扫描已 git add 的内容（适合 pre-commit）

退出码：0 = 干净；1 = 发现疑似凭据。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# (名称, 正则, 说明)
PATTERNS = [
    ("ModelScope Token", re.compile(r"\bms-[A-Za-z0-9]{16,}"),
     "ModelScope 访问令牌，形如 ms-xxxxxxxx"),
    ("会话 Cookie", re.compile(r"m_session_id\s*[=:]\s*[\"']?[0-9a-fA-F\-]{20,}"),
     "登录会话 ID，等同于账号权限"),
    ("CSRF Token", re.compile(r"csrf_token\s*[=:]\s*[\"']?[A-Za-z0-9%+/=]{16,}"),
     "CSRF 令牌"),
    ("GitHub Token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "GitHub 个人访问令牌"),
    ("OpenAI Key", re.compile(r"\bsk-[A-Za-z0-9\-_]{20,}"), "OpenAI API Key"),
    ("Anthropic Key", re.compile(r"\bsk-ant-[A-Za-z0-9\-_]{20,}"), "Anthropic API Key"),
    ("阿里云 AK", re.compile(r"\bLTAI[A-Za-z0-9]{12,}"), "阿里云 AccessKey ID"),
    ("AWS AK", re.compile(r"\bAKIA[0-9A-Z]{16}"), "AWS Access Key ID"),
    ("私钥", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "PEM 私钥文件"),
]

SKIP_DIRS = {".git", "__pycache__", ".workbuddy-ai", "node_modules",
             ".venv", "venv", "dist", "build"}
SKIP_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".zip",
            ".safetensors", ".bin", ".pt", ".onnx", ".pdf"}
MAX_BYTES = 3_000_000

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def _mask(s: str) -> str:
    """打码显示，避免在终端/日志里二次泄露。"""
    if len(s) <= 12:
        return s[:4] + "*" * max(0, len(s) - 4)
    return f"{s[:6]}...{s[-4:]}（共 {len(s)} 字符）"


def _scan_text(text: str, source: str) -> list[tuple[str, str, str]]:
    hits = []
    for name, pat, desc in PATTERNS:
        for m in pat.finditer(text):
            hits.append((source, name, _mask(m.group(0))))
    return hits


def scan_worktree(root: str) -> tuple[list, int]:
    hits, scanned = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                continue
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read()
            except Exception:  # noqa: BLE001
                continue
            scanned += 1
            rel = os.path.relpath(path, root)
            hits.extend(_scan_text(text, rel))
    return hits, scanned


def check_config_fields(root: str) -> list[str]:
    """专门检查 modelscope_config.json 里那两个 Token 字段。"""
    warnings = []
    path = os.path.join(root, "modelscope_config.json")
    if not os.path.isfile(path):
        return warnings
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        return [f"modelscope_config.json 解析失败：{exc}"]

    single = (cfg.get("api_token") or "").strip() if isinstance(cfg.get("api_token"), str) else ""
    if single:
        warnings.append(f"modelscope_config.json 的 api_token 非空 → {_mask(single)}")

    many = cfg.get("api_tokens")
    if isinstance(many, list):
        real = [t for t in many if isinstance(t, str) and t.strip()]
        if real:
            warnings.append(f"modelscope_config.json 的 api_tokens 有 {len(real)} 个非空项 "
                            f"→ {_mask(real[0])}")
    return warnings


def scan_history(root: str) -> list[tuple[str, str, str]]:
    """扫描全部 git 历史对象。"""
    try:
        out = subprocess.run(["git", "rev-list", "--objects", "--all"],
                             cwd=root, capture_output=True, text=True, timeout=60)
    except Exception as exc:  # noqa: BLE001
        print(f"{YELLOW}跳过历史扫描：{exc}{RESET}")
        return []

    hits = []
    paths = []
    for line in out.stdout.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            paths.append(parts[1])

    for p in paths:
        if os.path.splitext(p)[1].lower() in SKIP_EXT:
            continue
        try:
            blob = subprocess.run(["git", "show", f"HEAD:{p}"], cwd=root,
                                  capture_output=True, timeout=15)
        except Exception:  # noqa: BLE001
            continue
        text = blob.stdout.decode("utf-8", errors="ignore")
        if text:
            hits.extend(_scan_text(text, f"[历史] {p}"))
    return hits


def scan_staged(root: str) -> list[tuple[str, str, str]]:
    """只扫描已 git add 的内容（pre-commit 场景）。"""
    try:
        names = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                               cwd=root, capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:  # noqa: BLE001
        print(f"{YELLOW}无法读取暂存区：{exc}{RESET}")
        return []

    hits = []
    for rel in [n for n in names.splitlines() if n.strip()]:
        if os.path.splitext(rel)[1].lower() in SKIP_EXT:
            continue
        try:
            blob = subprocess.run(["git", "show", f":{rel}"], cwd=root,
                                  capture_output=True, timeout=15)
        except Exception:  # noqa: BLE001
            continue
        text = blob.stdout.decode("utf-8", errors="ignore")
        if text:
            hits.extend(_scan_text(text, f"[暂存] {rel}"))
    return hits


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="推送前的凭据体检")
    ap.add_argument("--history", action="store_true", help="额外扫描全部 git 历史")
    ap.add_argument("--staged", action="store_true", help="只扫描已 git add 的内容")
    ap.add_argument("--root", default=here, help="项目根目录")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    all_hits: list[tuple[str, str, str]] = []

    if args.staged:
        print("扫描暂存区...")
        all_hits += scan_staged(root)
    else:
        print("扫描工作区...")
        hits, n = scan_worktree(root)
        print(f"  已检查 {n} 个文件")
        all_hits += hits
        if args.history:
            print("扫描 git 历史...")
            all_hits += scan_history(root)

    print("\n检查 modelscope_config.json 的 Token 字段...")
    cfg_warnings = check_config_fields(root)
    for w in cfg_warnings:
        print(f"  {RED}✗{RESET} {w}")
    if not cfg_warnings:
        print(f"  {GREEN}✓{RESET} api_token / api_tokens 均为空")

    print("\n" + "=" * 70)
    if all_hits:
        print(f"{RED}发现 {len(all_hits)} 处疑似凭据：{RESET}")
        for src, name, masked in all_hits:
            print(f"  [{name}] {src}")
            print(f"      {masked}")
    else:
        print(f"{GREEN}✓ 未在任何文件中发现疑似凭据{RESET}")

    if all_hits or cfg_warnings:
        print(f"\n{RED}⛔ 请先清理干净再推送！{RESET}")
        print("提示：若凭据已经提交过，仅删除文件是不够的，历史里仍然可查。")
        print("     需要改写历史（git filter-repo / BFG）并**立刻到平台吊销该凭据**。")
        return 1

    print(f"\n{GREEN}✓ 可以安全推送{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
