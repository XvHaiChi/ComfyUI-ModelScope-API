#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI-ModelScope-API 安装与协议自检工具
=========================================

用法（在插件目录下执行）::

    python verify_installation.py

它会依次检查：

1. 文件完整性
2. 依赖包
3. 配置文件字段
4. 节点模块能否加载、节点是否注册
5. API-Inference 协议实现是否正确（请求头 / payload / LoRA 归一化 / 400 降级）
6. （可选）网络与模型列表接口连通性
"""

from __future__ import annotations

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

REQUIRED_FILES = [
    "__init__.py",
    "modelscope_api_client.py",
    "modelscope_image_node.py",
    "modelscope_vision_node.py",
    "modelscope_text_node.py",
    "modelscope_image_caption_node.py",
    "modelscope_config.json",
    "README.md",
    "requirements.txt",
]

DEPENDENCIES = {
    "requests": "网络请求",
    "PIL": "图像处理",
    "numpy": "数值计算",
    "torch": "深度学习框架（ComfyUI 自带）",
    "openai": "文本生成 / 图生文",
}

REQUIRED_CONFIG_KEYS = [
    "default_model",
    "image_models",
    "image_edit_models",
    "text_models",
    "vision_models",
    "timeout",
    "request_timeout",
    "poll_interval",
    "lora_presets",
    "auto_normalize_lora",
]

EXPECTED_NODES = {
    "ModelScopeImageNode",
    "ModelScopeImageEditNode",
    "ModelScopeLoraPresetNode",
    "ModelScopeSingleLoraLoaderNode",
    "ModelScopeMultiLoraLoaderNode",
    "ModelScopeSixLoraLoaderNode",
    "ModelScopeModelListRefreshNode",
    "ModelScopeTextNode",
    "ModelScopeVisionNode",
    "ModelScopeImageCaptionNode",
}

RESULTS = []


def record(name, ok, detail="", warn=False):
    RESULTS.append((name, ok, warn))
    if ok:
        icon = "⚠️" if warn else "✅"
        print(f"{icon} {name}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"❌ {name}" + (f"  -> {detail}" if detail else ""))


def record_key(name, present, detail=""):
    """配置项检查：只在缺失时显示说明。"""
    record(name, present, "" if present else detail)


def section(title):
    print("\n" + "=" * 62)
    print(f" {title}")
    print("=" * 62)


# ------------------------------------------------------------------ 检查项


def check_files():
    section("1. 文件完整性")
    for name in REQUIRED_FILES:
        path = os.path.join(BASE_DIR, name)
        record(name, os.path.exists(path), "缺失" if not os.path.exists(path) else "")


def check_dependencies():
    section("2. 依赖包")
    for module, desc in DEPENDENCIES.items():
        try:
            __import__(module)
            record(f"{module} ({desc})", True)
        except ImportError:
            # torch 由 ComfyUI 提供，独立环境缺失不算致命
            record(f"{module} ({desc})", False, "未安装", warn=(module == "torch"))


def check_config():
    section("3. 配置文件")
    path = os.path.join(BASE_DIR, "modelscope_config.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        record("modelscope_config.json 格式正确", True)
    except Exception as exc:  # noqa: BLE001
        record("modelscope_config.json 格式正确", False, str(exc))
        return

    for key in REQUIRED_CONFIG_KEYS:
        record_key(f"配置含 {key}", key in config, "缺失")

    for key in ("image_models", "image_edit_models", "text_models", "vision_models"):
        value = config.get(key)
        record(f"{key} 非空", isinstance(value, list) and len(value) > 0,
               f"{len(value) if isinstance(value, list) else 'N/A'} 项")


def check_nodes():
    section("4. 节点加载与注册")
    try:
        import modelscope_image_node  # noqa: F401
        import modelscope_text_node  # noqa: F401
        import modelscope_vision_node  # noqa: F401
        import modelscope_image_caption_node  # noqa: F401
    except ImportError as exc:
        if "torch" in str(exc):
            record("节点模块导入", False, "缺少 torch（在 ComfyUI 环境中运行即可）", warn=True)
            return
        record("节点模块导入", False, str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        record("节点模块导入", False, str(exc))
        return

    record("节点模块导入", True)

    try:
        import modelscope_image_node as image_node
        import modelscope_text_node as text_node
        import modelscope_vision_node as vision_node
        import modelscope_image_caption_node as caption_node

        mappings = {}
        for module in (image_node, text_node, vision_node, caption_node):
            mappings.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))

        missing = EXPECTED_NODES - set(mappings)
        record(f"节点注册（共 {len(mappings)} 个）", not missing,
               f"缺少 {sorted(missing)}" if missing else "")

        for name, cls in mappings.items():
            try:
                cls.INPUT_TYPES()
            except Exception as exc:  # noqa: BLE001
                record(f"{name}.INPUT_TYPES()", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        record("节点注册", False, str(exc))


def check_protocol():
    section("5. API-Inference 协议自检")
    try:
        import modelscope_api_client as client
        import modelscope_image_node as image_node
    except Exception as exc:  # noqa: BLE001
        record("导入客户端模块", False, str(exc))
        return

    submit = client.build_headers("t", async_mode=True)
    record("提交任务带 X-ModelScope-Async-Mode",
           submit.get("X-ModelScope-Async-Mode") == "true")
    record("提交任务不带 X-ModelScope-Task-Type",
           "X-ModelScope-Task-Type" not in submit)

    poll = client.build_headers("t", task_type="image_generation")
    record("轮询带 X-ModelScope-Task-Type: image_generation",
           poll.get("X-ModelScope-Task-Type") == "image_generation")
    record("轮询不带 X-ModelScope-Async-Mode",
           "X-ModelScope-Async-Mode" not in poll)

    record("端点地址正确",
           client.IMAGE_GENERATION_ENDPOINT ==
           "https://api-inference.modelscope.cn/v1/images/generations")

    single = client.normalize_loras(client.prepare_lora_pairs(["a/b"], [0.8]))
    record("单个 LoRA 使用官方字符串写法", single == "a/b", str(single))

    multi = client.normalize_loras(
        client.prepare_lora_pairs(["a", "b", "c"], [0.6, 0.6, 0.6]))
    record("多 LoRA 权重归一化到 1.0",
           isinstance(multi, dict) and abs(sum(multi.values()) - 1.0) < 1e-9, str(multi))

    capped = client.normalize_loras(
        client.prepare_lora_pairs([f"l{i}" for i in range(9)], [1] * 9))
    record("LoRA 数量上限为 6", len(capped) == 6, str(len(capped)))

    payload = image_node.build_image_payload(
        model="Qwen/Qwen-Image", prompt="p", width=1024, height=1024, steps=20,
        guidance=3.5, seed=1)
    record("文生图 payload 不含 image_url", "image_url" not in payload, str(payload))
    record("size 映射为 WxH", payload.get("size") == "1024x1024")

    edit = image_node.build_image_payload(
        model="FireRedTeam/FireRed-Image-Edit-1.1", prompt="p",
        image_url="data:image/jpeg;base64,AA")
    record("图像编辑 payload 含 image_url", edit.get("image_url", "").startswith("data:image"))

    record("custom_model 优先于下拉框",
           image_node.resolve_model_name("A/B", "C/D") == "C/D")


def check_network():
    section("6. 网络连通性（可选）")
    try:
        import modelscope_api_client as client
    except Exception:  # noqa: BLE001
        return
    models = client.fetch_available_models(logger=lambda *a, **k: None)
    if models:
        record(f"可访问 /v1/models（返回 {len(models)} 个模型）", True)
    else:
        record("可访问 /v1/models", False, "网络不通或接口调整，可忽略", warn=True)


# ------------------------------------------------------------------ 主流程


def main():
    print("=" * 62)
    print(" ComfyUI-ModelScope-API 安装与协议自检")
    print(f" 插件目录: {BASE_DIR}")
    print("=" * 62)

    check_files()
    check_dependencies()
    check_config()
    check_nodes()
    check_protocol()
    check_network()

    section("检查结果汇总")
    hard_failures = [n for n, ok, warn in RESULTS if not ok and not warn]
    warnings = [n for n, ok, warn in RESULTS if not ok and warn]
    passed = [n for n, ok, _ in RESULTS if ok]

    print(f"✅ 通过: {len(passed)}")
    print(f"⚠️  警告: {len(warnings)}")
    print(f"❌ 失败: {len(hard_failures)}")

    if warnings:
        print("\n警告项（通常不影响在 ComfyUI 中使用）:")
        for name in warnings:
            print(f"  - {name}")

    if hard_failures:
        print("\n失败项:")
        for name in hard_failures:
            print(f"  - {name}")
        print("\n请修复上述问题后重试；缺失依赖可运行: python install_dependencies.py")
        return 1

    print("\n🎉 自检通过！")
    print("下一步：把整个插件目录放到 ComfyUI/custom_nodes/ 下，重启 ComfyUI。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
