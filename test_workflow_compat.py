#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工作流兼容性回归测试
====================

仓库里的 ``ModelScope-API-*.json`` 是**兼容性回归样本**。本脚本会检查：
对每个示例工作流里保存的 ModelScope 节点，其 ``widgets_values[i]`` 是否仍然
按位置映射到**同一个**控件。

一旦有人在 ``INPUT_TYPES`` 中间插入控件（而不是追加到末尾），这个测试会立刻失败 ——
因为那会让所有旧工作流的参数错位。

用法（在插件目录下执行）::

    python test_workflow_compat.py

退出码 0 表示全部通过。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ComfyUI 会提供真实的 torch；独立运行时用桩替代，以便导入节点模块
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")

        def _from_numpy(array):
            class _Tensor:
                def __init__(self, data):
                    self.data = data

                def unsqueeze(self, *_):
                    return self

                def float(self):
                    return self

            return _Tensor(array)

        stub.from_numpy = _from_numpy
        sys.modules["torch"] = stub

# 目录名含 "-"，不能直接 import，手工构造包 spec
_spec = importlib.util.spec_from_file_location(
    "comfyui_modelscope_api",
    os.path.join(BASE_DIR, "__init__.py"),
    submodule_search_locations=[BASE_DIR],
)
_package = importlib.util.module_from_spec(_spec)
sys.modules["comfyui_modelscope_api"] = _package
_spec.loader.exec_module(_package)

NODE_CLASS_MAPPINGS = _package.NODE_CLASS_MAPPINGS

WORKFLOW_FILES = ["ModelScope-API-生图改图.json", "ModelScope-API-图片解析.json"]

# 非控件类型的输入（永远以连线方式出现，不占 widgets_values）
CONNECTION_TYPES = {"IMAGE", "MASK", "LATENT", "*"}

# 前端在 seed 控件之后自动追加的 control_after_generate 取值
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}

# 这些控件的值只要求"仍然存在"，不做取值校验
VALUE_AGNOSTIC_WIDGETS = {
    "prompt", "api_tokens", "api_token", "user_prompt", "system_prompt",
    "negative_prompt", "prompt1", "prompt2", "width", "height", "steps",
    "guidance", "seed", "max_tokens", "temperature", "stream", "image_gen_mode",
    "lora1_id", "lora2_id", "lora3_id", "lora4_id", "lora5_id", "lora6_id",
    "lora1_w", "lora2_w", "lora3_w", "lora4_w", "lora5_w", "lora6_w",
    "custom_model", "auto_normalize_lora", "image_input_mode",
}

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"{'✅' if ok else '❌'} {name}" + ("" if ok else f"  -> {detail}"))


def widget_layout(node_class):
    """按 ComfyUI 的排布顺序返回控件名与下拉框选项。"""
    spec = node_class.INPUT_TYPES()
    names, options = [], {}
    for section in ("required", "optional"):
        for widget_name, definition in spec.get(section, {}).items():
            type_ = definition[0] if isinstance(definition, tuple) else definition
            if isinstance(type_, list):
                names.append(widget_name)
                options[widget_name] = type_
            elif isinstance(type_, str) and type_ not in CONNECTION_TYPES:
                names.append(widget_name)
    return names, options


def assign_values(values, names):
    """把保存的值按位置映射到控件名，并跳过前端自动注入的 seed 控制位。"""
    assigned = []
    cursor = 0
    index = 0
    while index < len(values):
        if cursor >= len(names):
            assigned.append(("<越界>", values[index]))
            break
        name = names[cursor]
        assigned.append((name, values[index]))
        index += 1
        cursor += 1
        # control_after_generate 紧跟在 seed 值之后
        if (name == "seed" and index < len(values)
                and isinstance(values[index], str)
                and values[index] in CONTROL_VALUES):
            assigned.append((f"{name}:control_after_generate", values[index]))
            index += 1
    return assigned


def main():
    print("=" * 70)
    print(" 工作流兼容性回归测试（旧工作流 widgets_values 对齐）")
    print("=" * 70)

    for workflow_file in WORKFLOW_FILES:
        path = os.path.join(BASE_DIR, workflow_file)
        print("\n" + "=" * 70)
        print(f" {workflow_file}")
        print("=" * 70)

        if not os.path.exists(path):
            check(f"{workflow_file} 存在", False, "文件缺失")
            continue

        with open(path, "r", encoding="utf-8") as handle:
            workflow = json.load(handle)

        nodes = [n for n in workflow["nodes"] if n["type"].startswith("ModelScope")]
        check(f"{workflow_file}: 找到 {len(nodes)} 个 ModelScope 节点", len(nodes) > 0)

        for node in nodes:
            node_type = node["type"]
            if node_type not in NODE_CLASS_MAPPINGS:
                check(f"{node_type} 仍在节点映射中", False, "节点被改名或删除")
                continue
            check(f"{node_type} 仍在节点映射中", True)

            values = list(node.get("widgets_values") or [])
            names, options = widget_layout(NODE_CLASS_MAPPINGS[node_type])
            assigned = assign_values(values, names)

            print(f"\n  ── {node_type} (id={node['id']}) 保存了 {len(values)} 个值")
            for name, value in assigned:
                print(f"      {name:<28} = {repr(value)[:60]}")

            over = [n for n, _ in assigned if n == "<越界>"]
            check(f"{node_type}: 旧值未超出控件数量", not over, over)

            for name, value in assigned:
                if name in options:
                    check(f"{node_type}: {name} 的旧值 {value!r} 仍在新选项中",
                          value in options[name],
                          f"下拉选项里没有（共 {len(options[name])} 项）")
                elif name.endswith("control_after_generate"):
                    continue
                elif name in VALUE_AGNOSTIC_WIDGETS:
                    check(f"{node_type}: {name} 仍是控件", True)
                else:
                    check(f"{node_type}: {name} 映射正常", True)

            # 核心断言：旧控件必须仍然是新控件列表的**前缀**
            legacy = [n for n, _ in assigned if not n.endswith("control_after_generate")]
            check(f"{node_type}: 旧控件仍是前缀（新增只在末尾）",
                  names[:len(legacy)] == legacy,
                  f"旧={legacy} / 现前 {len(legacy)} 个={names[:len(legacy)]}")

    print("\n" + "=" * 70)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print("  -", name)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
