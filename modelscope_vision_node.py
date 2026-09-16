# -*- coding: utf-8 -*-
"""
ModelScope API-Inference 图生文（视觉理解）节点
===============================================

走魔搭 OpenAI 兼容接口：``https://api-inference.modelscope.cn/v1``

模型列表由 modelscope_config.json 的 ``vision_models`` 驱动，
并提供 ``custom_model`` 自定义输入，因此无需改代码即可使用任意新模型。
"""

from __future__ import annotations

import numpy as np

try:
    from .modelscope_image_node import (
        load_config,
        save_config,
        load_api_tokens,
        parse_api_tokens,
        tensor_to_base64_url,
    )
except ImportError:  # 允许作为顶层模块导入（测试 / 直接运行）
    from modelscope_image_node import (
        load_config,
        save_config,
        load_api_tokens,
        parse_api_tokens,
        tensor_to_base64_url,
    )

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装openai库，图生文功能将不可用")
    print("请运行: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None


def resolve_vision_model(selected_model: str, custom_model: str = "") -> str:
    """custom_model 非空时优先，从而支持任意魔搭模型。"""
    custom = (custom_model or "").strip()
    if custom:
        if custom != selected_model:
            print(f"✏️ 使用自定义模型: {custom}")
        return custom
    return selected_model


def _collect_tokens(api_token: str):
    """解析 Token，兼容旧配置里的单数 ``api_token`` 字段。"""
    tokens = parse_api_tokens(api_token)
    if tokens:
        return tokens
    legacy = str(load_config().get("api_token", "") or "").strip()
    return [legacy] if legacy else []


class ModelScopeVisionNode:
    """图像理解：输入图像 + 提示词，输出文字描述。"""

    @classmethod
    def INPUT_TYPES(cls):
        if not OPENAI_AVAILABLE:
            return {
                "required": {
                    "error_message": ("STRING", {
                        "default": "请先安装openai库: pip install openai",
                        "multiline": True,
                    }),
                }
            }

        config = load_config()
        saved_tokens = _collect_tokens("")
        model_options = list(config.get("vision_models", [])) or ["stepfun-ai/step3"]
        default_model = config.get("default_vision_model", model_options[0])
        if default_model not in model_options:
            default_model = model_options[0]

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_prompt", "描述这幅图"),
                }),
                "api_token": ("STRING", {
                    "default": f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else "",
                    "placeholder": "请输入魔搭API Token（支持多个，用逗号/换行分隔）",
                    "multiline": True,
                }),
            },
            "optional": {
                "model": (model_options, {"default": default_model}),
                "max_tokens": ("INT", {"default": 1000, "min": 100, "max": 16000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                # ↓↓↓ 新增控件，排在末尾以保证旧工作流兼容 ↓↓↓
                "custom_model": ("STRING", {
                    "default": "",
                    "label": "自定义模型ID（填了优先于上面的下拉框）",
                    "placeholder": "例如：Qwen/Qwen3-VL-235B-A22B-Instruct",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze_image"
    CATEGORY = "ModelScopeAPI"

    def analyze_image(
        self,
        image=None,
        prompt="",
        api_token="",
        model="stepfun-ai/step3",
        max_tokens=1000,
        temperature=0.7,
        custom_model="",
        error_message="",
    ):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)

        tokens = _collect_tokens(api_token)
        if not tokens:
            raise Exception("请输入有效的API Token或确保已保存token")

        raw = (api_token or "").strip()
        if raw and raw != f"***已保存{len(load_api_tokens())}个Token***":
            config = load_config()
            config["api_tokens"] = tokens
            config["api_token"] = tokens[0]
            if save_config(config):
                print(f"✅ 已保存 {len(tokens)} 个API Token")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")

        model_name = resolve_vision_model(model, custom_model)

        print("🔍 开始分析图像...")
        print(f"📝 提示词: {prompt}")
        print(f"🤖 模型: {model_name}")
        print(f"🔑 可用Token数量: {len(tokens)}")

        image_url = tensor_to_base64_url(image)
        print("🖼️ 图像已转换为base64格式")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }]

        last_error = None
        for index, token in enumerate(tokens):
            try:
                print(f"🔄 尝试使用第 {index + 1}/{len(tokens)} 个Token...")
                client = OpenAI(
                    base_url="https://api-inference.modelscope.cn/v1",
                    api_key=token,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
                description = response.choices[0].message.content
                print("✅ 分析完成!")
                print(f"📄 结果: {str(description)[:100]}...")
                return (description,)

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"❌ 第 {index + 1} 个Token调用失败: {exc}")
                if index < len(tokens) - 1:
                    print("⏳ 准备尝试下一个Token...")

        error_msg = f"图像分析失败: {last_error}"
        print(f"❌ {error_msg}")
        return (error_msg,)


if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": ModelScopeVisionNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope-Vision 图生文节点",
    }
else:
    class OpenAINotInstalledNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "install_command": ("STRING", {
                        "default": "pip install openai",
                        "multiline": False,
                    }),
                }
            }

        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("message",)
        FUNCTION = "show_install_message"
        CATEGORY = "ModelScopeAPI"

        def show_install_message(self, install_command):
            return ("请先安装openai库才能使用图生文功能: " + install_command,)

    NODE_CLASS_MAPPINGS = {
        "ModelScopeVisionNode": OpenAINotInstalledNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeVisionNode": "ModelScope-Vision 图生文节点 (需要安装openai)",
    }
