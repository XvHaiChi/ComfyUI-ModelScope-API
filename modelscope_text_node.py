# -*- coding: utf-8 -*-
"""
ModelScope API-Inference 文本生成节点
=====================================

走魔搭 OpenAI 兼容接口：``https://api-inference.modelscope.cn/v1``

模型列表由 modelscope_config.json 的 ``text_models`` / ``vision_models`` 驱动，
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
    )
except ImportError:  # 允许作为顶层模块导入（测试 / 直接运行）
    from modelscope_image_node import (
        load_config,
        save_config,
        load_api_tokens,
        parse_api_tokens,
    )

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装openai库，文本生成功能将不可用")
    print("请运行: pip install openai")
    OPENAI_AVAILABLE = False
    OpenAI = None


def resolve_text_model(selected_model: str, custom_model: str = "") -> str:
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


class ModelScopeTextNode:
    """文本生成（chat completions）。"""

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

        model_options = list(config.get("text_models", []))
        for extra in config.get("vision_models", []):
            if extra not in model_options:
                model_options.append(extra)
        if not model_options:
            model_options = ["Qwen/Qwen3-Coder-480B-A35B-Instruct"]

        default_model = config.get("default_text_model", model_options[0])
        if default_model not in model_options:
            default_model = model_options[0]

        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_user_prompt", "你好"),
                }),
                "api_token": ("STRING", {
                    "default": f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else "",
                    "placeholder": "请输入魔搭API Token（支持多个，用逗号/换行分隔）",
                    "multiline": True,
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": config.get("default_system_prompt", "You are a helpful assistant."),
                }),
                "model": (model_options, {"default": default_model}),
                "max_tokens": ("INT", {"default": 2000, "min": 100, "max": 32000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "stream": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                # ↓↓↓ 新增控件，排在末尾以保证旧工作流兼容 ↓↓↓
                "custom_model": ("STRING", {
                    "default": "",
                    "label": "自定义模型ID（填了优先于上面的下拉框）",
                    "placeholder": "例如：deepseek-ai/DeepSeek-V4-Pro",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "ModelScopeAPI"

    def generate_text(
        self,
        user_prompt="",
        api_token="",
        system_prompt="You are a helpful assistant.",
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        max_tokens=2000,
        temperature=0.7,
        stream=True,
        seed=-1,
        custom_model="",
        error_message="",
    ):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)

        if seed == -1:
            seed = int(np.random.randint(0, 2147483647))
        np.random.seed(seed % (2**32 - 1))

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

        model_name = resolve_text_model(model, custom_model)

        print("💬 开始文本生成...")
        print(f"🤖 模型: {model_name}")
        print(f"📝 用户提示: {str(user_prompt)[:50]}...")
        print(f"⚙️ 系统提示: {str(system_prompt)[:50]}...")
        print(f"🌡️ 温度: {temperature}")
        print(f"📊 最大tokens: {max_tokens}")
        print(f"⚡ 流式输出: {stream}")
        print(f"🔢 种子: {seed}")
        print(f"🔑 可用Token数量: {len(tokens)}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

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
                    stream=stream,
                )

                if stream:
                    print("📡 接收流式响应...")
                    full_response = ""
                    for chunk in response:
                        choices = getattr(chunk, "choices", None) or []
                        if choices and choices[0].delta and choices[0].delta.content:
                            content = choices[0].delta.content
                            full_response += content
                            print(content, end="", flush=True)
                    print("\n✅ 流式生成完成!")
                    print(f"📄 总长度: {len(full_response)} 字符")
                    return (full_response,)

                result = response.choices[0].message.content
                print("✅ 文本生成完成!")
                print(f"📄 结果长度: {len(result)} 字符")
                print(f"📝 结果预览: {str(result)[:100]}...")
                return (result,)

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"❌ 第 {index + 1} 个Token调用失败: {exc}")
                if index < len(tokens) - 1:
                    print("⏳ 准备尝试下一个Token...")

        error_msg = f"文本生成失败: {last_error}"
        print(f"❌ {error_msg}")
        return (error_msg,)


if OPENAI_AVAILABLE:
    NODE_CLASS_MAPPINGS = {
        "ModelScopeTextNode": ModelScopeTextNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeTextNode": "ModelScope-Text 文本生成节点",
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
            return ("请先安装openai库才能使用文本生成功能: " + install_command,)

    NODE_CLASS_MAPPINGS = {
        "ModelScopeTextNode": OpenAINotInstalledNode,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelScopeTextNode": "ModelScope-Text 文本生成节点 (需要安装openai)",
    }
