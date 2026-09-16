# -*- coding: utf-8 -*-
"""
ModelScope 图像描述生成节点
===========================

输入图像（可选）+ 提示词，输出文字描述。

* 支持 Qwen3-VL 等视觉语言模型；不接图像时可用作纯文本大语言模型。
* 模型列表由 modelscope_config.json 的 ``vision_models`` 驱动，
  并提供 ``custom_model`` 自定义输入，无需改代码即可使用任意新模型。
"""

from __future__ import annotations

import numpy as np
import torch

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
    OPENAI_AVAILABLE = False
    OpenAI = None

DEFAULT_CAPTION_PROMPT = "详细描述这张图片的内容，包括主体、背景、颜色、风格等信息"


def resolve_caption_model(selected_model: str, custom_model: str = "") -> str:
    """custom_model 非空时优先，从而支持任意魔搭模型。"""
    custom = (custom_model or "").strip()
    if custom:
        if custom != selected_model:
            print(f"✏️ 使用自定义模型: {custom}")
        return custom
    return selected_model


class ModelScopeImageCaptionNode:
    """图像描述生成 / 视觉问答。"""

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
        saved_tokens = load_api_tokens()

        # 视觉模型优先，其次允许选用文本模型（纯文本模式）
        model_options = list(config.get("vision_models", []))
        for extra in config.get("text_models", []):
            if extra not in model_options:
                model_options.append(extra)
        if not model_options:
            model_options = ["Qwen/Qwen3-VL-8B-Instruct"]

        default_model = "Qwen/Qwen3-VL-8B-Instruct"
        if default_model not in model_options:
            default_model = model_options[0]

        return {
            "required": {
                "api_tokens": ("STRING", {
                    "default": f"***已保存{len(saved_tokens)}个Token***" if saved_tokens else "",
                    "placeholder": "请输入API Token（支持多个，用逗号/换行分隔）",
                    "multiline": True,
                }),
            },
            "optional": {
                # image 可选：不接图像时作为纯文本大语言模型使用
                "image": ("IMAGE", {"optional": True}),
                "prompt1": ("STRING", {"multiline": True, "default": DEFAULT_CAPTION_PROMPT}),
                "prompt2": ("STRING", {"multiline": True, "default": ""}),
                "model": (model_options, {"default": default_model}),
                "max_tokens": ("INT", {"default": 1000, "min": 100, "max": 16000}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
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
    FUNCTION = "generate_caption"
    CATEGORY = "ModelScopeAPI"

    @staticmethod
    def create_blank_image(width=64, height=64):
        """创建空白图像张量（符合 ComfyUI 的图像格式要求）。"""
        blank_np = np.ones((height, width, 3), dtype=np.uint8) * 255
        return torch.from_numpy(blank_np).unsqueeze(0).float() / 255.0

    def generate_caption(
        self,
        image=None,
        api_tokens="",
        prompt1=DEFAULT_CAPTION_PROMPT,
        prompt2="",
        model="Qwen/Qwen3-VL-8B-Instruct",
        max_tokens=1000,
        temperature=0.7,
        seed=-1,
        custom_model="",
    ):
        if not OPENAI_AVAILABLE:
            return ("请先安装openai库: pip install openai",)

        if seed == -1:
            seed = int(np.random.randint(0, 2147483647))
        np.random.seed(seed % (2**32 - 1))

        if image is None:
            print("⚠️ 未输入图像，自动生成空白图像作为输入")

        prompt_parts = []
        if prompt1 and prompt1.strip():
            prompt_parts.append(prompt1.strip())
        if prompt2 and prompt2.strip():
            prompt_parts.append(prompt2.strip())
        prompt = ", ".join(prompt_parts) if prompt_parts else DEFAULT_CAPTION_PROMPT

        tokens = parse_api_tokens(api_tokens)
        if not tokens:
            raise Exception("请提供至少一个有效的API Token")

        raw = (api_tokens or "").strip()
        if raw and raw != f"***已保存{len(load_api_tokens())}个Token***":
            config = load_config()
            config["api_tokens"] = tokens
            if save_config(config):
                print(f"✅ 已保存 {len(tokens)} 个API Token")
            else:
                print("⚠️ API Token保存失败，但不影响当前使用")

        model_name = resolve_caption_model(model, custom_model)

        print("🔍 开始生成图像描述...")
        print(f"📝 提示词: {prompt}")
        print(f"🤖 模型: {model_name}")
        print(f"🔑 可用Token数量: {len(tokens)}")
        print(f"🌱 Seed: {seed}")

        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": tensor_to_base64_url(image)},
            })
            print("🖼️ 图像已转换为base64格式")
        else:
            print("📄 纯文本模式（未附带图像）")

        messages = [{"role": "user", "content": content}]

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
                print(f"✅ 第 {index + 1} 个Token调用成功!")
                print(f"📄 结果预览: {str(description)[:100]}...")
                return (description,)

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"❌ 第 {index + 1} 个Token调用失败: {exc}")
                if index < len(tokens) - 1:
                    print("⏳ 准备尝试下一个Token...")

        error_msg = f"图像描述生成失败: {last_error}"
        print(f"❌ {error_msg}")
        return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "ModelScopeImageCaptionNode": ModelScopeImageCaptionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelScopeImageCaptionNode": "ModelScope 图像描述生成",
}
