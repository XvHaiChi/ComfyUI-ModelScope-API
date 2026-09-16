# -*- coding: utf-8 -*-
"""
ComfyUI-ModelScope-API
======================

魔搭 ModelScope API-Inference 的 ComfyUI 节点集合。

所有节点统一使用官方 API-Inference 接口（异步任务模式）：
    https://api-inference.modelscope.cn/

节点列表
--------
* ModelScope-Image 生图节点        : 文生图
* ModelScope-Image 图像编辑节点    : 图像编辑 / 图生图（统一端点 + image_url）
* ModelScope-Text 文本生成节点     : 文本 / 代码生成
* ModelScope-Vision 图生文节点     : 视觉理解
* ModelScope 图像描述生成          : 图像描述 / 视觉问答
* ModelScope-LoRA 预设管理         : 管理 LoRA 预设
* ModelScope-LoRA 单/多LoRA加载    : 输出 LoRA ID 与权重（最多 6 个）
* ModelScope 模型列表刷新          : 从 /v1/models 拉取可用模型
"""

from .modelscope_image_node import (
    NODE_CLASS_MAPPINGS as IMAGE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_MAPPINGS,
)
from .modelscope_vision_node import (
    NODE_CLASS_MAPPINGS as VISION_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VISION_DISPLAY_MAPPINGS,
)
from .modelscope_text_node import (
    NODE_CLASS_MAPPINGS as TEXT_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as TEXT_DISPLAY_MAPPINGS,
)
from .modelscope_image_caption_node import (
    NODE_CLASS_MAPPINGS as IMAGE_CAPTION_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMAGE_CAPTION_DISPLAY_MAPPINGS,
)

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {
    **IMAGE_MAPPINGS,
    **VISION_MAPPINGS,
    **TEXT_MAPPINGS,
    **IMAGE_CAPTION_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **IMAGE_DISPLAY_MAPPINGS,
    **VISION_DISPLAY_MAPPINGS,
    **TEXT_DISPLAY_MAPPINGS,
    **IMAGE_CAPTION_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "2.0.0"
