# ComfyUI-ModelScope-API
 
ComfyUI-ModelScope-API 是一个强大的 ComfyUI 自定义节点，它架起了 ComfyUI 可视化工作流环境与 ModelScope 丰富模型库之间的桥梁。
 
This is a powerful custom node for ComfyUI that bridges the gap between ComfyUI's visual workflow environment and ModelScope's extensive collection.

---
# 没有Comfyui有需要WebUI界面的可以到龙大的项目
ModelScope API WebUI
基于ModelScope API的现代化Web应用，提供完整的AI模型服务访问界面。

---
## 更新日志：

**2026-09-15 更新（v2.0.0 · API-Inference 全面重构）**
- 全部图像节点统一走官方 **API-Inference 异步任务协议**：提交 `POST /v1/images/generations`（带 `X-ModelScope-Async-Mode: true`）→ 轮询 `GET /v1/tasks/{task_id}`（带 `X-ModelScope-Task-Type: image_generation`）。
- **文生图与图像编辑合并到同一端点**：图像编辑直接通过 payload 里的 `image_url` 传图（支持 `data:image/...;base64,` 本地图像），与官方示例完全一致。
- **修正请求头**：去掉了提交阶段误带的 `X-ModelScope-Task-Type` 以及非标准的 `X-ModelScope-Request-Params`。
- **默认使用 base64 传图**，不再强制依赖第三方图床（图床仍保留为可选模式）。
- **LoRA 支持扩展到 6 个**，并按官方规则自动把权重归一化到 1.0（单个 LoRA 自动改用官方字符串写法）。
- **每个节点新增 `custom_model` 自定义模型输入**，填了即优先使用，**无需改代码即可调用魔搭上任意新模型**。
- 新增 **ModelScope 模型列表刷新** 节点，可从 `/v1/models` 拉取当前可用模型并写回配置。
- 新增 **ModelScope-LoRA 多LoRA加载(6)** 节点。
- 模型下拉列表全部按魔搭线上实际存在的模型 ID 校验更新（含 FireRed-Image-Edit-1.1、Qwen-Image-Edit-2509/2511、Z-Image、GLM-5.2、DeepSeek-V4 等）。
- 新增 `modelscope_api_client.py` 统一客户端模块，集中处理鉴权、任务轮询、Token 轮换、参数降级重试。
- 参数被模型拒绝（HTTP 400）时会自动用精简参数重试一次。
- **支持多图编辑 / 多图融合**：`image` 输入为多张图（batch > 1）时自动按官方写法提交为数组，
  配合 `Qwen/Qwen-Image-Edit-2511`、`Qwen/Qwen-Image-Edit-2509`、
  `FireRedTeam/FireRed-Image-Edit-1.1` 等模型可实现「图一的狗去追图二的飞盘」
  这类多图指令编辑与内容融合。ComfyUI 里用两个 `Load Image` + 内置 `ImageBatch`
  拼成一个 batch 接到 `image` 输入即可，`image_url_format` 保持「自动」。
  实测 `Qwen/Qwen-Image-Edit-2511` 能把两张单人照融合成一张自然的双人合影。
- 旧工作流**完全兼容**：新增控件一律追加在节点末尾，原有控件顺序未变。
- 新增 `check_model_ids.py` **模型 ID 核验脚本**：走免登录的模型详情接口
  （`GET https://www.modelscope.cn/api/v1/models/<org>/<name>`），批量核验配置里的
  模型 ID 是否真实存在、是否开放推理，并按类别区分图像模型与文本/视觉模型。
  退出码可用于 CI。**替代了原先靠抓页面 `<title>` 的土办法**（该页面对不存在的模型
  也返回 HTTP 200，状态码完全不可靠）。
- 新增文生图模型 `krea/Krea-2-Turbo`，新增 Krea2 美学增强 LoRA `hf/mgwr-M87` 预设
  （底座 `krea/Krea-2-Turbo`，触发词 `--preview`）。
- 新增 `list_api_models.py`：**拉取账号可用模型清单**的脚本（需 Token）。
  同时查 API-Inference 与 AIGC 专区两个来源，支持 `--diff`（对比配置找候选新增）
  与 `--write`（合并进配置）。
- 新增 `search_models.py`：**免登录搜索/枚举全站模型**的脚本。
  走 `PUT https://modelscope.cn/api/v1/dolphin/models`（该路由**只接受 PUT**，且无需登录），
  支持按 `tags` / `aigc_type` / `sub_vision_foundation` 等维度筛选。
  实测：全站 **254,310** 个模型，AIGC 类 **91,653** 个，
  其中**可 API 调用的图像模型（Checkpoint）431 个**、图像 LoRA 91,090 个。

**2026-01-01 更新**  更新对Qwen-Image-Edit-2511与Qwen-Image-2512的支持。

**2025-12-12 更新**  对图片解析节点与文字描述节点增加随机种子选择，现在可以每次生成不同的描述了。

**2025-12-12 更新**  修正ModelScope-Vision 图生文节点报错，修正Qwen-VL模型 ModelScope 图像描述生成节点因拼写错误造成的解析错误（致谢ShmilyStar反馈）

**2025-12-11 更新**  生图模型支持Z-image模型。参考了https://github.com/otsluo/comfyui-modelscope-api 的lora代码对图像生成节点与图像编辑节点增加了lora功能，图像描述节点节点可不输入图片直接使用Qwen3-VL作为大语言模型使用。

**2025-10-23 更新**  ComfyUI-ModelScope-API  图像描述节点增加一个次要提示词输入，可同时输入2条提示词（自动合并）

**2025-10-22 更新**  ComfyUI-ModelScope-API  增加图像描述节点 支持Qwen3-VL系列模型反推

## 🏗️ 核心架构概述

架构核心是 `modelscope_api_client.py` 中的 **API-Inference 统一客户端**，它把魔搭官方
异步任务协议收敛到一处，供所有节点复用；各节点只负责「收集参数 → 组装 payload → 把返回图像转成张量」。

```
ComfyUI 节点
   ├── modelscope_image_node.py        生图 / 图像编辑 / LoRA 相关节点
   ├── modelscope_text_node.py         文本生成
   ├── modelscope_vision_node.py       图生文
   └── modelscope_image_caption_node.py 图像描述
                 │
                 ▼
   modelscope_api_client.py   ← 统一客户端
                 │
                 ▼
   https://api-inference.modelscope.cn/
```

### API-Inference 调用流程

```
1) 提交任务（异步）
   POST /v1/images/generations
   headers: Authorization: Bearer <token>
            Content-Type: application/json
            X-ModelScope-Async-Mode: true
   body:   { model, prompt, negative_prompt?, size?, seed?, steps?, guidance?,
             image_url?, loras? }        ← image_url 出现即代表「图像编辑/图生图」
   → { "task_id": "..." }

2) 轮询任务
   GET /v1/tasks/{task_id}
   headers: Authorization: Bearer <token>
            X-ModelScope-Task-Type: image_generation
   → { "task_status": "SUCCEED", "output_images": ["<url>"] }
```

### 架构特点

- **模块化设计**：API 协议集中在统一客户端，节点代码只关心业务参数
- **统一接口**：文生图与图像编辑共用 `/v1/images/generations` 端点
- **云端处理**：无需本地下载模型，直接在云端推理
- **多 Token 轮换**：可填多个 Token，失败自动切换下一个
- **自动降级**：模型不接受某个可选参数（HTTP 400）时，自动用精简参数重试
- **参数验证**：完整的输入参数验证和错误处理

---
 
### ✨ 功能特性
 
- **多模型支持**：下拉框内置经校验的常用模型，并可通过 `custom_model` 使用**任意**魔搭模型
- **三种生成模式**：
  - **文生图模式**：仅通过文本提示词生成图像
  - **图像编辑模式**：基于输入图像和指令进行局部修改
  - **图生图模式**：基于输入图像和提示词整体重绘
- **直接的 API 调用**：无需在本地下载模型，直接通过 API 在云端生成图像
- **完整的参数控制**：分辨率（宽高）、随机种子（Seed）、采样步数（Steps）、提示词引导系数（Guidance）
- **两种传图方式**：默认 base64（无需图床），也可选用图床 URL
- **最多 6 个 LoRA**：支持权重自动归一化到 1.0，符合官方规则
- **模型列表刷新**：一键从 `/v1/models` 拉取当前可用模型并写回配置
 
---
 
## 🖼️ 使用示例
 
### 文生图模式
 
[文本提示词] → [ModelScope API Node] → [生成图像]
 
### 图生图模式
 
[输入图像] + [文本提示词] → [ModelScope API Node] → [转换后图像]
 
---
 
## ⚙️ 安装
 
### 方法一：使用 Git
 
1. 打开一个终端或命令行窗口
2. 导航到你的 ComfyUI 安装目录下的 `custom_nodes` 文件夹

   ```bash
   cd /path/to/your/ComfyUI/custom_nodes/
   ```

3. 运行以下命令克隆本仓库

   ```bash
   git clone https://github.com/hujuying/ComfyUI-ModelScope-API.git
   ```

### 方法二：手动下载

点击本页面右上角的 Code 按钮，然后选择 Download ZIP，
解压下载的 ZIP 文件，将解压后的文件夹（确保文件夹名为 ComfyUI-ModelScope-API）
移动到 ComfyUI 的 `custom_nodes` 目录下，重启 ComfyUI。

### 🚀 使用方法

### 通用步骤

1. 在 ComfyUI 中，通过右键菜单或双击搜索 `ModelScope` 来添加节点
2. 在 `api_tokens` 字段中填入你的 ModelScope Token（支持多个，用逗号/换行分隔；填过一次后会记住）
3. 在 `model` 下拉框中选择模型；**若列表里没有，把模型 ID 填到 `custom_model` 即可**
4. 在 `prompt` 字段中，输入你想要的图像描述或修改提示
5. 将节点的 IMAGE 输出连接到 PreviewImage 或 SaveImage 节点以查看结果

### 文生图模式

使用 **ModelScope-Image 生图节点**，不连接任何图像，仅提供文本提示词。

### 图像编辑 / 图生图模式

使用 **ModelScope-Image 图像编辑节点**：

- 将图像连接到节点的 `image` 输入
- `image_gen_mode` 关闭 = **图像编辑模式**（用 `edit_model`，适合局部修改、换装、加/删物体）
- `image_gen_mode` 开启 = **图生图模式**（用 `gen_model`，整体重绘）
- 提示词描述**你想要进行的修改**，例如「给图中女孩戴上口罩，保持人物光影不变」

### 📋 参数说明（生图 / 图像编辑节点）

| 参数 | 类型 | 范围 | 用途 |
|---|---|---|---|
| `api_tokens` | 字符串 | - | ModelScope 访问令牌，支持多个（逗号/换行分隔） |
| `model` / `gen_model` / `edit_model` | 下拉框 | - | 从内置模型列表中选择 |
| `custom_model` | 字符串 | - | 自定义模型 ID，**填了优先于下拉框** |
| `prompt` | 字符串 | - | 生成 / 编辑指令 |
| `negative_prompt` | 字符串 | - | 反向提示词 |
| `image` | 图像（可选） | - | 图像编辑 / 图生图的输入图像 |
| `width` / `height` | 整数 | 64-2048 | 输出分辨率（映射为 `size: WxH`） |
| `seed` | 整数 | -1-2147483647 | 随机种子，-1 表示每次随机 |
| `steps` | 整数 | 1-100 | 采样步数 |
| `guidance` | 浮点数 | 1.5-20.0 | 提示词引导系数 |
| `lora1_id` ~ `lora6_id` | 字符串 | - | 最多 6 个 LoRA 模型 ID |
| `lora1_w` ~ `lora6_w` | 浮点数 | 0.0-2.0 | 对应 LoRA 权重 |
| `auto_normalize_lora` | 布尔 | - | 是否把 LoRA 权重自动归一化到 1.0 |
| `image_input_mode` | 下拉框 | - | 传图方式：base64（推荐）或图床 URL |
| `image_url_format` | 下拉框 | - | `image_url` 写法：自动 / 始终数组 / 始终字符串 |

### 🧩 全部节点

| 节点 | 说明 |
|---|---|
| ModelScope-Image 生图节点 | 文生图 |
| ModelScope-Image 图像编辑节点 | 图像编辑 / 图生图 |
| ModelScope-Text 文本生成节点 | 文本 / 代码生成 |
| ModelScope-Vision 图生文节点 | 视觉理解 |
| ModelScope 图像描述生成 | 图像描述 / 视觉问答（不接图可当纯文本 LLM） |
| ModelScope-LoRA 预设管理 | 管理 LoRA 预设 |
| ModelScope-LoRA 单LoRA加载 | 输出单个 LoRA ID 与权重 |
| ModelScope-LoRA 多LoRA加载(3) | 输出 3 个 LoRA ID 与权重 |
| ModelScope-LoRA 多LoRA加载(6) | 输出 6 个 LoRA ID 与权重 |
| ModelScope 模型列表刷新 | 从 `/v1/models` 拉取可用模型并写回配置 |

### 🎯 支持的模型

> **想用列表里没有的模型？** 直接把模型 ID 填进节点的 **`custom_model`** 输入框即可，
> `custom_model` 的优先级高于下拉框，无需修改任何代码。
> 也可以使用 **ModelScope 模型列表刷新** 节点从 `/v1/models` 拉取当前可用模型并写回配置。

#### 哪些图片模型可以通过 API 调用？

魔搭把「AIGC 专区」上架的模型全部通过 API-Inference 对外开放，权威入口：

- AIGC 专区：https://www.modelscope.cn/aigc/
- 可调用的模型清单（Qwen-Image 生态）：https://www.modelscope.cn/aigc/models?filter=QWEN_IMAGE_20_B

两个实用要点：

1. **LoRA 可以直接当模型用**。把 `model` 填成 LoRA 的模型 ID 即可（例如 `MoYouuu/MYHuman-QWen`），
   平台会自动加载它依赖的基础模型 —— 社区上千个 Qwen-Image LoRA 都能这样直接调用。
   也可以用节点上的 `lora1_id`~`lora6_id` 走 `loras` 字段，两种方式都支持。
2. **`/v1/models` 不等于全量清单**。未鉴权时它只返回约 37 个公共样例模型（其中图片模型
   只有 2 个），而且**它连 `Qwen/Qwen-Image` 这种主力模型都不在列表里**，所以既不能当
   全量清单，也不能用来判断某个模型是否存在。要拿到自己账号可用的完整列表，请在
   「ModelScope 模型列表刷新」节点里填入 Token 后执行。

#### 怎么知道魔搭一共有多少个模型可以调？

**有免登录的全量搜索接口**，直接查就行：

```
PUT https://modelscope.cn/api/v1/dolphin/models        ← 只吃 PUT，不需要登录
{"PageSize":100,"PageNumber":1,
 "Criterion":[{"category":"tags","predicate":"contains","values":["text-to-image"]}]}
```

> ⚠️ 这个路由**只接受 PUT**。用 GET/POST 会 404 或报错，很容易误判成「接口不存在」。

仓库里的 `search_models.py` 已经把它封装好了：

```bash
python search_models.py                  # 总览统计
python search_models.py --api-image      # 枚举 Checkpoint，筛出可调用的图像模型
python search_models.py --image-catalog  # 图像模型按底座分布
python search_models.py --facets         # 列出所有可筛选维度及取值
python search_models.py --foundation KREA_2_TURBO --limit 30
python search_models.py --export out.json
```

**实测数字（2026-09-16）**：

| 口径 | 数量 |
|---|---|
| 魔搭全站模型总数 | **254,310** |
| AIGC 类模型合计 | **91,653**（LoRA 91,090 + Checkpoint 527 + VAE 36） |
| **可 API 调用的图像模型（Checkpoint）** | **431** |
| 图像类 LoRA | 91,090（可直接当 `model` 传，或作为 `loras` 传给底座） |

那 527 个 Checkpoint 里被排除的 96 个，主要是视频模型（`image-to-video` 9 个、
`text-to-video-synthesis` 9 个）和未登记推理任务的模型（74 个）。

**图像模型按底座分布（前几名）**：

| 底座 | LoRA | Checkpoint | 合计 |
|---|---|---|---|
| `KREA_2_TURBO` | 14,807 | 17 | 14,824 |
| `QWEN_IMAGE_20_B` | 13,240 | 21 | 13,261 |
| `QWEN_IMAGE_2512` | 8,326 | 4 | 8,330 |
| `SD_XL` | 7,647 | 226 | 7,873 |
| `Z_IMAGE` | 7,764 | 9 | 7,773 |
| `QWEN_IMAGE_EDIT_2511` | 1,999 | 2 | 2,001 |
| `QWEN_IMAGE_EDIT_2509` | 1,414 | 4 | 1,418 |

> 下拉框里只放了常用的十几个，剩下的用节点的 **`custom_model`** 输入框直接填即可。
> 想批量扩充下拉列表，可以配合 `list_api_models.py --write`（需 Token）。

**筛选语法**（`Criterion` 的键名是全小写）：

| 项 | 取值 |
|---|---|
| `category` | `tags` / `aigc_type` / `sub_vision_foundation` / `vision_foundation` / `model_type` / `libraries` / `language` / `license` |
| `predicate` | 实测**只有 `contains` 有效**；`eq` / `in` 会被静默忽略（返回全量） |
| `values` | 字符串数组 |

#### 怎么确认一个模型 ID 是不是有效的？

用仓库里的核验脚本（走**免登录**的模型详情接口）：

```bash
python check_model_ids.py                     # 核验 modelscope_config.json 里的全部 ID
python check_model_ids.py krea/Krea-2-Turbo   # 核验指定 ID
python check_model_ids.py --quiet a/b c/d     # 只输出结论行，适合接 CI
```

输出会给出 `[可用] / [待验证] / [不可用]` 三级结论，并列出 AigcType、底座模型、
下载量、可见性等信息。退出码 0 = 无死 ID。

> ⚠️ **别用 `SupportApiInference` 字段判断**。实测连 `Qwen/Qwen-Image`、
> `FireRedTeam/FireRed-Image-Edit-1.1` 这类官方示例主力模型该字段也是 `false`。
> 可靠依据是 `SupportInference` + `Tasks[].Name` 这两个**稳定**字段
> （`widgets` 字段会随时间抖动，只作参考不作判据）。
>
> ⚠️ 直接访问 `https://www.modelscope.cn/models/<org>/<name>` 时，
> **不存在的模型也返回 HTTP 200**（纯前端 SPA），所以状态码完全不可靠。

#### 图像编辑的两种 `image_url` 写法

官方不同模型的示例写法不一致，本插件提供 `image_url_format` 开关：

| 模型 | 官方写法 |
|---|---|
| `FireRedTeam/FireRed-Image-Edit-1.1` | 单图用**字符串** |
| `Qwen/Qwen-Image-Edit-2509` | 多图（1-3 张）用**数组** |

默认「自动」：1 张图发字符串，多张图发数组。把 `LoadImage` 的 batch 接到 `image` 输入即可多图编辑。

### 文生图模型（image_models）

| 模型名称 | 说明 |
|---|---|
| `Qwen/Qwen-Image` | 通义千问图像生成，通用文生图首选 |
| `Qwen/Qwen-Image-2512` | Qwen-Image 2512 版本 |
| `MusePublic/Qwen-image` | 细节丰富，适合复杂构图 |
| `MusePublic/489_ckpt_FLUX_1` | FLUX 系列变体 |
| `MusePublic/flux-high-res` | FLUX 高分辨率版本 |
| `black-forest-labs/FLUX.1-Krea-dev` | FLUX.1 Krea |
| `MAILAND/majicflus_v1` | 麦橘超然，艺术风格 |
| `MoYouuu/MYHuman-QWen` | 墨幽人造人 |
| `Tongyi-MAI/Z-Image-Turbo` | 造相 Z-Image Turbo，出图快 |
| `Tongyi-MAI/Z-Image` | 造相 Z-Image |
| `krea/Krea-2-Turbo` | Krea 2 Turbo，电影感/美术感强，turbo 模型（官方推荐 `steps=8`、`cfg=0`） |
| `ideogram-ai/ideogram-4-fp8` | Ideogram 4，擅长文字排版与海报 |
| `LaxharLAB/NoobAI-XL` | NoobAI-XL，二次元/插画 |
| `atonyxu/Illustrious-XL` | Illustrious-XL，二次元/插画 |
| `MusePublic/Qwen-image-fp8` | Qwen-Image 的 FP8 量化版，显存占用更低 |

### 图像编辑 / 图生图模型（image_edit_models）

| 模型名称 | 说明 |
|---|---|
| `Qwen/Qwen-Image-Edit` | 通义千问图像编辑 |
| `Qwen/Qwen-Image-Edit-2511` | 图像编辑 2511 版本 |
| `Qwen/Qwen-Image-Edit-2509` | 图像编辑 2509 版本 |
| `MusePublic/Qwen-Image-Edit` | Qwen-Image-Edit 衍生版 |
| `MusePublic/FLUX.1-Kontext-Dev` | FLUX Kontext 图像编辑 |
| `black-forest-labs/FLUX.1-Kontext-dev` | FLUX.1 Kontext |
| `FireRedTeam/FireRed-Image-Edit-1.1` | 小红书 FireRed 图像编辑 1.1 |
| `FireRedTeam/FireRed-Image-Edit-1.0` | 小红书 FireRed 图像编辑 1.0 |
| `black-forest-labs/FLUX.2-klein-9B` | FLUX.2 Klein 9B，下载量最高的图像编辑模型（149 万+） |
| `black-forest-labs/FLUX.2-klein-4B` | FLUX.2 Klein 4B，更轻量 |
| `black-forest-labs/FLUX.2-klein-base-9B` | FLUX.2 Klein 9B 基础版 |
| `black-forest-labs/FLUX.2-dev` | FLUX.2 完整版 |

### 文本 / 视觉模型

文本节点与视觉节点同样由 `modelscope_config.json` 的 `text_models` / `vision_models` 驱动，
内置了 Qwen3 / Qwen3.5 / Qwen3.8、DeepSeek-V4、GLM-5.2、MiniMax-M3、Step、InternVL 等系列，
并同样支持 `custom_model` 自定义输入。

### 💡 最佳实践
提示词增强
在 ModelScope API 之前使用提示词增强节点来改进您的文本描述
后处理
将输出连接到图像增强或放大节点进行最终精修
批量处理
创建多个具有不同种子的 ModelScope API 节点，以生成相同概念的变体
图生图提示词技巧
对于图像到图像生成，提示词应该描述您想要进行的更改，而不是整个图像。例如，如果您输入一张猫的照片，像"让它看起来像水彩画"这样的提示词会比"水彩风格的猫"更有效。

### 🔑 如何获取 ModelScope 访问令牌
1.登录或注册：访问 https://www.modelscope.cn/，如果您已有账户，请点击右上角的“登录”按钮进行登录。如果没有账户，请先点击“注册”按钮完成账户创建。
2.进入个人主页：登录成功后，将鼠标悬停在页面右上角的个人头像上。
3.访问令牌页面：在弹出的下拉菜单中，点击“访问令牌”选项。
4.查看或生成令牌：在“访问令牌”页面，您可以查看到您的个人访问令牌（Access Token）。如果之前没有生成过，系统可能会提示您生成一个新的令牌。

### 🙏 致谢
API服务提供方: 魔搭 ModelScope
模型提供方: MusePublic
图片上传服务: freeimage.host
📄 许可证
本项目采用 MIT License 开源。


这个新的 README.md 内容基于概述页面的架构信息重新组织，增加了核心架构概述部分，优化了参数说明表格，并添加了最佳实践部分，使文档更加完整和实用。

[通用API架构](4-universal-api-architecture)
[快速开始](2-quick-start)
[在ComfyUI中使用ModelScope模型](3-working-with-modelscope-models-in-comfyui)
