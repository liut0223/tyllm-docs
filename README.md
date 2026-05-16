

# 更新说明

本文记录了``Edge10``系列大模型工具链的变更情况。

**20260513/v1.1.1**

- 🚀量化工具更新 (v1.1.1)
    - 新增 Qwen3.5 模型 GPTQv2 W4A16 量化支持
    - 新增 Qwen3.5 模型敏感层分析工具
    - 新增 Qwen3.5 GPTQv2 W4A16 混合精度量化配置
- 🚀更新量化工具镜像
    - 镜像地址：`113.100.143.90:8091/edgex/tyquantize:v1.1.1`
- 🚀更新量化工具使用说明

**20260327/v1.1.0**

- 🚀量化工具全新版本 (v1.1.0)
    - PETQuant 完成系统性重构，API 全面统一
    - 新增 per block 量化支持（已验证 DeepSeek-R1 模型）
    - 新增性能模式选择，根据显存大小自动适配
    - 重构后顶层 API 更加简洁规范
- 🚀更新量化工具镜像
    - 镜像地址：`113.100.143.90:8091/edgex/tyquantize:v1.1.0`
- 🚀新增支持模型
    - Qwen3-14B、Qwen3-32B
    - MiniCPM-V-4_5、InternVL3-8B
    - Qwen3-Reranker-0.6B
- 🚀更新量化工具使用说明

**202600228/v1.2.1**

- 🚀更新版本v1.2.1
- 🚀编译工具更新(v1.2.1)
    - 修复qwen3-vl-8B精度下降问题
    - 修复一些模型编译出错问题（llama3-8B、qwen1.5-1.8B、qwen2-vl-7B、qwen3-32B 40k）
    - 修复某些图像shape输入下编译崩溃问题
    - 配套hcp版本更新到1.2.21
- 🚀更新大模型支持列表
- 🚀更新编译工具使用说明

**202600115/v1.1.8**

- 🚀更新版本v1.1.8
- 🚀量化工具更新(v1.0.4)
    - 支持AWQ\GPTQv2\Quarot量化方法
- 🚀编译工具更新(v1.1.8)
    - Qwen3 vl 8B vit性能优化
    - 解决Qwen3 vl单Die编译问题
    - vllm升级到0.11.0，解决min max pixel配置问题
    - ubuntu升级到22.04
    - 支持编译Die remap模型
- 🚀更新大模型支持列表
- 🚀更新量化工具使用说明
- 🚀更新编译工具使用说明

**20251128/v1.1.7**

- 🚀正式版本v1.1.7
- 🚀独立的量化工具镜像(v1.0.3)
- 🚀独立的编译工具镜像(v1.1.7)
- 🚀更新大模型支持列表
- 🚀更新量化工具使用说明
- 🚀更新编译工具使用说明

**20250414/v1.0.8**

- 🚀优化``Qwen2.5-VL-7B`` ``ViT``部分性能

**20250320/v1.0.6**

- 🚀新增支持``Qwen2.5-VL-7B``模型``3Die``编译

**20250305/v0.0.2**

- 🚀新增支持``Qwen2.5-VL-7B``模型``4Die/1Die``编译


<br>

# 整体介绍

``TyQuant``为云天励飞大模型量化工具，用户可通过本工具或其它开源量化工具对大模型完成量化；``TyLLM``是云天励飞推出的大模型工具链，可帮助用户将大模型编译为``Edge10``系列芯片上执行的模型。``TyLLM``为云天基于``TyTVM``工具链针对大模型增量开发的工具，主要基于``PyTorch``和``vLLM``对大模型做专属和定制优化；``TyTVM``为云天模型转换、量化、仿真、编译工具链，主要负责将模型编译为芯片执行的模型。

### 整体架构如图：

<div style="text-align:center;">
  <img src="./assets/whiteboard_exported_image.png" alt="架构图" style="width:100%; height:auto;" />
</div>

### 支持模型列表

已经支持的模型如下（包括不限于）：

| Model                                      | Quant Support | Compile Support |
| :----------------------------------------- | :-----------: | :-------------: |
| Qwen/Qwen3-VL-8B                           |      ✅       |       ✅        |
| Qwen/Qwen3-VL-4B                           |      ✅       |       ✅        |
| Qwen/Qwen3-VL-2B                           |      ✅       |       ✅        |
| Qwen/Qwen3-32B                             |      ✅       |       ✅        |
| Qwen/Qwen3-14B                             |      ✅       |       ✅        |
| Qwen/Qwen3-8B                              |      ✅       |       ✅        |
| Qwen/Qwen3-4B                              |      ✅       |       ✅        |
| Qwen/Qwen3-1.7B                            |      ✅       |       ✅        |
| Qwen/Qwen2.5-VL-7B                         |      ✅       |       ✅        |
| Qwen/Qwen2.5-VL-3B                         |      ✅       |       ✅        |
| Qwen/Qwen2-VL-7B                           |      ❌       |       ✅        |
| Qwen/Qwen2-7B                              |      ✅       |       ✅        |
| Qwen/Qwen1.5-1.8B                          |      ❌       |       ✅        |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  |      ✅       |       ✅        |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    |      ✅       |       ✅        |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B   |      ✅       |       ✅        |
| Llama3-8B                                  |      ❌       |       ✅        |
| MiniCPM-V-4_5                              |      ✅       |       ❌        |
| InternVL3-8B                               |      ✅       |       ❌        |
| Qwen3-Reranker-0.6B                        |      ✅       |       ❌        |
<br>
<br>

# 一、环境准备

本节介绍使用``TyLLM``工具链前的开发环境准备工作。``TyLLM``使用``Docker``容器进行工具链集成，用户可通过``Docker``加载``TyLLM``镜像文件，然后进行模型量化、编译、评估(未来)等工作，因此开发环境准备阶段需要正确安装``Docker``环境，同时目前需要量化阶段需要``GPU``来加速，以及多模态模型的编译依赖``vLLM``框架来推理，因此暂时需要``GPU``。

- **Nvidia GPU**
- **Nvidia Container Toolkit**
- **Docker>19.03**

### 1.1 安装Nvidia GPU 驱动

```shell
# 驱动版本尽量选择最高（当前量化工具cuda-12.6,驱动建议安装580及以上）
sudo apt install nvidia-driver-530 
# 安装完成后，执行nvidia-smi命令显示如下，表示安装成功。
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.09              Driver Version: 580.82.09      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:67:00.0 Off |                  Off |
| 56%   49C    P0             68W /  450W |       3MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

### 1.2 安装Docker

```shell
sudo apt install docker.io
sudo docker -v
```

### 1.3 安装Nvidia Container Toolkit

添加包仓库和``GPG key``:
```shell
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
             sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
             sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

更新源，安装nvidia-container-toolkit

```shell
sudo apt update
sudo apt install nvidia-container-toolkit
```

### 1.4 安装TyQuant量化工具

量化工具镜像获取途径如下，请务必将``${version}``替换为实际对应的版本号，比如``v1.1.1``：

```shell
sudo docker login 113.100.143.90:8091 -u custom -p DE@sz_intellif_2021
sudo docker pull 113.100.143.90:8091/edgex/tyquantize:${version}
```


### 1.5 安装TyLLM工具链

编译工具链获取途径如下，请务必将``${version}``替换为实际对应的工具链版本号，比如``v1.2.1``：

```shell
sudo docker login 113.100.143.90:8091 -u custom -p DE@sz_intellif_2021
sudo docker pull 113.100.143.90:8091/edgex/tyllm:${version}
```

> **注意**
> 
> 需要将``113.100.143.90:8091``加入``/etc/docker/daemon.json``中的``insecure-registries``字段中，如下：
> 
> ```json
> {     
>      "insecure-registries": ["113.100.143.90:8091"]
> }
>  ```
> 修改后，重启``docker``生效，``sudo systemctl restart docker``

---
<br>
<br>

# 二、模型量化

必须在带有GPU的主机量化以加快速度。

## 2.1 启动量化工具镜像

以下命令创建容器，其中``${your_data_dir}``表示宿主机中用户数据目录，``${version}``需改为实际版本``tag``。
```shell
sudo docker run --gpus all -v ${your_data_dir}:/data -it 113.100.143.90:8091/edgex/tyquantize:${version} bash
```
> 注意：v1.1.1 镜像内完整量化样例位于 `/opt/app/test`，配置文件位于 `/opt/app/configs`，敏感层分析工具位于 `/opt/app/tools/llm_sensitivity_analysis`。

### 2.1.1 v1.1.1 目录结构

- `/opt/app/test`
  - `test.py`：拆分式入口，分别传入 `quant_config`、`alg_config`、`dataset_config`、`sampling_config`
  - `test_unified_config.py`：推荐入口，使用单个 `unified_config` 同时描述量化配置和算法配置
  - 其余 `test_*.py`：各模型/算法的独立示例
- `/opt/app/configs/alg_configs`
  - 算法配置，当前镜像包含 `awq.json`、`gptqv2.json`、`gptqv2_vlm.json`、`ostquant.json`、`quarot.json`、`rtn.json`、`smoothquant.json`
- `/opt/app/configs/quant_configs`
  - 模型量化配置，例如 `Qwen3_wui4_g.json`、`Qwen3_wi4_c_ai8_k_mse.json`、`Qwen3_VL_wi8_c_ai8_t.json`
- `/opt/app/configs/dataset_configs`
  - 校准/评测数据配置，命名规则为 `算法_模型_calib.json` 或 `算法_模型_eval.json`
- `/opt/app/configs/unified_configs`
  - 推荐使用的一体化配置，例如 `awq_qwen3.json`、`gptqv2_qwen3.json`、`ostquant_qwen3.json`
- `/opt/app/configs/sampling_configs`
  - 推理验证默认参数，当前镜像包含 `llm_default.json`、`vlm_default.json`、`reranker_default.json`
- `/opt/app/tools/llm_sensitivity_analysis`
  - LLM 敏感层分析工具，包含 `run_llm_sensitivity.py` 和 `analysis_configs/*`

### 2.1.2 推荐使用方式

v1.1.1 的量化入口已经统一，建议优先使用 `test/test_unified_config.py`。流程如下：

1. 从 `configs/unified_configs/*.json` 读取 `quant_config` 和 `alg_config`
2. 根据 `model_path` 与 `dataset_path` 自动创建 `DataSelector`
3. 调用 `PETQuantizer.run(...)` 执行量化
4. 调用 `save_model(...)` 保存量化模型
5. 调用 `generate(...)` 或 `load_model(...)` 做回归验证

如需分别切换量化配置、算法配置、采样配置，可改用 `test/test.py`。

### 2.1.3 新版 API 使用指南

PETQuant 完成系统性重构后采用统一的顶层 API，支持通过 JSON 配置文件管理所有量化参数。

**核心类说明：**

| 类名 | 说明 |
|------|------|
| `ConfigHelper` | 量化配置类，包含权重量化和激活值量化配置 |
| `AlgConfig` | 量化算法配置类，支持 AWQ、GPTQv2、OSTQuant、Quarot 等算法 |
| `PETQuantizer` | 量化器核心类，执行量化、保存、加载、推理等操作 |
| `DataSelector` | 数据集选择器，根据模型和算法自动匹配校准数据 |

**基本使用流程：**

```python
from pathlib import Path
from PETQuant import ConfigHelper, AlgConfig, PETQuantizer, DataSelector, PerformanceModeEnum

# 1. 从文件获取量化配置和算法配置
quant_configs = [ConfigHelper.create_from_file(Path(quant_config_file))]
alg_configs = [AlgConfig.create_from_file(Path(alg_config_file))]

# 2. 创建 PETQuantizer 对象
mode = PerformanceModeEnum.Auto  # 根据显存自动选择性能模式
pet_quantizer = PETQuantizer(
    model_path=args.model_path,
    alg_configs=alg_configs,
    quant_configs=quant_configs,
    mode=mode,
)

# 3. 创建数据集选择器
data_selector = DataSelector.from_model(
    pet_quantizer.get_model(), args.dataset_path
)

# 4. 获取模型名称和算法名称
model_name = pet_quantizer.get_model_name()
alg_names = pet_quantizer.get_alg_names()

# 5. 从文件读取校准数据集配置参数
calib_dataset_params = dict_from_json_file(Path(args.calib_dataset_config_file))

# 6. 选择 dataloader
calib_dataloader = data_selector.get_dataloader(
    model_name, alg_names[0], **calib_dataset_params
)

# 7. 执行量化
pet_quantizer.run(dataloader=calib_dataloader)
print("量化成功")

# 8. 保存量化模型
pet_quantizer.save_model(save_path=args.save_path)

# 9. 验证量化模型
sampling_params = {
    "prompt": "什么是量子力学？",
    "enable_thinking": False,
    "sampling_params": {
        "max_new_tokens": 100,
        "do_sample": False,
    },
}
print(pet_quantizer.generate(**sampling_params))
```

**示例代码路径：** `PETQuant/test/test.py` 或 `PETQuant/test/test_unified_config.py`

**配置文件路径：** `PETQuant/configs/`

## 2.2 量化模型指标评估专属环境（w4a8 必用）
官方vllm不支持w4a8量化模型，需使用定制镜像，创建容器如下：
镜像地址：`192.168.14.129:80/library/aied/custom-vllm:0.11.0`
创建容器命令：
```bash
docker run -it --name vllm_custom --gpus all -v  /data/:/data/ --ipc host  -p 8001:8000 --shm-size 16g 192.168.14.129:80/library/aied/custom-vllm:0.11.0 bash
```


## 2.3 支持的量化算法
v1.1.1 版本支持以下量化算法，以下内容依据镜像内 `/opt/app/test`、`/opt/app/configs` 与 `/opt/app/tools` 实际文件整理：


| 量化算法 | 支持精度 | 权重量化 | 激活值量化 | 激活值动态量化 | 推荐硬件配置 |
|----------|----------|----------|------------|----------------|--------------|
| AWQ | Wi4Af16 | 非对称 | 不量化 | 否 | Qwen3-8B/14B: A800/4090*1; Qwen3-32B: A800/4090*1 |
| GPTQv2 | Wi4Ai8、Wi8Ai8、Wi4Af16 | 非对称 | 非对称 | 支持 int8 动态/静态 | 4090*1（Qwen3 / Qwen3-VL / Qwen2.5-VL / Qwen3.5 已有样例） |
| OSTQuant | Ai8 动态激活量化 | 不量化 | 对称 | 是 | 需使用 `torchrun` 分布式启动，卡数按模型规模选择 |
| Quarot | Wi4Ai8、Wi8Ai8 | 对称 | 对称 | 是 | Qwen3: 4090*1；DeepSeek-R1: 建议 8xA800，CPU 内存至少 1000G |
| SmoothQuant | Wi8Ai8 | 对称 | 对称 | 是 | Qwen3-32B: 4090*1 |
| RTN | Wi8Ai8、Wi4Af16、NVFP4 | 取决于配置 | 取决于配置 | 支持 | Qwen3 / Qwen3-VL / ViT 已有样例 |

**注意：**
- 所有量化算法均不支持 Attention 层量化
- `ostquant` 在当前版本中已经不再采用“两阶段（先 OSTQuant 训练再 GPTQv2 实量化）”流程，直接使用 `ostquant.json + Qwen3_ai8_k.json` 即可完成量化
- 对于 VLM，多子模块量化场景通常需要传入多份 `quant_config` / `alg_config`，或直接使用对应的 `unified_config`
- Qwen3.5 的 GPTQv2 W4A16 配置默认只量化 LLM 子模块，`submodel_0` 保持不量化；`fallback` 会跳过 `lm_head`、`mtp.*` 和 `linear_attn.*`

---

## 2.4 量化示例
### 2.4.1 推荐命令入口：`test/test_unified_config.py`

推荐直接使用统一配置文件。其优势是一个 JSON 同时描述 `quant_config`、`alg_config` 和默认 `calib_dataset_params`，更适合文档化和批量复现。

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/awq_qwen3.json \
  --model-path /data/models/Qwen3-0.6B \
  --dataset-path /data/datasets/cnn_dailymail \
  --save-path /data/quant_models/Qwen3-0.6B-awq \
  --mode Auto
```

### 2.4.2 可选命令入口：`test/test.py`

如需分别切换量化配置、算法配置、采样配置，可使用拆分式入口：

```bash
cd /opt/app
python3 test/test.py \
  --quant-config-file configs/quant_configs/Qwen3_wui4_g.json \
  --alg-config-file configs/alg_configs/awq.json \
  --model-path /data/models/Qwen3-0.6B \
  --dataset-path /data/datasets/cnn_dailymail \
  --calib-dataset-config-file configs/dataset_configs/awq_qwen3_calib.json \
  --save-path /data/quant_models/Qwen3-0.6B-awq \
  --mode Auto \
  --sampling-params-file configs/sampling_configs/llm_default.json
```

### 2.4.3 镜像内已提供的统一配置样例

| 算法 | 统一配置文件 | 说明 |
|------|--------------|------|
| AWQ | `awq_qwen3.json` | Qwen3 LLM |
| AWQ | `awq_qwen3_vl.json` | Qwen3-VL，多子模块 |
| AWQ | `awq_qwen2_5.json` | Qwen2.5 LLM |
| AWQ | `awq_qwen2_5_vl.json` | Qwen2.5-VL |
| AWQ | `awq_intern_vl.json` | InternVL |
| AWQ | `awq_minicpm_v.json` | MiniCPM-V |
| AWQ | `awq_qwen3_reranker.json` | Qwen3-Reranker |
| AWQ | `awq_qwen3_5.json` | Qwen3.5 |
| GPTQv2 | `gptqv2_qwen3.json` | Qwen3，Wi4Ai8 |
| GPTQv2 | `gptqv2_qwen3_vl.json` | Qwen3-VL，视觉 W8A8 + LLM W4A8 |
| GPTQv2 | `gptqv2_qwen2_5_vl.json` | Qwen2.5-VL，视觉 W8A8 + LLM W4A8 |
| GPTQv2 | `gptqv2_per_channel_w4a16_qwen3.json` | Qwen3，Wi4Af16 |
| GPTQv2 | `gptqv2_per_group_w4a16_int4_qwen3.json` | Qwen3，per-group W4A16 |
| GPTQv2 | `gptqv2_qwen3_5_w4a16.json` | Qwen3.5，LLM W4A16 |
| GPTQv2 | `gptqv2_qwen3_5_w4a16_mix.json` | Qwen3.5，LLM W4A16 混合精度 |
| OSTQuant | `ostquant_qwen3.json` | Qwen3，单阶段流程 |
| Quarot | `quarot_qwen3.json` | Qwen3，Wi4Ai8 |
| Quarot | `quarot_deepseek_v3.json` | DeepSeek-R1，Wi4Ai8 |
| RTN | `rtn_per_block_per_group_qwen3.json` | Qwen3，per-block/per-group |
| RTN | `rtn_qwen3_nvfp4.json` | Qwen3，NVFP4 |
| RTN | `rtn_vit.json` | ViT |
| SmoothQuant | `smooth_qwen3.json` | Qwen3，Wi8Ai8 |

### 2.4.4 常用量化命令

#### AWQ：Qwen3

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/awq_qwen3.json \
  --model-path /data/models/Qwen3-0.6B \
  --dataset-path /data/datasets/cnn_dailymail \
  --save-path /data/quant_models/Qwen3-0.6B-awq \
  --mode Auto
```

#### AWQ：Qwen3-VL

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/awq_qwen3_vl.json \
  --model-path /data/models/Qwen3-VL-2B-Instruct \
  --dataset-path /data/datasets/flickr30k_test512 \
  --save-path /data/quant_models/Qwen3-VL-2B-Instruct-awq \
  --mode Auto
```

#### GPTQv2：Qwen3

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/gptqv2_qwen3.json \
  --model-path /data/models/Qwen3-0.6B \
  --dataset-path /data/datasets/wikitext2 \
  --save-path /data/quant_models/Qwen3-0.6B-gptqv2 \
  --mode High
```

#### GPTQv2：Qwen3.5 W4A16

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/gptqv2_qwen3_5_w4a16.json \
  --model-path /data/models/Qwen3.5-2B \
  --dataset-path /data/datasets/HuggingFaceH4/ultrachat_200k \
  --save-path /data/quant_models/Qwen3.5-2B-gptqv2-w4a16 \
  --mode High \
  --torch-dtype auto
```

#### GPTQv2：Qwen3.5 W4A16 混合精度

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/gptqv2_qwen3_5_w4a16_mix.json \
  --model-path /data/models/Qwen3.5-2B \
  --dataset-path /data/datasets/HuggingFaceH4/ultrachat_200k \
  --save-path /data/quant_models/Qwen3.5-2B-gptqv2-w4a16-mix \
  --mode High \
  --torch-dtype auto
```

#### GPTQv2：Qwen3-VL（视觉 W8A8 + LLM W4A8）

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/gptqv2_qwen3_vl.json \
  --model-path /data/models/Qwen3-VL-8B-Instruct \
  --dataset-path /data/datasets/flickr30k_test512 \
  --save-path /data/quant_models/Qwen3-VL-8B-gptqv2 \
  --mode Auto
```

#### OSTQuant：Qwen3

`ostquant` 在当前版本中不再拆分为两阶段。直接使用 `ostquant_qwen3.json` 即可，运行时 `DataSelector` 会返回 `train/eval` 两路 dataloader，`PETQuantizer.run(...)` 一次完成训练和量化。镜像自带样例如下：

```bash
cd /opt/app
torchrun --nnodes 1 --nproc_per_node 4 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/ostquant_qwen3.json \
  --model-path /data/models/Qwen3-0.6B \
  --dataset-path /data/datasets/Salesforce_wikitext \
  --save-path /data/quant_models/Qwen3-0.6B-ostquant \
  --mode Auto
```

#### Quarot：DeepSeek-R1

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/quarot_deepseek_v3.json \
  --model-path /data/models/DeepSeek-R1-BF16 \
  --dataset-path "" \
  --save-path /data/quant_models/DeepSeek-R1-quarot \
  --mode Auto
```

> DeepSeek-R1 模型较大，建议使用 8xA800，CPU 内存至少 1000G。

#### SmoothQuant：Qwen3

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/smooth_qwen3.json \
  --model-path /data/models/Qwen3-32B \
  --dataset-path /data/datasets/pile-val \
  --save-path /data/quant_models/Qwen3-32B-smooth \
  --mode Auto
```

#### RTN：Qwen3 NVFP4

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/rtn_qwen3_nvfp4.json \
  --model-path /data/models/Qwen3-4B \
  --dataset-path /data/datasets/ultrachat_200k \
  --save-path /data/quant_models/Qwen3-4B-rtn-nvfp4 \
  --mode Auto
```

#### RTN：ViT

```bash
cd /opt/app
python3 test/test_unified_config.py \
  --unified-config-file configs/unified_configs/rtn_vit.json \
  --model-path /data/models/vit-base-patch16-224 \
  --dataset-path /data/datasets/quant_data \
  --eval-dataset-config-file configs/dataset_configs/rtn_vit_eval.json \
  --save-path /data/quant_models/vit-base-patch16-224-rtn \
  --mode Auto
```

### 2.4.5 配置文件说明

#### `quant_configs`

文件命名规则为：

```text
<模型架构枚举值>[_w<dtype缩写>_<granularity缩写>][_a<dtype缩写>_<granularity缩写>].json
```

常见示例：

- `Qwen3_wui4_g.json`：Qwen3，权重 `uint4/per_group`，用于 AWQ
- `Qwen3_wi4_c_ai8_k_mse.json`：Qwen3，权重 `int4/per_channel` + 激活 `int8/per_token`，用于 GPTQv2
- `Qwen3_ai8_k.json`：Qwen3，仅激活 `int8/per_token`，用于 OSTQuant
- `Qwen3_wnvfp4_g_anvfp4_g.json`：Qwen3，权重和激活均为 `nvfp4/per_group`
- `Qwen3_VL_wi8_c_ai8_t.json` + `Qwen3_VL_wi4_c_ai8_k.json`：Qwen3-VL 多子模块配置

完整命名规则见 `/opt/app/configs/quant_configs/name.md`。

#### `dataset_configs`

当前镜像中大部分 `*_calib.json` 内容为 `{}`，表示使用 `DataSelector` 的默认参数匹配逻辑；文档中仍建议保留对应文件路径，便于后续按算法或模型扩展。ViT 额外使用 `rtn_vit_eval.json` 作为评测集配置。

#### `sampling_configs`

- `llm_default.json`：LLM 文本生成验证
- `vlm_default.json`：VLM 图文生成验证
- `reranker_default.json`：Reranker 相关性验证

### 2.4.6 Qwen3.5 敏感层分析

`v1.1.1` 镜像新增 LLM 敏感层分析工具，可用于生成混合精度量化所需的敏感层排序。

脚本路径：

```text
/opt/app/tools/llm_sensitivity_analysis/run_llm_sensitivity.py
```

示例命令：

```bash
cd /opt/app
python3 tools/llm_sensitivity_analysis/run_llm_sensitivity.py \
  --model-path /data/models/Qwen3.5-2B \
  --calibration-dataset-path /data/datasets/HuggingFaceH4/ultrachat_200k \
  --unified-config-file tools/llm_sensitivity_analysis/analysis_configs/gptqv2_llm_qwen3_5.json \
  --output-dir /data/analysis_results/Qwen3.5-2B-gptqv2 \
  --mode High
```

输出目录的 `results/` 下会保存敏感层分析结果：

- `topk.txt`：各层敏感度排序及 MAE 值
- `mae_by_layer.png`：各层敏感度趋势图

混合精度量化可参考：

```text
configs/unified_configs/gptqv2_qwen3_5_w4a16_mix.json
```

其中 `fallback` 字段用于指定不量化的层，例如：

```json
"fallback": [
  "lm_head",
  "mtp.*",
  "model.language_model.layers.*.linear_attn.*",
  "model.language_model.layers.(0|2|3|4|5|7)(\\..*)?$"
]
```
## 2.5 指标评估


### 2.5.1 量化模型指标评估
#### 步骤1：进入vllm定制容器，启动vllm服务
```bash
# 进入容器
docker exec -it vllm_custom bash
# 指定GPU
export CUDA_VISIBLE_DEVICES=4,5,6,7 
# 启动vllm openai接口服务
python3 -m vllm.entrypoints.openai.api_server --model /data/llmodels/Qwen3-32B_ostquant_gptqv2_1 --tensor-parallel-size 4 --served-model-name qwen3-32b --trust-remote-code  --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.5 --max-num-seqs 16   --quantization awq_triton_w4a8 --port 8000
```

#### 步骤2：容器外执行评估框架（evalscope）
```python
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen3-32b',
    api_url='http://127.0.0.1:8001/v1/chat/completions',
    eval_type='openai_api',
    datasets=['mmlu'],    
    eval_batch_size=32,
    generation_config={
        'max_tokens': 8192,  # 最大生成token数，建议设置为较大值避免输出截断
        'temperature': 0.7,  # 采样温度 (qwen 报告推荐值)
        'top_p': 0.8,  # top-p采样 (qwen 报告推荐值)
        'top_k': 20,  # top-k采样 (qwen 报告推荐值)
        'n': 1,  # 每个请求产生的回复数量
        'extra_body':{'chat_template_kwargs': {'enable_thinking': False}}  # 关闭思考模式
    },
    timeout=60000,  # 超时时间
    stream=True,  # 是否使用流式输出
    # limit=1000,  # 设置为1000条数据进行测试
)

run_task(task_cfg=task_cfg)
```

---

## 2.6、编译前置注意事项
### 2.6.1 编译前置说明
- ✅ w4a8 模型：**必须修改config.json配置文件** 后才能编译，是核心前置步骤

### 2.6.2 w4a8 模型 config.json 配置修改
打开量化模型目录下的 `Qwen3-32B_ostquant_gptqv2_1/config.json`，替换为以下完整内容：
```json
{
    "architectures": [
        "Qwen3ForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 25600,
    "max_position_embeddings": 40960,
    "max_window_layers": 64,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 1000000,
    "sliding_window": null,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.3",
    "use_cache": false,
    "use_sliding_window": false,
    "vocab_size": 151936,
    "quantization_config": {
        "config_groups": {
            "group_0": {
                "input_activations": {
                    "actorder": null,
                    "block_structure": null,
                    "dynamic": true,
                    "group_size": null,
                    "num_bits": 8,
                    "observer": null,
                    "observer_kwargs": {},
                    "strategy": "token",
                    "symmetric": true,
                    "type": "int"
                },
                "output_activations": null,
                "targets": [
                    "Linear"
                ],
                "weights": {
                    "actorder": null,
                    "block_structure": null,
                    "dynamic": false,
                    "group_size": null,
                    "num_bits": 8,
                    "observer": "mse",
                    "observer_kwargs": {},
                    "strategy": "channel",
                    "symmetric": true,
                    "type": "int"
                }
            }
        },
        "format": "int-quantized",
        "global_compression_ratio": null,
        "ignore": [
            "lm_head"
        ],
        "kv_cache_scheme": null,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    },
    "online-rotate": {
        "online_full_had": true,
        "tp_size": 16
    }
}
```

### 2.6.2 重要注意事项
1. Qwen3-32B 模型的 `config.json` 中 `max_position_embeddings: 40960`，编译时只能设置**低于40960**的cache len，更长上下文推理问题将在后期版本修复。
2. 在X6000设备上运行Qwen3-32B编译后模型，需要手动拷贝原模型的tokenizer相关文件到编译目录：
    ```bash
    cp /data/pipline/original_models/Qwen3-32B/vocab.json /data/pipline/compiled_models2026/目标编译目录/16die/
    cp /data/pipline/original_models/Qwen3-32B/tokenizer* /data/pipline/compiled_models2026/目标编译目录/16die/
    cp /data/pipline/original_models/Qwen3-32B/merges.txt /data/pipline/compiled_models2026/目标编译目录/16die/
    cp /data/pipline/original_models/Qwen3-32B/configuration.json /data/pipline/compiled_models2026/目标编译目录/16die/
    ```
3. 所有量化算法均不支持Attention层量化，是当前版本的固定策略。
4. `ostquant` 当前版本已经取消两阶段流程，但仍需通过 `torchrun` 分布式启动；具体卡数按模型规模和显存情况选择。
5. 量化模型推理时，均通过 `accelerate` 的 `dispatch_model` 做显存调度，避免单卡显存溢出。

<br>
<br>


# 模型编译

本节介绍量化大模型的编译，目前分为语言大模型和视觉语言大模型，编译方式稍有不同，以下通过详细示例代码说明.

### 启动工具链镜像

以下命令创建容器，其中``${your_data_dir}``表示宿主机中用户数据目录，``${version}``需改为实际版本``tag``。
```shell
sudo docker run --gpus all -v ${your_data_dir}:/data -it 113.100.143.90:8091/edgex/tyllm:${version} bash
```

### 语言大模型

以``Qwen3-1.7B-AWQ``为例：

```python
from tyllm.build_util import build_and_compile_llm
from tyllm import torch_edgex
from vllm.config import ModelConfig
ModelConfig.verify_with_parallel_config = lambda a, b: True

quant_path = "./quantized_models/Qwen3-1.7B-AWQ"
aot_path = f"./compiled_models/Qwen3-1.7B-AWQ-AOT_tc1.2.0_20251231"

# 预填充序列长度
prefill_seq_len = 96
# 最大KV键值对数，控制模型推理期间上下文长度
max_kv_cache_size = 8192
# 指定多die编译，多die并行计算
die_num = 4
# 是否将embedding操作作为输入，默认False；如果True，embedding计算将被offload到cpu
embedding_as_input = False

torch_edgex.set_device_mode("page_mode", True)
torch_edgex.set_device_trace_only("edgex", True)
# torch_edgex.set_device_mode("enable_proj_comm", True) 
# torch_edgex.set_device_mode("attn_reduce_groups",[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]) # x6000 group配置
# torch_edgex.set_device_mode("mlp_reduce_groups",[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]) # x6000 group配置
# torch_edgex.set_device_mode("attn_tp_size",16)# 除了qwen3-32B tp16以外的所有模型都不要配置这个变量

build_and_compile_llm(
    model_path=quant_path,
    artifacts_path=f"{aot_path}_{prefill_seq_len}_{max_kv_cache_size}",
    max_kv_cache_size=max_kv_cache_size,
    seq_len_list=[1, prefill_seq_len],
    dev_count=die_num,
    embedding_as_input=embedding_as_input,
)
```

**参数说明**：
- **model_path(str)** ``huggingface``模型和配置文件的路径；
- **max_kv_cache_size(int, optional)** ``kv``缓存的最大容量，默认为``4096``；
- **seq_len_list(list of int, optional)** 用于构建和编译模型的序列长度列表，默认为``[1, 8]``；
- **dev_count(int, optional)** 用于运行已编译模型的设备数量（如 NNP 设备），默认为``1``；
- **artifacts_path(str, optional)** 保存模型编译产物（如权重、嵌入层等）的目录路径。如果未提供，将使用``model_path``作为默认路径；
- **embedding_as_input(bool, optional)** 如果为``True``，将提取嵌入层并单独保存为``NumPy``数组，默认为``False``；

**编译后产物目录**：

```shell
Qwen3-1.7B-AWQ-AOT/
└── 1die
    ├── batch_1
    │   ├── common_die0.params
    │   ├──constant_die0_1.params
    │   ├──constant_die0_8.params
    │   ├── seqlen_1
    │   │   ├── llm_die0.params
    │   │   └── llm_decode_die0.so
    │   └── seqlen_8
    │       ├── llm_die0.params
    │       ├── llm_decode_die0.so
    |       └── llm_prefill_die0.so 
    ├── buffer_config.json
    ├── config.json
    └── empty.bin
```

### 视觉语言大模型

以``Qwen3-VL-4b-AWQ``示例：

```python
import os
import logging
from datetime import datetime 
import numpy as np
import torch
from PIL import Image
from vllm import LLM
from vllm.config import ModelConfig, ParallelConfig
from tyllm import torch_edgex
from pathlib import Path

torch_edgex.set_device_mode('jit_device', 'cpu')
# if torch.cuda.is_available():
#     os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
#     torch_edgex.set_device_mode('jit_device', 'cuda')
# else:
#     torch_edgex.set_device_mode('jit_device', 'cpu')
from tyllm.vllm_ext.edgex_executor import EdgeXExecutor
from vllm.platforms import current_platform
import shutil
import glob
import argparse
from vllm.config import ModelConfig


# 全局初始化配置
ModelConfig.verify_with_parallel_config = lambda a, b: True

args = None
IMAGE_ORG_PATH = "./960_540.jpg" 
# 预处理后的图片路径
IMAGE_PATH = "./test.jpg"

# 设备配置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["COMPILE_THREAD"] = "1"
logging.getLogger("vllm").setLevel(logging.DEBUG)

# 解析命令行参数并初始化全局配置
def parse_args():
    global args
    parser = argparse.ArgumentParser(description="vLLM多模态推理")
    parser.add_argument("--model_dir", type=str, default="./quantized_models/chenmin/qwen3vl-4b-AWQ", help="模型路径")
    parser.add_argument("--num_die", type=int, default=4, help="设备数量")
    parser.add_argument("--input_height", type=int, default=540, help="输入图像高度")
    parser.add_argument("--input_width", type=int, default=960, help="输入图像宽度")
    parser.add_argument("--modality", type=str, default="image", choices=["image", "video"], help="输入模态")
    parser.add_argument("--source_tokenizer", type=str, default="./tokenizer.json", help="原模型tokenizer.json文件路径")
    parser.add_argument("--prefill_lens", type=int, default=96, help="prefill长度")
    parser.add_argument("--max_model_len", type=int, default=8192, help="模型最大kv缓存")
    args = parser.parse_args()

# 参数需在torch_edgex配置前完成
parse_args()
input_size = (args.input_height, args.input_width, 3)

aot_dir = f"./compiled_models/{Path(args.model_dir).name}_{input_size[1]}x{input_size[0]}_{args.max_model_len}_{args.num_die}die_{args.modality}_{datetime.now().strftime('%Y%m%d%H%M')}"

# 配置torch_edgex
torch_edgex.edgex_module.set_trace_only_mode(True)
torch_edgex.set_device_mode("exec_mode", "AOT")
torch_edgex.set_device_mode("eager_on_chip", False)
# torch_edgex.set_device_mode("attn_tp_size", args.num_die) # 除了qwen3-32B tp16以外的所有模型都不要配置这个变量
torch_edgex.set_device_mode("prefill_lens", [1, 8, args.prefill_lens])
torch_edgex.set_device_mode("AOT_DIR", aot_dir)


# 动态修改ParallelConfig
torch._dynamo.reset()
ModelConfig.verify_with_parallel_config = lambda a, b: True
origin_post_init = ParallelConfig.__post_init__

def modified_post_init(self):
    origin_post_init(self)
    self.world_size = args.num_die

ParallelConfig.__post_init__ = modified_post_init

def main():
    global args, aot_dir, IMAGE_PATH
    
    # 图像预处理
    Image.open(IMAGE_ORG_PATH).resize((args.input_width, args.input_height)).save(IMAGE_PATH)

    # 创建目录
    mrope_dir = os.path.join(aot_dir, str(args.num_die)+"die", "mrope")
    visual_dir = os.path.join(aot_dir, str(args.num_die)+"die", "visual")
    for dir_path in [aot_dir, mrope_dir, visual_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 准备输入数据
    modality = args.modality
    if modality == "image":
        data = Image.open(IMAGE_PATH)
    elif modality == "video":
        data = np.array([Image.open(IMAGE_PATH) for _ in range(10)])
    
    question = "请描述图片中的内容"
    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    import vllm.envs as envs
    envs.VLLM_ENABLE_V1_MULTIPROCESSING = False
    envs.VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS = None
    # 初始化模型
    llm = LLM(
        model=args.model_dir,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.num_die,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        disable_mm_preprocessor_cache=True,
        trust_remote_code=True,
        dtype="float16", 
        enforce_eager=True,
        # disable_async_output_proc=True,
        block_size=64,
        distributed_executor_backend=EdgeXExecutor,
        worker_cls="tyllm.vllm_ext.edgex_executor.EdgeXWorker",
        gpu_memory_utilization=0.15,
        # device="cpu"
    )
    
    # 执行编译
    print("执行首次推理以触发AOT编译...")
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {modality: data},
    }
    _ = llm.generate(inputs, use_tqdm=False)
    print("编译完成，开始处理生成的文件...")
    
    # 文件处理
    try:
        # 处理mrope目录
        print(f"处理{mrope_dir}下的文件...")
        mrope_so_files = glob.glob(os.path.join(mrope_dir, "*.so"))
        mrope_params_files = glob.glob(os.path.join(mrope_dir, "*.params"))
        print(f"mrope_so_files={mrope_so_files}")
        print(f"mrope_params_files={mrope_params_files}")
        if mrope_so_files:
            shutil.copy2(mrope_so_files[0], os.path.join(aot_dir, str(args.num_die)+"die", "compute_rope_param.so"))
            print(f"已复制并重命名SO文件: {mrope_so_files[0]} -> {aot_dir}/compute_rope_param.so")
        
        if mrope_params_files:
            shutil.copy2(mrope_params_files[0], os.path.join(aot_dir,  str(args.num_die)+"die", "compute_rope_param.params"))
            print(f"已复制并重命名params文件: {mrope_params_files[0]} -> {aot_dir}/compute_rope_param.params")
        
        # 处理visual目录
        print(f"处理{visual_dir}下的文件...")
        
        # 处理配置文件和so/params文件
        aot_config_files = glob.glob(os.path.join(visual_dir, "*aot_config.json"))
        if aot_config_files:
            os.replace(aot_config_files[0], os.path.join(visual_dir, "aot_config.json"))
        
        buffer_config_files = glob.glob(os.path.join(visual_dir, "*buffer_config.json"))
        if buffer_config_files:
            os.replace(buffer_config_files[0], os.path.join(visual_dir, "buffer_config.json"))
        
        # 处理die0-3的so和params文件
        die_so_map = {f"die{i}.so": f"vit_die{i}.so" for i in range(4)}
        die_params_map = {f"die{i}.params": f"constant_die{i}.params" for i in range(4)}
        
        for src, dst in die_so_map.items():
            files = glob.glob(os.path.join(visual_dir, f"*{src}"))
            if files:
                os.replace(files[0], os.path.join(visual_dir, dst))
        
        for src, dst in die_params_map.items():
            files = glob.glob(os.path.join(visual_dir, f"*{src}"))
            if files:
                os.replace(files[0], os.path.join(visual_dir, dst))
        
        # 复制tokenizer
        if args.source_tokenizer and os.path.exists(args.source_tokenizer):
            shutil.copy2(args.source_tokenizer, os.path.join(aot_dir, str(args.num_die)+"die", "tokenizer.json"))
        
        print("文件处理完成!")
        
    except Exception as e:
        print(f"文件处理过程中发生错误: {e}")

if __name__ == "__main__":
    main()
```

**说明**：
- 文件处理的代码将部分生成产物改名，以匹配云天大模型SDK的文件名要求
- AOT_DIR 请配置为 Qwen3-VL-4B... 这样的模式中间用-隔开，以匹配板上运行云天大模型SDK的目录名要求
- IMAGE_ORG_PATH可以配置本地任意其它图片，只为生成图像输入
- 此脚本GPU环境下整个编译过程有一定加速；如没有GPU可以尝试"COMPILE_THREAD"配置为"2"以多线程加速。

**备注**
- 模型编译时，会执行一次forward的trace过程，如果报缺少config.json、preprocessor_config.json、vocab.json等文件的错误，可能是量化过程文件没有拷贝全，此时从浮点模型目录手动拷贝相应的文件到量化模型目录即可解决。
- 模型编译时，如果因为一些错误导致需要重启任务，建议删除已生成的文件，重新运行脚本。

**编译后产物类似如下结构**：

```shell
Qwen3-VL-4b-AWQ-AOT_960x540_8192_4die_image_01230123$ tree
.
└── 4die
    ├── batch_1
    │   ├── common_die0.params
    │   ├── common_die1.params
    │   ├── common_die2.params
    │   ├── common_die3.params
    │   ├── constant_die0_1.params
    │   ├── constant_die0_8.params
    │   ├── constant_die0_96.params
    │   ├── constant_die1_1.params
    │   ├── constant_die1_8.params
    │   ├── constant_die1_96.params
    │   ├── constant_die2_1.params
    │   ├── constant_die2_8.params
    │   ├── constant_die2_96.params
    │   ├── constant_die3_1.params
    │   ├── constant_die3_8.params
    │   ├── constant_die3_96.params
    │   ├── seqlen_1
    │   │   ├── ckpt_llm_decode_die0.json
    │   │   ├── ckpt_llm_decode_die1.json
    │   │   ├── ckpt_llm_decode_die2.json
    │   │   ├── ckpt_llm_decode_die3.json
    │   │   ├── llm_decode_die0.so
    │   │   ├── llm_decode_die1.so
    │   │   ├── llm_decode_die2.so
    │   │   ├── llm_decode_die3.so
    │   │   ├── llm_die0.params
    │   │   ├── llm_die1.params
    │   │   ├── llm_die2.params
    │   │   └── llm_die3.params
    │   ├── seqlen_8
    │   │   ├── ckpt_llm_decode_die0.json
    │   │   ├── ckpt_llm_decode_die1.json
    │   │   ├── ckpt_llm_decode_die2.json
    │   │   ├── ckpt_llm_decode_die3.json
    │   │   ├── ckpt_llm_die0.json
    │   │   ├── ckpt_llm_die1.json
    │   │   ├── ckpt_llm_die2.json
    │   │   ├── ckpt_llm_die3.json
    │   │   ├── ckpt_llm_prefill_die0.json
    │   │   ├── ckpt_llm_prefill_die1.json
    │   │   ├── ckpt_llm_prefill_die2.json
    │   │   ├── ckpt_llm_prefill_die3.json
    │   │   ├── llm_decode_die0.so
    │   │   ├── llm_decode_die1.so
    │   │   ├── llm_decode_die2.so
    │   │   ├── llm_decode_die3.so
    │   │   ├── llm_die0.params
    │   │   ├── llm_die0.so
    │   │   ├── llm_die1.params
    │   │   ├── llm_die1.so
    │   │   ├── llm_die2.params
    │   │   ├── llm_die2.so
    │   │   ├── llm_die3.params
    │   │   ├── llm_die3.so
    │   │   ├── llm_prefill_die0.so
    │   │   ├── llm_prefill_die1.so
    │   │   ├── llm_prefill_die2.so
    │   │   └── llm_prefill_die3.so
    │   └── seqlen_96
    │       ├── ckpt_llm_die0.json
    │       ├── ckpt_llm_die1.json
    │       ├── ckpt_llm_die2.json
    │       ├── ckpt_llm_die3.json
    │       ├── ckpt_llm_prefill_die0.json
    │       ├── ckpt_llm_prefill_die1.json
    │       ├── ckpt_llm_prefill_die2.json
    │       ├── ckpt_llm_prefill_die3.json
    │       ├── llm_die0.params
    │       ├── llm_die0.so
    │       ├── llm_die1.params
    │       ├── llm_die1.so
    │       ├── llm_die2.params
    │       ├── llm_die2.so
    │       ├── llm_die3.params
    │       ├── llm_die3.so
    │       ├── llm_prefill_die0.so
    │       ├── llm_prefill_die1.so
    │       ├── llm_prefill_die2.so
    │       └── llm_prefill_die3.so
    ├── buffer_config.json
    ├── compute_rope_param.params
    ├── compute_rope_param.so
    ├── config.json
    ├── embedding.params
    ├── empty.bin
    ├── mrope
    │   ├── 3_8192_[int32]_0.onnx
    │   ├── 3_8192_[int32]_1.onnx
    │   ├── 3_8192_[int32]_2.onnx
    │   ├── 3_8192_[int32]_3.onnx
    │   ├── 3_8192_[int32]_aot_config.json
    │   ├── 3_8192_[int32]_buffer_config_0.json
    │   ├── 3_8192_[int32]_buffer_config_1.json
    │   ├── 3_8192_[int32]_buffer_config_2.json
    │   ├── 3_8192_[int32]_buffer_config_3.json
    │   ├── 3_8192_[int32]_buffer_config.json
    │   ├── 3_8192_[int32]_die0.params
    │   ├── 3_8192_[int32]_die0.so
    │   ├── 3_8192_[int32]_die1.params
    │   ├── 3_8192_[int32]_die1.so
    │   ├── 3_8192_[int32]_die2.params
    │   ├── 3_8192_[int32]_die2.so
    │   ├── 3_8192_[int32]_die3.params
    │   ├── 3_8192_[int32]_die3.so
    │   └── 3_8192_[int32]_graph.json
    ├── tokenizer.json
    └── visual
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_0.onnx
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_1.onnx
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_2.onnx
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_3.onnx
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_buffer_config_0.json
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_buffer_config_1.json
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_buffer_config_2.json
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_buffer_config_3.json
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_graph.json
        ├── 2040_1536_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]2040_1024_[float16]_preset_kwargs.pt
        ├── aot_config.json
        ├── buffer_config.json
        ├── constant_die0.params
        ├── constant_die1.params
        ├── constant_die2.params
        ├── constant_die3.params
        ├── vit_die0.so
        ├── vit_die1.so
        ├── vit_die2.so
        └── vit_die3.so
```




## 常见问题

若在使用产品过程中遇到问题，可以参考此文档。

### 1. 编译阶段出现 RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

**问题描述**：
GPU环境须限制编译线程环境变量

**解决方法**：
将 os.environ["COMPILE_THREAD"] 配置为 "1"即可解决。

### 2. 编译阶段出现 Condition: status == DCL_ERROR_REPEAT_INITIALIZE failed

**问题描述**：
检测到GPU设备时，编译脚本对全局设置有顺序要求

**解决方法**：
保持文档中编译脚本各操作先后顺序不做改动。
