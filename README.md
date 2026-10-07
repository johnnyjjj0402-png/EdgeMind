# EdgeMind

专注于小参数大语言模型能力增强，面向端侧的小参数大语言模型项目：从零预训练，逐步探索小模型能力增强以及其在特定任务，Agent场景中的能力上限，再验证端侧部署后的效果与成本。采用小步实现，v0.1版本目前参考 [MiniMind](https://github.com/jingyaogong/minimind) 的实践，首先搭建完整的预训练流程。

当前版本：**v0.1 Base（预训练）**。已实现 Base 模型、单卡 GPU 预训练脚本和文本续写测试脚本；正式 GPU 训练效果待验证，后训练、量化与手机部署尚未实现。

## 当前模型

- 手写 Dense Decoder-only 模型：24 层、隐藏宽度 1024，约 0.277B 参数。
- 使用 RMSNorm、RoPE、分组查询注意力和 SwiGLU，共享输入词嵌入与输出权重。
- 复用 MiniMind 的 6400 词元 BPE 分词器，词嵌入从随机权重开始训练。
- 模型位置上限 2048；当前预训练默认每条样本总长 512。

```text
model/configuration_edgemind.py   模型结构配置
model/modeling_edgemind.py        模型计算
model/tokenization_edgemind.py    本地分词器加载
model/tokenizer*.json             现成的分词器文件
dataset/lm_dataset.py             读取、分词和补齐预训练样本
trainer/train_pretrain.py         单卡 GPU 预训练
eval_pretrain.py                  加载训练权重，测试文本续写
```

## 在 AutoDL 上准备

选择带 CUDA PyTorch 的 GPU 镜像，建议 Python 3.11。若租用 RTX 5090，应选择支持 Blackwell 的镜像，例如 PyTorch 2.7 / CUDA 12.8 或更新的兼容组合（[PyTorch 官方说明](https://pytorch.org/blog/pytorch-2-7/)）。

以下命令在 AutoDL 的终端运行，代码与数据放在 [数据盘 `/root/autodl-tmp`](https://www.autodl.com/docs/env/)。

```bash
cd /root/autodl-tmp
git clone https://github.com/johnnyjjj0402-png/EdgeMind.git
cd EdgeMind

# 保留镜像已经安装的 GPU PyTorch，只安装模型配置与分词器需要的库。
python -m pip install "transformers==5.17.0"

# 确认当前 Python 能使用 GPU，并支持本项目采用的 BF16 精度。
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU 可用:', torch.cuda.is_available()); print('BF16 可用:', torch.cuda.is_bf16_supported())"
```

本地检查使用 Transformers 5.17.0；租用 GPU 后仍需实际验证上述镜像与运行环境。

## 下载第一批语料

使用 MiniMind 的 [`pretrain_t2t_mini.jsonl`](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/pretrain_t2t_mini.jsonl)，约 1.24 GB。只下载这一个文件，不克隆整个数据集仓库。

```bash
mkdir -p /root/autodl-tmp/datasets
curl -fL --retry 3 \
  https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main/pretrain_t2t_mini.jsonl \
  -o /root/autodl-tmp/datasets/pretrain_t2t_mini.jsonl
```

数据是一行一条 JSON 记录，至少包含非空 `text` 字段，例如：

```json
{"text": "人工智能可以用于处理自然语言。"}
```

项目中的 `dataset/pretrain_smoke.jsonl` 只有 64 条记录，用于检查数据格式，不作为正式预训练语料。

## 开始预训练

在项目根目录运行：

```bash
python -u -m trainer.train_pretrain \
  --data_path /root/autodl-tmp/datasets/pretrain_t2t_mini.jsonl \
  --save_dir /root/autodl-tmp/edgemind/checkpoints \
  --epochs 1 \
  --batch_size 1 \
  --max_length 512
```

每批更新一次模型，不累积多批梯度；`batch_size=1` 是保守起点，显存与速度需在租用的 GPU 上实测后调整。每十批打印一次当前批次的 `loss`，它表示预测下一词元的误差。

每轮结束保存 `pretrain_epoch_1.pth` 等文件，内容是模型权重和结构配置。当前没有轮内保存点，也没有中断后恢复训练的功能。

## 测试预训练效果

```bash
python eval_pretrain.py \
  --weights /root/autodl-tmp/edgemind/checkpoints/pretrain_epoch_1.pth \
  --prompt "人工智能是一种" \
  --max_new_tokens 100
```

脚本每次选择分数最高的下一个词元，接到文本后继续预测，遇到结束标记或达到长度限制就停止。输入开头加起始标记，不加结束标记；这是文本续写，不使用聊天模板。

可以固定几个不来自训练语料的文本开头，例如“人工智能是一种”“春天到了，”“要读取一个文本文件，可以”，用相同开头比较不同轮次的权重。先观察是否能连贯续写、是否反复重复；这只提供直观检查，不能代替独立测试集上的定量评估。

## 来源与范围

分词器文件来自 [MiniMind 的固定提交](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f/model)；模型和训练流程参考其设计，按当前需要逐步实现。分词器、语料等上游资源的来源与使用条件以各自项目说明为准。

Git 仓库只保存代码、分词器和小样本；正式语料与训练权重由 `.gitignore` 排除。

## 更新日志

### v0.1 — Base 预训练（2026-10-07）

- 实现约 0.277B 参数的 Dense 模型，复用 MiniMind 的 BPE 分词器，模型权重从随机初始化开始训练。
- 实现预训练数据读取、注意力掩码和填充位置的损失忽略。
- 实现单卡 GPU 预训练：每批更新一次权重，打印训练误差，每轮结束保存权重与模型配置。
- 新增文本续写测试脚本，以及 AutoDL 环境准备、语料下载、训练和测试说明。

当前状态：已完成本地加载与生成逻辑验证，正式 AutoDL GPU 预训练和效果评估待完成。本版保存文件用于加载测试，暂不支持中断后恢复训练。
