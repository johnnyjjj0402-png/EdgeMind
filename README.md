<p align="center">
  <img src="docs/assets/edgemind-banner.svg" alt="EdgeMind" width="880">
</p>

<h2 align="center"><em>A journey of a thousand miles begins with a single step.</em></h2>
<p align="center">
  <sub>— Laozi, <a href="https://ctext.org/dao-de-jing/zh"><em>Tao Te Ching</em>, Chapter 64</a> · English rendering</sub>
</p>

<p align="center"><strong>0.3B · 走向端侧 · 探索自我改进</strong></p>
<p align="center">
  <a href="#两条主线">两条主线</a> ·
  <a href="#当前进度">当前进度</a> ·
  <a href="#架构设计">架构设计</a> ·
  <a href="#借鉴与来源">借鉴与来源</a>
</p>

**EdgeMind 是一个从零训练、面向端侧的约 0.3B 语言模型项目：先建立能力与效率的基线，再研究小模型能否学会改进自己。**

**正在做：第二轮预训练，使用完整语料继续训练已有 Base。首轮预训练与一轮 mini SFT 已完成。**

## 两条主线

| 主线 | 目标 | 要交付的结果 |
| --- | --- | --- |
| **v1.0 · 端侧能力与效率** | 预训练 → SFT（示范微调）→ RL（强化学习）→ 量化与真机部署 | 与同量级参数模型比较任务成功率、首次响应时间、生成速度和内存占用 |
| **v2.0 · 0.3B 的 RSI 探索** | 让模型分析失败、提出学习计划，并通过训练更新自身权重 | 检验多轮迭代后，模型是否更强、它提出的学习计划是否更有效 |

RSI（Recursive Self-Improvement，递归自我改进）的研究设想是：**模型做任务 → 分析失败 → 决定练什么、练多难 → 验证与训练 → 新模型安排下一轮学习。** 在相同预算下比较固定计划与模型生成的计划；首版计划在 AutoDL 更新权重，再部署到端侧。

v1.0 的 RL 任务、反馈与评测将为 v2.0 提供实验基础。RSI 的具体方法与效果待研究验证。

> [!NOTE]
> **感谢开源贡献者。** 首版参考 [MiniMind](https://github.com/jingyaogong/minimind) 作者 [Jingyao Gong](https://github.com/jingyaogong) 及社区的模型组件与训练思路，复用分词器、对话模板和阶段语料，以尽快获得可见成品、积累完整流程经验。
>
> 同时感谢 [PyTorch](https://github.com/pytorch/pytorch)、[Transformers](https://github.com/huggingface/transformers)、[Tokenizers](https://github.com/huggingface/tokenizers)、[Jinja](https://github.com/pallets/jinja)、[Python](https://www.python.org/) 及相关库、数据和社区贡献者。复用范围与许可见 [来源说明](THIRD_PARTY_NOTICES.md)及[借鉴与来源](#借鉴与来源)。

## 当前进度

| 状态 | 阶段 | 进展 |
| --- | --- | --- |
| ✅ 已完成 | v0.1 Base | 首轮 mini 预训练、权重保存与重载、文本续写 |
| ✅ 已完成 | v0.2 首轮 SFT | 使用 `sft_t2t_mini.jsonl` 完成一轮示范微调 |
| 🔄 进行中 | 第二轮预训练 | 从已有 Base 出发，使用 `pretrain_t2t.jsonl` 完整语料续训 |

**下一步：新 Base 的 SFT → RL → 量化与端侧部署 → 同量级模型对比。** 对比将统一设备、量化精度、上下文长度和测试任务，并记录每个训练阶段的能力变化。

目前尚未完成 RL、端侧对比或 RSI 实验；首轮续写仍有常识与连贯性问题，后续以独立评测记录改进。

## 架构设计

**24 层 × 1024 隐藏宽度，277,136,384 个参数。** 核心组件使用手写 PyTorch，实现 Dense Decoder；共享输入与输出权重以控制词表开销。宽深配置作为实验基线，收益由对照评测确认。

<details>
<summary><strong>展开模型配置与实现入口</strong></summary>

| 配置 | 当前值 |
| --- | --- |
| Decoder | RMSNorm · GQA · RoPE · SwiGLU |
| 前馈宽度 / 查询头 / 键值头 | 2816 / 16 / 4 |
| 注意力计算 | PyTorch SDPA |
| 词表 / 权重共享 | 6400 / 输入与输出共享 |
| 位置上限 / RoPE 基数 | 2048 / `theta=1e6` |
| 默认训练长度 | 预训练 512 / SFT 1024；位置上限不代表已验证的长文本能力 |

- [模型配置](model/configuration_edgemind.py)与[模型计算](model/modeling_edgemind.py)：结构约束、前向计算与预测损失。
- [预训练](trainer/train_pretrain.py)：已有 Base 加载、学习率预热与衰减、定期保存和中断恢复。
- [SFT](trainer/train_sft.py)与[数据处理](dataset/lm_dataset.py)：对话模板、工具字段与仅助手输出参与损失的标签；模板支持不代表工具执行能力已验证。
- [续写测试](eval_pretrain.py)：温度与 `top_p` 采样。
- [能力增强调研](docs/small_model_enhancement.md)：保留的研究资料，当前路线以首页两条主线为准。

</details>

## 更新日志

| 日期 | 进展 |
| --- | --- |
| **2026-10-14** | 第二轮预训练进行中；明确 v1.0 端侧能力与效率对比、v2.0 低参数 RSI 研究两条主线 |
| 2026-10-13 | 首轮 mini SFT 已完成；已接入 SDPA、Base 续训与完整状态恢复，修正学习率起点与日志 |
| 2026-10-09 | v0.2 接入对话模板、SFT 数据与训练入口、模板和标签检查 |
| 2026-10-07—08 | v0.1 完成首轮 Base 预训练，跑通权重重载与续写，补充温度和 `top_p` 采样 |

## 借鉴与来源

复用与改写范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，相关文件保留上游出处。MiniMind 复用内容保留 [Apache-2.0 许可全文](LICENSES/Apache-2.0.txt)；其他依赖遵循各自许可，本仓库尚未为所有原创新增内容统一声明整体许可证。

训练数据有独立的来源条件，上游数据卡同时列出 Apache-2.0 与 CC-BY-NC-2.0，不能用代码许可覆盖全部数据或未来权重。正式语料与训练权重不随本代码仓库分发。首页 SVG 标识为 EdgeMind 原创。
