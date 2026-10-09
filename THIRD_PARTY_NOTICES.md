# 上游来源与许可说明

EdgeMind v1.0 借鉴成熟的小模型训练流程，以尽快完成从零预训练、对话微调到端侧部署的可见成品，并积累工程经验。以下说明保留复用资源的来源；项目设计、验证状态与后续路线见 [README](README.md)。

## 复用与改写的资源

| 内容 | 上游来源 | EdgeMind 中的使用方式 |
| --- | --- | --- |
| 分词器文件 | [MiniMind 的固定提交](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f/model) | 复用 `model/tokenizer.json`、`model/tokenizer_config.json`；不使用上游模型权重初始化词嵌入 |
| 对话模板 | [MiniMind-3 官方模板](https://huggingface.co/jingyaogong/minimind-3/blob/main/chat_template.jinja) | 复用 `model/chat_template.jinja`，文件头保留来源 |
| 模型与数据处理思路 | [MiniMind 项目](https://github.com/jingyaogong/minimind) | 参考 Decoder、配置、样本组织与助手标签规则；按 EdgeMind 的 24 层、1024 宽度和当前任务改写 |
| 训练与生成流程 | [MiniMind 项目](https://github.com/jingyaogong/minimind) | 参考分阶段训练及采样流程；采用本项目的单卡训练循环、权重格式与必要检查 |
| 第一阶段训练语料 | [公开数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset) | 使用 `pretrain_t2t_mini.jsonl`；计划使用 `sft_t2t_mini.jsonl`。正式语料不随代码仓库分发 |

分词器固定提交来自本项目已有的来源记录；聊天模板同时保留文件头的上游链接。新增的 SVG 标识为 EdgeMind 原创，不复用上游项目的图形标识。

## 代码与模板

[MiniMind 上游代码许可证](https://github.com/jingyaogong/minimind/blob/master/LICENSE)为 Apache License 2.0；[官方模型页](https://huggingface.co/jingyaogong/minimind-3)也标注 Apache-2.0。本仓库保留 [Apache-2.0 许可全文](LICENSES/Apache-2.0.txt)，以及相关文件中的来源说明和 EdgeMind 改写标记。

这份上游许可副本用于说明复用内容的许可，不是为本仓库所有原创新增内容统一指定许可证。EdgeMind 原创新增内容尚未另行声明整体许可证。

## 数据与权重

[上游数据卡](https://huggingface.co/datasets/jingyaogong/minimind_dataset/blob/main/README.md)同时列出 Apache-2.0 与 CC-BY-NC-2.0。数据来源条件独立于代码许可；使用、再分发数据或发布训练权重时，应保留并核对对应来源的条件。README 不将全部语料或未来权重统一标为可商用。

当前仓库只包含少量格式检查样本，正式语料、训练日志和模型权重由 `.gitignore` 排除。两条 SFT 检查样本用于展示本项目对话与工具字段的组织方式，不代表真实工具服务已接入。
