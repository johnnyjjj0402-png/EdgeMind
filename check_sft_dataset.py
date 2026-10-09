# 验证对话样本的构造：打印样本统计，并逐词元标出哪些位置参与训练损失。
# 本脚本只使用 CPU 做数据读取和分词，不创建模型、不训练。

# 读取模型结构配置，分词器加载时需要核对词表大小与特殊标记编号。
from model.configuration_edgemind import EdgeMindConfig

# 复用项目统一的分词器加载入口，保证验证用的分词器与训练时完全一致。
from model.tokenization_edgemind import load_tokenizer

# 使用刚实现的对话数据集，验证对象是冒烟数据文件。
from dataset.lm_dataset import SFTDataset


# 读取第一条样本，打印整体统计和前若干词元的学习标记。
def main():
    # 使用与训练一致的配置和分词器，避免验证环境与真实环境不一致。
    config = EdgeMindConfig()
    tokenizer = load_tokenizer(config)

    # 构造数据集；样本长度用默认的一千零二十四。
    dataset = SFTDataset("dataset/sft_smoke.jsonl", tokenizer)

    # 读取第一条（带工具调用的完整对话）。
    sample = dataset[0]

    # 统计有效词元数和参与损失的词元数，先看整体是否合理。
    real_tokens = int(sample["attention_mask"].sum())
    learned_tokens = int((sample["labels"] != -100).sum())
    print(f"样本数={len(dataset)}, 有效词元={real_tokens}, 参与损失的词元={learned_tokens}")

    # 逐个打印有效词元：编号、还原的文本、是否参与损失。
    print("逐词元检查（学 = 参与损失，不学 = 忽略）：")
    for index in range(real_tokens):
        # 单个词元还原成文本，便于肉眼对照。
        text = tokenizer.decode([int(sample["input_ids"][index])])
        # 标签为 -100 表示该位置被忽略。
        flag = "学" if sample["labels"][index] != -100 else "不学"
        print(f"{index:4d} {text!r:24} {flag}")


# 直接运行时执行验证；被导入时不产生任何动作。
if __name__ == "__main__":
    main()
