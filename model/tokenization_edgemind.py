# 复用 MiniMind 已训练的分词器，本文件只负责本地加载和与模型配置的必要核对。
# 两个分词器 JSON 文件原样取自下面的固定提交，便于以后复现本次使用的词表版本。
# 来源：https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f/model

# 使用标准库定位分词器文件，使加载过程不依赖运行脚本时的工作目录。
from pathlib import Path

# 复用现成的分词器加载和编码能力，不重新实现或训练分词算法。
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# 读取 EdgeMind 的词表大小和特殊标记编号，核对它们是否与分词器一致。
from .configuration_edgemind import EdgeMindConfig


# 加载项目自带的分词器；配置不匹配时立即报错，避免错误编号进入模型训练。
def load_tokenizer(config: EdgeMindConfig) -> PreTrainedTokenizerFast:
    # 两个 JSON 文件与本文件放在同一个 model 目录，直接按本文件位置查找。
    tokenizer_directory = Path(__file__).resolve().parent

    # 只读取本地文件，训练时不需要再次联网下载分词器。
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory), local_files_only=True)

    # 核对完整词表大小以及起始、结束、填充标记的编号。
    for name in ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id"):
        # len 包含新增标记，能避免只检查基础词表而漏掉模型无法接收的编号。
        actual = len(tokenizer) if name == "vocab_size" else getattr(tokenizer, name)
        # 从模型配置读取当前字段应有的值。
        expected = getattr(config, name)

        # 编号或词表大小不一致时停止加载，并明确指出是哪一项不匹配。
        if actual != expected:
            # 同时报告实际值与配置值，方便定位需要调整的设置。
            raise ValueError(f"分词器与模型配置不一致：{name}，分词器为 {actual}，配置为 {expected}。")

    # 固定从文本右侧补齐长度，供后续数据集构造对应的注意力掩码。
    tokenizer.padding_side = "right"
    # 将分词器的默认长度上限对齐模型；实际截断仍需后续数据集明确启用。
    tokenizer.model_max_length = config.max_position_embeddings

    # 返回核对通过的现成分词器，供后续数据读取和文本生成使用。
    return tokenizer
