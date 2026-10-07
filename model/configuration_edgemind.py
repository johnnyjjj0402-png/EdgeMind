# 参考 MiniMind 的配置方式：手写模型参数，仅复用 Transformers 的配置保存与加载能力。
# 只导入配置基类，它不构建模型，也不加载预训练权重。
from transformers import PretrainedConfig


# 配置类只描述模型结构；学习率、训练数据路径和批大小留给后续训练代码管理。
# 沿用 MiniMind 的继承方式，避免重复编写 JSON 序列化逻辑。
class EdgeMindConfig(PretrainedConfig):
    # 写入配置文件的模型类型标识，用于区分 EdgeMind 与其他模型。
    model_type = "edgemind"

    # 显式保留宽度与层数，其他参数像 MiniMind 一样通过关键字传入。
    def __init__(self, hidden_size: int = 1024, num_hidden_layers: int = 24, **kwargs):
        # 交给基类处理通用字段，使保存后的配置能够重新加载。
        super().__init__(**kwargs)

        # 残差流的宽度，决定每个 token 在模型内部用多少个数表示。
        self.hidden_size = hidden_size
        # 堆叠的 DecoderBlock 数量；默认主模型用 24 层，调试时可传入 8。
        self.num_hidden_layers = num_hidden_layers
        # 默认匹配 MiniMind 的现成词表，也决定嵌入矩阵的行数。
        self.vocab_size = kwargs.get("vocab_size", 6400)

        # 查询头数量；默认把 1024 维拆为 16 个 64 维的头。
        self.num_attention_heads = kwargs.get("num_attention_heads", 16)
        # 键和值各使用 4 个头，使每组 4 个查询头共享一套键和值。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        # SwiGLU 的中间宽度；这是当前参数预算的候选值，可独立调整。
        self.intermediate_size = kwargs.get("intermediate_size", 2816)

        # 首版计划支持的上下文长度，不能仅靠调大此值获得长文本能力。
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)
        # SwiGLU 门控分支使用的激活函数名称，供后续前馈层读取。
        self.hidden_act = kwargs.get("hidden_act", "silu")
        # RMSNorm 分母中的稳定项，防止均方值接近零时出现数值问题。
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # RoPE 频率的基数，沿用 MiniMind-3 的设置，首版不启用位置外推。
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        # 默认不随机丢弃激活，保持当前基线的计算行为简单。
        self.dropout = kwargs.get("dropout", 0.0)
        # 请求输入嵌入与输出投影共享权重，实际绑定由后续模型代码完成。
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)

        # MiniMind 分词器的起始标记编号；配置该编号不会自动在输入前插入标记。
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        # MiniMind 分词器的结束标记编号，后续生成逻辑可用它判断何时停止。
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        # MiniMind 分词器的填充标记编号；填充位置的注意力与损失屏蔽需另行实现。
        self.pad_token_id = kwargs.get("pad_token_id", 0)

        # 集中列出必须为正整数的结构参数，避免零个头、负层数或小数维度进入模型。
        dimensions = (
            # 检查隐藏宽度、层数和词表大小，保证基本结构可以创建。
            self.hidden_size, self.num_hidden_layers, self.vocab_size,
            # 在进行除法前检查查询头与键值头数量，避免除零错误。
            self.num_attention_heads, self.num_key_value_heads,
            # 检查前馈层宽度和上下文长度，保证张量维度有效。
            self.intermediate_size, self.max_position_embeddings,
        # 完成待检查参数的收集；这里只验证结构，不分配模型权重。
        )

        # 布尔值也会被拒绝，防止把 True 误当成整数维度 1。
        if any(type(value) is not int or value <= 0 for value in dimensions):
            # 尽早报告无效配置，避免在后续张量运算中才失败。
            raise ValueError("模型宽度、层数、词表大小、头数、前馈宽度和上下文长度必须是正整数。")

        # 当前设计要求查询头均分隐藏维度，才能正确拆分注意力张量。
        if self.hidden_size % self.num_attention_heads != 0:
            # 提示需要一起调整的两个参数。
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除。")

        # GQA 要求每个键值头服务相同数量的查询头。
        if self.num_attention_heads % self.num_key_value_heads != 0:
            # 防止产生无法均匀分组的注意力结构。
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除。")

        # 每个头的维度始终由宽度和查询头数推导，加载配置时也重新计算。
        self.head_dim = self.hidden_size // self.num_attention_heads

        # RoPE 将特征维度成对旋转，因此每个头必须包含偶数个特征。
        if self.head_dim % 2 != 0:
            # 将位置编码的形状约束提前到配置阶段检查。
            raise ValueError("head_dim 必须为偶数，才能应用 RoPE。")
