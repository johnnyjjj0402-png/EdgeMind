# 参考 MiniMind 的模型文件组织方式：全部模型组件按依赖顺序放在同一文件中。
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从当前 model 包导入配置类，使项目根目录中的训练脚本能够正常导入模型并读取结构参数。
from .configuration_edgemind import EdgeMindConfig

# 按配置中的名称取用激活函数，与 MiniMind 保持同一来源。
from transformers.activations import ACT2FN

# 带 LM 头的模型继承 PreTrainedModel，复用其权重保存与加载能力。
from transformers import PreTrainedModel


# 每个 DecoderBlock 内部要使用两次 RMSNorm，先把它作为独立组件实现并验证。
class EdgeMindRMSNorm(nn.Module):

    # 只需要特征宽度和数值稳定项，后续建模时从 EdgeMindConfig 的对应字段读取。
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        # 调用基类初始化，使本模块的参数能被模型统一保存和移动到 GPU。
        super().__init__()

        # 每个特征维度一个可学习缩放系数，初始化为 1 表示开局不做额外缩放。
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 加在分母均方值上的小量，防止某一步输入全零时发生除零。
        self.eps = eps

    # 输入形状为 (..., hidden_size)，归一化只沿最后一个特征维度进行。
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 记录输入精度，归一化结束后恢复，避免破坏混合精度训练的dtype约定。
        input_dtype = hidden_states.dtype

        # 转到 float32 再计算平方和，半精度下平方容易上溢或损失精度。
        hidden_states = hidden_states.to(torch.float32)
        # 沿特征维求均方值；keepdim 保留维度以便下一步广播相乘。
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 乘以均方根的倒数，把每个位置的特征向量长度缩放到 1 附近。
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 先在 float32 中乘可学习缩放系数，避免提前转回低精度造成额外舍入。
        # 最后将完整结果恢复为输入精度，防止 float32 权重把 BF16 或 FP16 输出重新提升为 float32。
        return (self.weight * hidden_states).to(input_dtype)


# 预计算 RoPE 的角度表：每个位置、每对特征维度对应一个 cos 和一个 sin 值。
# 建表只发生一次，之后训练与推理都按位置切片查表，避免重复计算三角函数。
def precompute_freqs_cis(dim: int, end: int, theta: float = 1e6) -> tuple[torch.Tensor, torch.Tensor]:
    # 每对特征共享一个频率，因此只需要 dim//2 个频率值。
    # 频率按几何级数递减：越靠后的特征对旋转越慢，负责表达更长的相对距离。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 生成 0 到 end-1 的位置编号，end 取配置中的 max_position_embeddings。
    positions = torch.arange(end).float()

    # 外积得到形状 (end, dim//2) 的角度表：第 i 行是位置 i 的全部旋转角度。
    angles = torch.outer(positions, freqs)

    # 把角度表沿特征维复制拼接成 dim 列，配合下面的 rotate_half 完成对全部特征的旋转。
    freqs_cos = torch.cat([angles, angles], dim=-1).cos()
    freqs_sin = torch.cat([angles, angles], dim=-1).sin()

    # 两张表形状均为 (end, dim)，使用时按当前序列位置切片即可。
    return freqs_cos, freqs_sin


# 把旋转施加到查询和键上；沿用 MiniMind 的布局约定，q/k 形状为 (batch, seq_len, heads, head_dim)。
# cos/sin 需先按当前序列位置切成 (seq_len, head_dim) 再传入。
def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    # 构造每对特征的伴随向量：对半切开后交换，并给原后半段加负号。
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # 取出前半段特征，即每对旋转特征中的第一个分量。
        x1 = x[..., : x.shape[-1] // 2]
        # 取出后半段特征，即每对旋转特征中的第二个分量。
        x2 = x[..., x.shape[-1] // 2:]
        # 拼接为 (-x2, x1)，与复制过的 cos/sin 表配合正好等价于逐对旋转公式。
        return torch.cat((-x2, x1), dim=-1)

    # 在批次维和头维处补 1，使 (seq_len, head_dim) 能广播到 (batch, seq_len, heads, head_dim)。
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 旋转公式 x' = x·cos + rotate_half(x)·sin；q 和 k 必须使用同一张角度表，相对位置才成立。
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    # 输出形状与输入完全一致，只是每个位置的向量被旋转了对应角度。
    return q_embed, k_embed


# 把键值头沿头维重复，使键值头数与查询头数对齐；这是 GQA 共享键值的实现方式。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # 读出现有形状；x 的布局与注意力内部一致，为 (batch, seq_len, kv_heads, head_dim)。
    bs, slen, num_key_value_heads, head_dim = x.shape

    # 每组只有一个查询头时无需重复，直接返回原张量。
    if n_rep == 1:
        return x

    # 在头维后插入一个新维度并扩展，把每个键值头复制 n_rep 份。
    expanded = x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim)
    # 把 (kv_heads, n_rep) 两维展平，得到 kv_heads * n_rep 个头。
    return expanded.reshape(bs, slen, num_key_value_heads * n_rep, head_dim)


# GQA 注意力层：参考 MiniMind 的手动计算分支，首版不启用 flash 分支与 QK-norm。
class EdgeMindAttention(nn.Module):

    # 结构参数全部从 EdgeMindConfig 读取，配置阶段已保证头数的整除关系。
    def __init__(self, config: EdgeMindConfig):
        # 调用基类初始化，使四个投影矩阵能被模型统一管理。
        super().__init__()

        # 查询头数量。
        self.n_local_heads = config.num_attention_heads
        # 键值头数量；配置为 None 时退化为标准多头注意力。
        self.n_local_kv_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        # 每个键值头要服务的查询头数，即键值头的重复次数。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，配置阶段已推导并保证为偶数。
        self.head_dim = config.head_dim

        # 查询投影：hidden_size 维映射到全部查询头。
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        # 键投影：只映射到较少的键值头，这是 GQA 省参数、省 KV cache 的来源。
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        # 值投影：与键投影同宽。
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        # 输出投影：把拼接后的全部查询头映射回 hidden_size 维。
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # 注意力权重上的 dropout；配置默认为 0，此时等价于恒等映射。
        self.attn_dropout = nn.Dropout(config.dropout)
        # 输出投影后的 dropout，同样默认不生效。
        self.resid_dropout = nn.Dropout(config.dropout)

    # x 形状 (batch, seq_len, hidden_size)；cos/sin 按新 token 的位置切片，历史位置不需要。
    # past_key_value 为之前步骤缓存的 (键, 值)；use_cache 控制是否把更新后的缓存返回给调用方。
    # attention_mask 形状 (batch, kv总长度)，1 表示真实 token、0 表示 padding，用于遮住填充位。
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_key_value: tuple = None, use_cache: bool = False, attention_mask: torch.Tensor = None):
        # 读取批次大小与序列长度，供后续 reshape 使用。
        bsz, seq_len, _ = x.shape

        # 三个线性投影，分别得到查询、键、值。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 把最后一维拆成 (头数, 头维度)，布局变为 (batch, seq_len, heads, head_dim)。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 对查询和键施加旋转位置编码；值不携带位置信息，不参与旋转。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 拼接历史键值：缓存中的键早已旋转过，直接沿序列维接在新键值前面。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 需要继续生成时，把拼接后的键值传出去；训练时不返回以节省显存。
        past_kv = (xk, xv) if use_cache else None

        # 把键和值的头数重复到与查询头数一致，同组查询头共享同一份键值。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 把头维提到序列维之前，变为 (batch, heads, seq_len, head_dim) 以做批量矩阵乘。
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        # 计算注意力分数并除以根号头维度；转 float32 保证后续 softmax 数值稳定。
        scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
        # 因果掩码只加在最后 seq_len 列：新 token 可见全部历史，但彼此之间仍只能看过去。
        scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
        # padding 掩码：把填充位置的分数压到接近 -inf，使 softmax 后的权重近似为零。
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        # softmax 得到注意力权重，转回输入精度后按需施加 dropout。
        probs = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq))
        # 用注意力权重对值加权求和，得到每个位置的注意力输出。
        output = probs @ xv

        # 把头维并回特征维，恢复 (batch, seq_len, hidden_size)。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # 经过输出投影与 dropout，与更新后的缓存一起返回。
        return self.resid_dropout(self.o_proj(output)), past_kv


# SwiGLU 前馈层：参考 MiniMind 的 FeedForward，三个无偏置投影加门控激活。
class EdgeMindFeedForward(nn.Module):

    # 只读取隐藏宽度和中间宽度；中间宽度是参数预算的主要调节旋钮。
    def __init__(self, config: EdgeMindConfig):
        # 调用基类初始化，使三个投影矩阵被模型统一管理。
        super().__init__()

        # 门控分支投影：把 hidden_size 维映射到中间宽度，输出要经过激活函数。
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 内容分支投影：与门控分支同形，输出不经过激活函数。
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 降维投影：把逐元素相乘的结果映射回 hidden_size 维。
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 按配置读取激活函数，默认为 SiLU；与门控结构组合即为 SwiGLU。
        self.act_fn = ACT2FN[config.hidden_act]

    # 输入输出形状相同，均为 (..., hidden_size)；前馈层不做任何位置间交互。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门控分支经激活后与内容分支逐元素相乘，再降维回隐藏宽度。
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# DecoderBlock：把一个 GQA 注意力子层和一个 SwiGLU 前馈子层按 pre-norm 结构组装。
class EdgeMindBlock(nn.Module):

    # 两个子层加两个归一化层，结构参数全部来自配置。
    def __init__(self, config: EdgeMindConfig):
        # 调用基类初始化，使子层参数被模型统一管理。
        super().__init__()

        # 注意力子层：负责跨位置交换信息。
        self.self_attn = EdgeMindAttention(config)
        # 前馈子层：负责逐位置加工信息。
        self.mlp = EdgeMindFeedForward(config)
        # 注意力前的归一化；pre-norm 结构中归一化只作用于子层输入，不动残差主干。
        self.input_layernorm = EdgeMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 前馈前的归一化，与上面对称。
        self.post_attention_layernorm = EdgeMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # hidden_states 形状 (batch, seq_len, hidden_size)；cos/sin、缓存与 padding 掩码的语义同注意力层。
    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, past_key_value: tuple = None, use_cache: bool = False, attention_mask: torch.Tensor = None):
        # 记录残差主干：子层只学习对主干的修正量。
        residual = hidden_states
        # 先归一化再进注意力，归一化的结果不进入残差。
        hidden_states, past_kv = self.self_attn(self.input_layernorm(hidden_states), cos, sin, past_key_value, use_cache, attention_mask)
        # 把注意力的输出加回主干，形成第一条残差通路。
        hidden_states = hidden_states + residual

        # 同样记录第二条的残差主干。
        residual = hidden_states
        # 先归一化再进前馈层。
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        # 加回主干，完成一个 DecoderBlock。
        hidden_states = hidden_states + residual

        # 输出形状与输入一致，连同更新后的缓存一起交给下一层。
        return hidden_states, past_kv


# 完整模型本体：词嵌入 + RoPE 角度表 + 堆叠的 DecoderBlock + 最终归一化。
class EdgeMindModel(nn.Module):

    # 按配置搭建全部子模块；RoPE 角度表在此预计算并注册为缓冲区。
    def __init__(self, config: EdgeMindConfig):
        # 调用基类初始化，使全部子模块参数被统一管理。
        super().__init__()
        # 保存配置引用，便于后续读取词表大小等字段。
        self.config = config

        # 词嵌入层：把 token 编号映射为 hidden_size 维向量。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 嵌入后的 dropout；配置默认为 0，此时不生效。
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠全部 DecoderBlock，层数由配置决定。
        self.layers = nn.ModuleList([EdgeMindBlock(config) for _ in range(config.num_hidden_layers)])
        # 最终归一化：pre-norm 结构的必要收尾，残差主干在此第一次被归一化。
        self.norm = EdgeMindRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算覆盖整个上下文长度的 RoPE 角度表。
        freqs_cos, freqs_sin = precompute_freqs_cis(config.head_dim, config.max_position_embeddings, config.rope_theta)
        # 注册为缓冲区：随模型移动设备，但 persistent=False 表示不写入权重文件，加载时重算。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # input_ids 形状 (batch, seq_len)；past_key_values 为逐层缓存组成的列表。
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, past_key_values: list = None, use_cache: bool = False):
        # 首次前向没有缓存，为每层准备 None 占位。
        past_key_values = past_key_values or [None] * len(self.layers)
        # 已有缓存说明处于增量生成：新 token 的起始位置等于历史长度。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 查表得到词嵌入并施加 dropout。
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 按新 token 的位置切出本步所需的 RoPE 角度。
        seq_length = input_ids.shape[1]
        cos = self.freqs_cos[start_pos:start_pos + seq_length]
        sin = self.freqs_sin[start_pos:start_pos + seq_length]

        # 逐层前向，把隐状态和缓存逐层串联。
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(hidden_states, cos, sin, past_key_value, use_cache, attention_mask)
            presents.append(present)

        # 残差主干出栈前做最终归一化。
        hidden_states = self.norm(hidden_states)
        # 返回隐状态与逐层缓存；logits 的计算交给外面的 LM 头。
        return hidden_states, presents


# 带语言模型头的完整模型：在隐状态上接输出投影，用于预训练与生成。
class EdgeMindForCausalLM(PreTrainedModel):

    # 告诉 transformers 本模型使用的配置类，保存与加载时按此解析。
    config_class = EdgeMindConfig

    # 不传配置时使用默认配置，便于快速实例化。
    def __init__(self, config: EdgeMindConfig = None):
        # 先落定配置再交给基类，PreTrainedModel 会接管权重的保存与加载。
        self.config = config or EdgeMindConfig()
        super().__init__(self.config)

        # 模型本体：输出每个位置的隐状态。
        self.model = EdgeMindModel(self.config)
        # 输出投影：把隐状态映射到词表大小的 logits，无偏置。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享：输出投影与词嵌入共用同一份权重，省参数且通常效果更好。
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        # 按 transformers 默认方案初始化全部权重。
        self.post_init()

    # labels 传入时计算语言模型损失；不传时只返回 logits，供生成使用。
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, past_key_values: list = None, use_cache: bool = False, labels: torch.Tensor = None):
        # 先过模型本体，得到隐状态与逐层缓存。
        hidden_states, presents = self.model(input_ids, attention_mask, past_key_values, use_cache)
        # 隐状态经输出投影，得到每个位置对整个词表的打分。
        logits = self.lm_head(hidden_states)

        # 默认不算损失，仅在提供标签时计算。
        loss = None
        if labels is not None:
            # 语言模型的错位对应：位置 i 的 logits 负责预测位置 i+1 的 token。
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平后计算交叉熵；-100 是约定的忽略编号，padding 位置的标签应设为 -100。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # 返回 logits、损失与逐层缓存，分别服务评估、训练与生成。
        return logits, loss, presents
