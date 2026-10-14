# 只测试 SFT 模型的单轮对话，读取 train_sft.py 保存的配置和权重。
# 与 eval_pretrain.py 的区别：输入套用对话模板，让模型以助手身份回答，而不是裸文本续写。

# 使用标准库读取权重路径、对话内容、工具说明和采样设置。
import argparse

# 使用标准库读取工具说明的 JSON 文件。
import json

# 使用标准库检查温度是否为有限数值，避免无效设置进入概率计算。
import math

# 复用 PyTorch 的模型加载、GPU 计算和张量操作。
import torch

# 按保存时的结构配置重新创建模型，避免测试时误用不同的层数或宽度。
from model.configuration_edgemind import EdgeMindConfig

# 使用项目自己实现的语言模型，其第一个返回值是各词元的预测分数。
from model.modeling_edgemind import EdgeMindForCausalLM

# 复用训练时使用的本地分词器，确保文字与编号的对应关系不变。
from model.tokenization_edgemind import load_tokenizer


# 加载一份 SFT 结果，用对话模板组织提问并逐步生成助手的回答。
def chat(args):
    # 空白提问无法测试对话，回答长度也必须为正。
    if not args.prompt.strip() or args.max_new_tokens <= 0:
        # 在创建模型前说明无效的输入设置。
        raise ValueError("prompt 不能为空，max_new_tokens 必须大于零。")

    # 温度控制候选概率的集中程度；零表示沿用每次选择最高分的方式。
    if not math.isfinite(args.temperature) or args.temperature < 0:
        # 拒绝负数、无穷大和非数值温度，防止生成时出现无效概率。
        raise ValueError("temperature 必须是大于等于零的有限数值。")

    # top_p 表示保留候选的累计概率范围；一表示不按累计概率筛选。
    if not 0 < args.top_p <= 1:
        # 只接受能够保留至少一个候选的有效概率阈值。
        raise ValueError("top_p 必须大于零，并且小于等于一。")

    # 与训练入口一致，本脚本在 AutoDL 的 CUDA GPU 上运行。
    if not torch.cuda.is_available():
        # 没有 GPU 时停止，避免意外在本地 CPU 上加载正式模型。
        raise RuntimeError("请在有 CUDA GPU 的 AutoDL 环境运行本脚本。")

    # 固定使用当前进程可见的第一张 GPU。
    device = torch.device("cuda:0")

    # 设置当前设备，让下面的精度检查针对这张 GPU。
    torch.cuda.set_device(device)

    # 测试采用与训练一致的 BF16 计算精度。
    if not torch.cuda.is_bf16_supported():
        # GPU 或镜像不支持时直接说明原因。
        raise RuntimeError("当前 GPU 环境不支持 BF16，请检查 GPU 和 PyTorch 镜像。")

    # 先在 CPU 上读取本项目保存的权重文件，不加载任意 Python 对象。
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)

    # 从权重文件恢复实际训练时的结构设置。
    config = EdgeMindConfig(**checkpoint["config"])

    # 加载同一套分词器；对话模板由 model 目录中的 chat_template.jinja 自动提供。
    tokenizer = load_tokenizer(config)

    # 分词器没有对话模板时无法组织提问，先检查 transformers 版本和模板文件。
    if tokenizer.chat_template is None:
        # 明确说明缺少模板，避免退回裸文本续写造成误判。
        raise RuntimeError("分词器没有对话模板，请确认 transformers 版本不低于 4.43，且 model/chat_template.jinja 存在。")

    # 只放一条用户提问；system 内容可选，与训练数据中多数无 system 的样本保持一致。
    messages = []
    if args.system:
        # 用户明确要求时，把 system 放在对话开头。
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    # 工具说明从 JSON 文件读取，内容应与训练样本中 tools 字段解析后的结构一致。
    tools = None
    if args.tools:
        # 以 UTF-8 读取工具说明文件，避免中文工具描述乱码。
        with open(args.tools, encoding="utf-8") as tools_file:
            tools = json.load(tools_file)

    # 套用对话模板并追加“助手开始回答”的开头，与训练时看到的教材格式一致。
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)

    # 模板已经包含全部特殊标记，直接编码即可，不能再额外添加起始标记。
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 提问与计划回答的总长度不能超过模型的位置表范围。
    if len(prompt_ids) + args.max_new_tokens > config.max_position_embeddings:
        # 提醒用户缩短输入或回答长度，不悄悄截断提问。
        raise ValueError(f"输入与回答总长度不能超过 {config.max_position_embeddings} 个词元。")

    # 正常创建模型，使包括位置角度表在内的模型内部数据完成初始化。
    model = EdgeMindForCausalLM(config)

    # 严格加载训练权重，任何缺失或多出的参数都会报错。
    model.load_state_dict(checkpoint["model"], strict=True)

    # 权重已经复制到模型中，释放读取文件时的额外 CPU 权重副本。
    del checkpoint

    # 将完整模型移动到 GPU。
    model = model.to(device)

    # 关闭训练模式中的随机丢弃，让测试使用模型的推理行为。
    model.eval()

    # 一次只测试一轮对话，不补齐长度，因此不需要屏蔽填充位置。
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 单独记录回答的编号，最后一次性还原中文，避免拆开字节片段造成乱码。
    generated_ids = []

    # 测试只做预测，不计算梯度，也不更新模型权重。
    with torch.no_grad():
        # 最多生成指定数量的词元；一个词元不一定等于一个汉字。
        for _ in range(args.max_new_tokens):
            # 与训练相同，计算时使用 BF16，模型权重保持 float32。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 输入当前全部文本，取出模型给出的每个位置的词表分数。
                logits, _, _ = model(input_ids)

            # 取出预测下一个词元的分数，并用 float32 计算概率，减少低精度舍入误差。
            scores = logits[0, -1].float()

            # 温度为零时保留每次选择最高分的方式，方便复现和对照。
            if args.temperature == 0:
                # 直接选择分数最高的词元，不进行随机抽样。
                next_id = scores.argmax().item()
            # 正温度时按温度与 top_p 采样，让合理候选都有被选中的机会。
            else:
                # 较低温度放大候选之间的差距，让高分词元更容易被选中。
                scores = scores / args.temperature

                # top_p 小于一时，只在累计概率达到阈值的高分候选中抽样。
                if args.top_p < 1:
                    # 将分数从高到低排序，同时保留各分数原本对应的词元编号。
                    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                    # 将排序后的分数转换为概率，用于计算候选集合的累计概率。
                    sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
                    # 标记累计概率超过阈值的位置，准备移除后面的低概率候选。
                    remove_mask = sorted_probabilities.cumsum(dim=-1) > args.top_p
                    # 将移除标记右移一位，让使累计概率首次超过阈值的候选也参与抽样。
                    remove_mask[1:] = remove_mask[:-1].clone()
                    # 始终保留概率最高的候选，避免阈值很小时所有候选都被移除。
                    remove_mask[0] = False
                    # 将需要移除的原始词元分数设为负无穷，使它们的采样概率变为零。
                    scores[sorted_indices[remove_mask]] = float("-inf")

                # 将剩余分数重新转换为总和为一的概率分布。
                probabilities = torch.softmax(scores, dim=-1)
                # 按概率抽取一个词元，而不是始终使用概率最高的词元。
                next_id = torch.multinomial(probabilities, num_samples=1).item()

            # 助手说完话会产出结束标记 <|im_end|>，看到它就该停下。
            if next_id == config.eos_token_id:
                # 提前退出，不把结束标记显示成普通文字。
                break

            # 记录本次选出的词元，供最终解码使用。
            generated_ids.append(next_id)

            # 把新词元接到输入末尾，让下一次预测能够看到自己刚生成的内容。
            input_ids = torch.cat((input_ids, input_ids.new_tensor([[next_id]])), dim=1)

    # 显示模型实际看到的模板文本，便于核对“考卷格式”是否正确。
    print(f"模板文本：\n{prompt_text}")

    # 保留特殊标记的原文，便于检查 <tool_call> 等标记是否被正确产出。
    reply = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # 返回助手的完整回答，包含可能出现的工具调用标记。
    return reply


# 直接运行文件时读取命令；被其他代码导入时不自动加载模型。
if __name__ == "__main__":
    # 只保留对话测试需要的基本设置。
    parser = argparse.ArgumentParser(description="EdgeMind SFT 单轮对话测试")

    # 必须明确选择某一轮 SFT 保存的权重文件。
    parser.add_argument("--weights", required=True, help="train_sft.py 保存的 .pth 文件")

    # 输入一句用户提问，脚本自动套用对话模板。
    parser.add_argument("--prompt", default="你好，介绍一下你自己", help="用户想说的话")

    # 可选的对话开头设定，默认与训练数据中多数样本一样不加 system。
    parser.add_argument("--system", default=None, help="可选的 system 内容")

    # 可选的工具说明 JSON 文件，用于测试模型是否会发起工具调用。
    parser.add_argument("--tools", default=None, help="工具说明 JSON 文件路径")

    # 限制新增词元数量，避免回答无限进行。
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最多回答的词元数")

    # 与续写测试一致；设为零即可复现每次选择最高分的方式。
    parser.add_argument("--temperature", type=float, default=0.85, help="采样温度；零表示每次选择最高分词元")

    # 与续写测试一致，只保留累计概率达到该值的高分候选参与抽样。
    parser.add_argument("--top_p", type=float, default=0.95, help="候选累计概率阈值，范围为 (0, 1]")

    # 解析命令中给出的设置。
    args = parser.parse_args()

    # 调用唯一的对话函数，获得助手的回答。
    reply = chat(args)

    # 显示用户提问，方便区分输入与回答。
    print(f"用户：{args.prompt}")

    # 显示回答；立即结束或只生成结束标记时也明确显示空结果。
    print(f"助手：{reply if reply else '（没有生成可显示的文字）'}")
