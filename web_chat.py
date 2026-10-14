# 在 AutoDL 实例上启动一个只监听本机的网页对话界面，配合 SSH 隧道在本地浏览器访问。
# 复用 eval_chat.py 的加载与生成思路；区别是模型只加载一次，并保留多轮对话历史。

# 使用标准库读取权重路径、工具说明和界面启动设置。
import argparse

# 使用标准库读取工具说明的 JSON 文件。
import json

# 使用标准库检查温度是否为有限数值，避免无效设置进入概率计算。
import math

# 复用 PyTorch 的模型加载、GPU 计算和张量操作。
import torch

# 使用 Gradio 快速搭建聊天网页，只在实例本机监听，不暴露公网。
import gradio as gr

# 按保存时的结构配置重新创建模型，避免测试时误用不同的层数或宽度。
from model.configuration_edgemind import EdgeMindConfig

# 使用项目自己实现的语言模型，其第一个返回值是各词元的预测分数。
from model.modeling_edgemind import EdgeMindForCausalLM

# 复用训练时使用的本地分词器，确保文字与编号的对应关系不变。
from model.tokenization_edgemind import load_tokenizer


# 启动时加载一次模型，返回之后每次提问都要用到的全部对象。
def load_chatbot(args):
    # 与训练入口一致，本脚本在 AutoDL 的 CUDA GPU 上运行。
    if not torch.cuda.is_available():
        # 没有 GPU 时停止，避免意外在本地 CPU 上加载正式模型。
        raise RuntimeError("请在有 CUDA GPU 的 AutoDL 环境运行本脚本。")

    # 固定使用当前进程可见的第一张 GPU。
    device = torch.device("cuda:0")

    # 设置当前设备，让下面的精度检查针对这张 GPU。
    torch.cuda.set_device(device)

    # 网页服务采用与训练一致的 BF16 计算精度。
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

    # 正常创建模型，使包括位置角度表在内的模型内部数据完成初始化。
    model = EdgeMindForCausalLM(config)

    # 严格加载训练权重，任何缺失或多出的参数都会报错。
    model.load_state_dict(checkpoint["model"], strict=True)

    # 权重已经复制到模型中，释放读取文件时的额外 CPU 权重副本。
    del checkpoint

    # 将完整模型移动到 GPU。
    model = model.to(device)

    # 关闭训练模式中的随机丢弃，让网页使用模型的推理行为。
    model.eval()

    # 工具说明从 JSON 文件读取，之后每轮对话都随模板提供给模型。
    tools = None
    if args.tools:
        # 以 UTF-8 读取工具说明文件，避免中文工具描述乱码。
        with open(args.tools, encoding="utf-8") as tools_file:
            tools = json.load(tools_file)

    # 返回之后每次提问都要用到的对象集合。
    return model, tokenizer, config, device, tools


# 从当前输入分数中选出一个词元；温度为零时每次选最高分，正温度时按 top_p 采样。
def pick_next_token(scores, temperature, top_p):
    # 温度为零时保留每次选择最高分的方式，方便复现和对照。
    if temperature == 0:
        # 直接选择分数最高的词元，不进行随机抽样。
        return scores.argmax().item()

    # 较低温度放大候选之间的差距，让高分词元更容易被选中。
    scores = scores / temperature

    # top_p 小于一时，只在累计概率达到阈值的高分候选中抽样。
    if top_p < 1:
        # 将分数从高到低排序，同时保留各分数原本对应的词元编号。
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        # 将排序后的分数转换为概率，用于计算候选集合的累计概率。
        sorted_probabilities = torch.softmax(sorted_scores, dim=-1)
        # 标记累计概率超过阈值的位置，准备移除后面的低概率候选。
        remove_mask = sorted_probabilities.cumsum(dim=-1) > top_p
        # 将移除标记右移一位，让使累计概率首次超过阈值的候选也参与抽样。
        remove_mask[1:] = remove_mask[:-1].clone()
        # 始终保留概率最高的候选，避免阈值很小时所有候选都被移除。
        remove_mask[0] = False
        # 将需要移除的原始词元分数设为负无穷，使它们的采样概率变为零。
        scores[sorted_indices[remove_mask]] = float("-inf")

    # 将剩余分数重新转换为总和为一的概率分布。
    probabilities = torch.softmax(scores, dim=-1)
    # 按概率抽取一个词元，而不是始终使用概率最高的词元。
    return torch.multinomial(probabilities, num_samples=1).item()


# 用模型逐个生成回答的词元，遇到结束标记或达到长度上限时停止。
def generate_reply(model, config, device, prompt_ids, temperature, top_p, max_new_tokens):
    # 一次只生成一段回答，不补齐长度，因此不需要屏蔽填充位置。
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 单独记录回答的编号，最后一次性还原中文，避免拆开字节片段造成乱码。
    generated_ids = []

    # 生成只做预测，不计算梯度，也不更新模型权重。
    with torch.no_grad():
        # 最多生成指定数量的词元；一个词元不一定等于一个汉字。
        for _ in range(max_new_tokens):
            # 与训练相同，计算时使用 BF16，模型权重保持 float32。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 输入当前全部文本，取出模型给出的每个位置的词表分数。
                logits, _, _ = model(input_ids)

            # 取出预测下一个词元的分数，并用 float32 计算概率，减少低精度舍入误差。
            scores = logits[0, -1].float()

            # 按当前温度与 top_p 设置选出下一个词元。
            next_id = pick_next_token(scores, temperature, top_p)

            # 助手说完话会产出结束标记 <|im_end|>，看到它就该停下。
            if next_id == config.eos_token_id:
                # 提前退出，不把结束标记显示成普通文字。
                break

            # 记录本次选出的词元，供最终解码使用。
            generated_ids.append(next_id)

            # 把新词元接到输入末尾，让下一次预测能够看到自己刚生成的内容。
            input_ids = torch.cat((input_ids, input_ids.new_tensor([[next_id]])), dim=1)

    # 保留特殊标记的原文，便于在网页上检查 <tool_call> 等标记是否被正确产出。
    return generated_ids


# 网页每次提问都会调用这个函数；history 是界面传入的已有对话记录。
def respond(message, history, temperature, top_p, chatbot):
    # 拆出启动时加载好的对象，避免每次提问重复加载模型。
    model, tokenizer, config, device, tools = chatbot

    # 空白提问不进入模板，直接请用户重新输入。
    if not message.strip():
        # 返回提示文字，不消耗任何计算。
        return "请输入内容后再发送。"

    # 温度与 top_p 的合法范围与命令行测试保持一致。
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1:
        # 界面滑块一般给出合法值，这里只兜底异常输入。
        return "温度或 top_p 设置无效，请调整滑块。"

    # 把界面历史整理成模板需要的消息列表；system 与工具说明由启动设置决定。
    messages = [{"role": item["role"], "content": item["content"]} for item in history]

    # 把本次提问追加到历史末尾，形成完整的多轮对话。
    messages.append({"role": "user", "content": message})

    # 套用对话模板并追加“助手开始回答”的开头，与训练时看到的教材格式一致。
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools)

    # 模板已经包含全部特殊标记，直接编码即可，不能再额外添加起始标记。
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # 多轮历史会不断变长，总长度超过位置表时本轮无法可靠回答。
    if len(prompt_ids) + 256 > config.max_position_embeddings:
        # 提醒用户清空对话重新开始，不悄悄截断历史造成自相矛盾。
        return "对话太长了，请点击清空按钮重新开始。"

    # 逐词元生成回答，保留 <tool_call> 等特殊标记的原文。
    generated_ids = generate_reply(model, config, device, prompt_ids, temperature, top_p, 256)

    # 一次性还原整段回答；立即结束或只生成结束标记时明确说明。
    if not generated_ids:
        # 模型直接结束时不显示空气，而是说明情况。
        return "（没有生成可显示的文字）"

    # 返回包含特殊标记原文的回答，交给网页显示。
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


# 直接运行文件时加载模型并启动网页；被其他代码导入时不自动启动。
if __name__ == "__main__":
    # 只保留网页服务需要的设置。
    parser = argparse.ArgumentParser(description="EdgeMind SFT 网页对话界面")

    # 必须明确选择某一轮 SFT 保存的权重文件。
    parser.add_argument("--weights", required=True, help="train_sft.py 保存的 .pth 文件")

    # 可选的工具说明 JSON 文件，用于在网页上演示工具调用。
    parser.add_argument("--tools", default=None, help="工具说明 JSON 文件路径")

    # 网页只在本机监听这个端口，配合 SSH 隧道访问。
    parser.add_argument("--port", type=int, default=6006, help="本机监听端口，默认 6006")

    # 解析命令中给出的设置。
    args = parser.parse_args()

    # 启动时加载一次模型，之后每次提问复用。
    chatbot = load_chatbot(args)

    # 用偏函数把加载好的对象固定到回答函数上，界面的滑块只控制温度与 top_p。
    from functools import partial

    # 固定模型对象，前三个参数由界面提供：本次提问、历史、温度、top_p。
    respond_with_chatbot = partial(respond, chatbot=chatbot)

    # 创建聊天界面：输入框、历史记录、清空按钮由 Gradio 自动提供。
    demo = gr.ChatInterface(
        # 每次发送消息时调用的回答函数。
        fn=respond_with_chatbot,
        # Gradio 6 的历史记录只使用角色消息格式，与对话模板的消息结构一致。
        # 温度与 top_p 以滑块形式出现在输入框下方。
        additional_inputs=[
            # 温度滑块：零表示每次选最高分，默认与命令行测试一致。
            gr.Slider(minimum=0, maximum=1.5, value=0.85, step=0.05, label="temperature"),
            # top_p 滑块：一表示不筛选，默认与命令行测试一致。
            gr.Slider(minimum=0.05, maximum=1.0, value=0.95, step=0.05, label="top_p"),
        ],
        # 网页标题，便于确认连上的是 EdgeMind。
        title="EdgeMind 对话测试",
    )

    # 只监听本机地址，实例外部无法直接访问；通过 SSH 隧道在本地浏览器打开。
    demo.launch(server_name="127.0.0.1", server_port=args.port)
