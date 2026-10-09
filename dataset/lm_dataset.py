# 参考 MiniMind 的预训练样本组织：正文前后添加起止标记，填充位置不参与训练损失。
# 来源：https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/dataset/lm_dataset.py

# 使用标准库读取每行独立保存的一条 JSON 记录。
import json

# 对话样本需要以小概率补充身份介绍、按概率移除空思考标签。
import random

# 用紧凑整数数组保存样本在文件中的位置，不把全部正文放入内存。
from array import array

# 使用路径对象保存数据文件的位置，后续按需打开读取。
from pathlib import Path

# 使用 PyTorch 构造模型需要的整数张量。
import torch

# 继承标准 Dataset，只实现样本数量和按编号读取，供后续 DataLoader 使用。
from torch.utils.data import Dataset

# 为现成分词器声明类型，分词算法本身由已经接入的分词器完成。
from transformers import PreTrainedTokenizerFast


# 按二进制逐行扫描 JSONL 文件，返回每条非空记录的字节位置；预训练和对话数据集共用同一份扫描逻辑。
def _scan_line_offsets(data_path: Path) -> array:
    # 每条记录只保存一个文件位置；Q 表示无符号长整数，能够表示大文件位置。
    offsets = array("Q")
    # 位置按字节计数，不受中文字符占多个字节的影响。
    with data_path.open("rb") as data_file:
        # 第一行从文件的第零个字节开始。
        offset = 0
        # 扫描时只记录位置，不解析或保存全部训练正文。
        for line in data_file:
            # 跳过纯空行，防止把没有 JSON 内容的行当成训练样本。
            if line.strip():
                # 保存当前记录的起点，后续可直接跳到此处读取。
                offsets.append(offset)
            # 下一行的位置增加当前行的字节数，包括行尾换行符。
            offset += len(line)

    # 返回紧凑的位置数组，空文件由调用方检查并给出针对性提示。
    return offsets


# 一条 JSONL 记录生成一条定长预训练样本，包含模型输入、注意力掩码和训练目标。
class PretrainDataset(Dataset):
    # 默认使用较短的 512 token 样本先跑通流程；长度包括起始和结束两个标记。
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 512):
        # 初始化标准数据集基类，不创建训练器或加载模型。
        super().__init__()

        # 至少给正文留一个位置，并且不能超过已与模型对齐的分词器长度上限。
        if type(max_length) is not int or not 3 <= max_length <= tokenizer.model_max_length:
            # 提前指出无效长度，避免后续无法保留结束标记或超出模型的位置表。
            raise ValueError(f"max_length 必须是 3 到 {tokenizer.model_max_length} 之间的整数。")

        # 固定数据文件的绝对路径，避免工作目录变化后读取到其他文件。
        self.data_path = Path(data_path).resolve()
        # 使用已经核对过词表大小和特殊编号的分词器。
        self.tokenizer = tokenizer
        # 保存每条样本最终需要达到的总长度。
        self.max_length = max_length

        # 与对话数据集共用逐行位置扫描，避免两份扫描逻辑不一致。
        self.offsets = _scan_line_offsets(self.data_path)

        # 空文件或只有空行的文件不能用于训练，直接给出明确提示。
        if not self.offsets:
            # 防止后续训练循环收到一个没有样本的数据集。
            raise ValueError("数据文件中没有可读取的 JSONL 样本。")

    # 返回记录数量，让后续 DataLoader 知道可以读取多少条样本。
    def __len__(self) -> int:
        # 文件位置数组与有效 JSONL 行一一对应。
        return len(self.offsets)

    # 按样本编号读取一条正文，生成可以直接传给现有模型的三个张量。
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 每次只读取当前样本，不保存跨进程共享的打开文件句柄。
        with self.data_path.open("rb") as data_file:
            # 跳到初始化时记录的字节位置，避免每次从文件开头重新扫描。
            data_file.seek(self.offsets[index])
            # 按 UTF-8 解码当前行，再读取其中的 JSON 字段。
            sample = json.loads(data_file.readline().decode("utf-8"))

        # 预训练语料必须是带有非空字符串 text 字段的 JSON 对象。
        if not isinstance(sample, dict) or not isinstance(sample.get("text"), str) or not sample["text"].strip():
            # 不把缺失、数字或空白正文偷偷转换成训练文本，方便及时发现数据问题。
            raise ValueError(f"第 {index + 1} 条样本必须包含非空字符串 text 字段。")

        # 正文最多占总长度减去两个位置，提前为起始和结束标记留出空间。
        text_ids = self.tokenizer(sample["text"], add_special_tokens=False, truncation=True, max_length=self.max_length - 2)["input_ids"]
        # 显式添加起始和结束编号，保证长正文截断后仍然保留结束标记。
        tokens = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]
        # 只在正文和起止标记之后补齐，不改变真实内容的位置。
        padding_length = self.max_length - len(tokens)

        # 输入使用词表中的填充编号，所有位置都保持合法的模型输入编号。
        input_ids = torch.tensor(tokens + [self.tokenizer.pad_token_id] * padding_length, dtype=torch.long)
        # 有效内容标为 1，真正添加的填充位置标为 0，使用现有模型支持的整数掩码。
        attention_mask = torch.tensor([1] * len(tokens) + [0] * padding_length, dtype=torch.long)

        # 先完整复制输入作为目标，预测下一位置的错位由模型内部执行，数据集不重复错位。
        labels = input_ids.clone()
        # 只忽略实际补齐位置；原文若含相同编号，仍视为真实内容，而不是误判为填充。
        labels[attention_mask == 0] = -100

        # 三个字段与 EdgeMindForCausalLM.forward 的参数同名，后续可直接作为模型输入。
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# 身份介绍池：没有 system 的对话以小概率补一句，让模型适应有无开场说明两种情况，并记住自己是谁。
EDGE_SYSTEM_PROMPTS = (
    # 中文介绍为主，贴合以中文为主的对话数据。
    "你是 EdgeMind，一个小巧但有用的语言模型。",
    "你是 EdgeMind，请尽力帮助用户解决问题。",
    "你是一个可靠的AI，请给出准确的回答。",
    # 保留一条英文介绍，兼容英文对话样本。
    "You are EdgeMind, a small but useful language model.",
)


# 一条 JSONL 记录生成一条定长对话样本；只在助手发言位置计算损失，其余位置全部忽略。
class SFTDataset(Dataset):
    # 对话通常比预训练段落更长，默认一千零二十四；长度上限仍受模型位置表约束。
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 1024):
        # 初始化标准数据集基类，不创建训练器或加载模型。
        super().__init__()

        # 与预训练相同的长度检查，避免截断后失去结束标记或超出模型位置表。
        if type(max_length) is not int or not 3 <= max_length <= tokenizer.model_max_length:
            # 提前指出无效长度。
            raise ValueError(f"max_length 必须是 3 到 {tokenizer.model_max_length} 之间的整数。")

        # 固定数据文件的绝对路径，避免工作目录变化后读取到其他文件。
        self.data_path = Path(data_path).resolve()
        # 使用已经核对过词表大小和特殊编号的分词器。
        self.tokenizer = tokenizer
        # 保存每条样本最终需要达到的总长度。
        self.max_length = max_length

        # 与预训练数据集共用逐行位置扫描。
        self.offsets = _scan_line_offsets(self.data_path)
        # 空文件或只有空行的文件不能用于训练，直接给出明确提示。
        if not self.offsets:
            # 防止后续训练循环收到一个没有样本的数据集。
            raise ValueError("数据文件中没有可读取的 JSONL 样本。")

        # 助手发言的开头（角色行）和结尾（结束标记加换行）对应的编号序列，用于定位需要学习的区间。
        self.assistant_start = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False)["input_ids"]
        self.assistant_end = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False)["input_ids"]

    # 返回记录数量，让后续 DataLoader 知道可以读取多少条样本。
    def __len__(self) -> int:
        # 文件位置数组与有效 JSONL 行一一对应。
        return len(self.offsets)

    # 把一条记录的对话数组渲染成模板文本；工具定义与工具调用从字符串解析回 JSON 对象。
    def render_conversations(self, conversations: list[dict]) -> str:
        # 逐条复制消息，解析后的消息按原顺序收集。
        messages = []
        # 工具定义单独保存，最后作为模板的 tools 参数传入。
        tools = None
        for message in conversations:
            # 复制后修改，不改动文件里读出的原始记录。
            message = dict(message)
            # system 消息里的工具定义是 JSON 字符串，解析成对象列表交给模板。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # assistant 消息里的工具调用同样是字符串，模板需要对象形式。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            # 收集处理后的消息，保持原有顺序。
            messages.append(message)

        # 没有工具且没有开场的对话，以两成概率补一句身份介绍。
        if tools is None and messages[0].get("role") != "system" and random.random() < 0.2:
            # 插到最前面，成为模板的 system 段。
            messages.insert(0, {"role": "system", "content": random.choice(EDGE_SYSTEM_PROMPTS)})

        # 调用功能一接入的排版规则，tokenize=False 表示这一步只要文本。
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

        # 模板会给助手补一对空思考标签；以八成概率删掉，避免模型养成开口先输出空标签的习惯。
        if "<think>\n\n</think>\n\n" in rendered and random.random() > 0.2:
            # 全文替换，一条对话里的多处空标签一并处理。
            rendered = rendered.replace("<think>\n\n</think>\n\n", "")

        # 返回最终用于训练的文本。
        return rendered

    # 只保留助手发言区间（含结束标记）的标签，其余位置全部标为不参与损失。
    def generate_labels(self, tokens: list[int]) -> list[int]:
        # 先假设所有位置都不学习，再逐个助手区间打开。
        labels = [-100] * len(tokens)
        # 从左到右扫描编号序列，寻找助手发言的开头。
        index = 0
        while index < len(tokens):
            # 当前位置与助手开头序列完全重合时，认为进入一个助手区间。
            if tokens[index:index + len(self.assistant_start)] == self.assistant_start:
                # 学习内容从助手开头之后开始。
                start = index + len(self.assistant_start)
                # 继续向右找到该区间的结束标记。
                end = start
                while end < len(tokens):
                    # 结束标记出现的位置就是助手内容的结尾。
                    if tokens[end:end + len(self.assistant_end)] == self.assistant_end:
                        break
                    # 没找到就逐位右移，序列耗尽时自然退出。
                    end += 1
                # 学习范围覆盖助手正文到结束标记（含），让模型学会自己收尾。
                for position in range(start, min(end + len(self.assistant_end), len(tokens))):
                    # 标签就是当前位置的真实编号，错位由模型内部完成。
                    labels[position] = tokens[position]
                # 跳过已处理的区间，继续寻找下一轮助手发言。
                index = end + len(self.assistant_end) if end < len(tokens) else len(tokens)
            else:
                # 不是助手开头就检查下一个位置。
                index += 1

        # 返回与编号等长的标签序列。
        return labels

    # 按样本编号读取一条对话记录，生成可以直接传给现有模型的三个张量。
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # 每次只读取当前样本，不保存跨进程共享的打开文件句柄。
        with self.data_path.open("rb") as data_file:
            # 跳到初始化时记录的字节位置。
            data_file.seek(self.offsets[index])
            # 按 UTF-8 解码当前行，再读取其中的 JSON 字段。
            sample = json.loads(data_file.readline().decode("utf-8"))

        # 对话样本必须是带有非空 conversations 数组的 JSON 对象。
        conversations = sample.get("conversations") if isinstance(sample, dict) else None
        if not isinstance(conversations, list) or not conversations:
            # 不悄悄跳过坏数据，方便及时发现数据问题。
            raise ValueError(f"第 {index + 1} 条样本必须包含非空的 conversations 数组。")

        # 逐条检查消息结构，角色和内容必须是字符串；调用工具时内容可以是空串。
        for position, message in enumerate(conversations, start=1):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                # 报告样本编号和消息位置，便于回到数据中修正。
                raise ValueError(f"第 {index + 1} 条样本的第 {position} 条消息必须包含字符串 role 和 content。")

        # 渲染成模板文本并整体编号，超出长度的部分直接截断。
        tokens = self.tokenizer(self.render_conversations(conversations), add_special_tokens=False)["input_ids"][: self.max_length]
        # 只在真实内容之后补齐，不改变真实内容的位置。
        padding_length = self.max_length - len(tokens)

        # 输入使用词表中的填充编号，所有位置都保持合法的模型输入编号。
        input_ids = torch.tensor(tokens + [self.tokenizer.pad_token_id] * padding_length, dtype=torch.long)
        # 有效内容标为 1，真正添加的填充位置标为 0。
        attention_mask = torch.tensor([1] * len(tokens) + [0] * padding_length, dtype=torch.long)

        # 先在真实内容上生成“只学助手”的标签，再用 -100 补齐到定长。
        labels = torch.tensor(self.generate_labels(tokens) + [-100] * padding_length, dtype=torch.long)

        # 三个字段与预训练样本完全一致，训练循环无需区分样本来源。
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
