# 参考 MiniMind 的预训练样本组织：正文前后添加起止标记，填充位置不参与训练损失。
# 来源：https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/dataset/lm_dataset.py

# 使用标准库读取每行独立保存的一条 JSON 记录。
import json

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

        # 每条记录只保存一个文件位置；Q 表示无符号长整数，能够表示大文件位置。
        self.offsets = array("Q")
        # 按二进制行扫描，位置按字节计数，不受中文字符占多个字节的影响。
        with self.data_path.open("rb") as data_file:
            # 第一行从文件的第零个字节开始。
            offset = 0
            # 扫描时只记录位置，不解析或保存全部训练正文。
            for line in data_file:
                # 跳过纯空行，防止把没有 JSON 内容的行当成训练样本。
                if line.strip():
                    # 保存当前记录的起点，后续可直接跳到此处读取。
                    self.offsets.append(offset)
                # 下一行的位置增加当前行的字节数，包括行尾换行符。
                offset += len(line)

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
