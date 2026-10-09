# 在预训练权重基础上做对话微调；训练顺序与 train_pretrain.py 一致，只替换数据、起点和保存名。
# 对话格式与标签规则参考 MiniMind 的 SFT 流程，实现沿用本项目的训练骨架。

# 使用标准库读取语料路径、初始权重和基本训练设置。
import argparse

# 使用路径对象创建权重保存目录。
from pathlib import Path

# 复用 PyTorch 的计算、自动求导、优化器和权重保存能力。
import torch

# 使用标准读取器组批和打乱样本，不再手动管理样本编号。
from torch.utils.data import DataLoader

# 使用对话数据集，它已负责模板渲染和"只学助手"的标签。
from dataset.lm_dataset import SFTDataset

# 模型结构以预训练保存的配置为准，不在微调时改变结构。
from model.configuration_edgemind import EdgeMindConfig

# 使用项目自己的模型，其第二个返回值是训练损失。
from model.modeling_edgemind import EdgeMindForCausalLM

# 加载已有分词器，并自动读取同目录的对话排版规则。
from model.tokenization_edgemind import load_tokenizer


# 只完成加载预训练权重、按对话数据更新模型和按轮保存三个任务。
def train(args):
    # 轮数、每批样本数和学习率必须为正，避免无效训练。
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        # 在创建模型前指出基本设置错误。
        raise ValueError("epochs、batch_size 和 learning_rate 必须大于零。")

    # 这个入口只用于 AutoDL 的 CUDA GPU 训练。
    if not torch.cuda.is_available():
        # 没有 GPU 时停止，不创建 CPU 测试模型。
        raise RuntimeError("请在有 CUDA GPU 的 AutoDL 环境运行本脚本。")

    # 固定使用当前进程可见的第一张 GPU。
    device = torch.device("cuda:0")

    # 设置当前 GPU，使后面的 BF16 检查针对同一张卡。
    torch.cuda.set_device(device)

    # 与预训练一致使用 BF16，需要服务器环境支持这种计算精度。
    if not torch.cuda.is_bf16_supported():
        # 环境不支持时直接说明原因。
        raise RuntimeError("当前 GPU 环境不支持 BF16，请检查 GPU 和 PyTorch 镜像。")

    # 固定样本打乱使用的基础随机种子。
    torch.manual_seed(42)

    # 先在 CPU 上读取预训练保存的权重文件，不加载任意 Python 对象。
    checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=True)

    # 以预训练保存的结构配置为准，保证微调不改变模型形状。
    config = EdgeMindConfig(**checkpoint["config"])

    # 加载同一套分词器；对话排版规则由分词器目录中的模板文件提供。
    tokenizer = load_tokenizer(config)

    # 从指定的 JSONL 对话数据按需生成训练样本。
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 每次读取一批并打乱记录；批量和读取进程数由基本设置指定。
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 按保存的结构创建模型，并严格加载预训练权重，任何缺失或多出的参数都会报错。
    model = EdgeMindForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)

    # 权重已经复制到模型中，释放读取文件时的额外 CPU 权重副本。
    del checkpoint

    # 将载好预训练权重的模型移动到 GPU。
    model = model.to(device)

    # 微调沿用 AdamW 默认历史平均设置并关闭权重衰减；学习率通常比预训练小。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    # 将保存位置转换为绝对路径。
    save_directory = Path(args.save_dir).resolve()

    # 创建输出目录，已有目录可以继续使用。
    save_directory.mkdir(parents=True, exist_ok=True)

    # 切换训练模式；真正修改权重仍由后面的 optimizer.step 完成。
    model.train()

    # 输出实际 GPU、模型参数量和样本数，便于确认训练对象。
    print(f"GPU={torch.cuda.get_device_name(device)}, 参数量={sum(p.numel() for p in model.parameters()):,}, 样本数={len(dataset)}", flush=True)

    # 每轮完整读取一次对话数据，不处理保存位置或跳过已读样本。
    for epoch in range(args.epochs):
        # 记录本轮真正完成的参数更新，避免把没有学习过的权重当成微调结果保存。
        update_count = 0

        # 读取当前轮的每个批次，编号从零开始。
        for batch_index, batch in enumerate(loader):
            # 模型预测下一位置，所以从第二个标签开始检查；整批没有助手目标时无法计算有效误差。
            if not (batch["labels"][:, 1:] != -100).any().item():
                # 提示截断后没有可学习回答的批次，让使用者知道它没有参与更新。
                print(f"轮次={epoch + 1}/{args.epochs}, 批次={batch_index + 1}/{len(loader)}：没有有效助手目标，跳过。", flush=True)
                # 保留已有权重，直接读取下一批。
                continue

            # 清除上一批的梯度，让当前批次独立决定这次更新。
            optimizer.zero_grad(set_to_none=True)

            # 把输入、注意力掩码和标签都移动到 GPU。
            batch = {name: value.to(device) for name, value in batch.items()}

            # 在前向和损失计算中使用 BF16 混合精度，模型权重仍保留 float32。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 读取第二个返回值；非助手位置已在数据集中标为忽略。
                _, loss, _ = model(**batch)

            # 损失出现无效数值时停止，避免继续修改参数。
            if not torch.isfinite(loss).item():
                # 报告发生问题的轮次和批次。
                raise RuntimeError(f"第 {epoch + 1} 轮、第 {batch_index + 1} 批出现无效损失。")

            # 根据预测误差计算梯度，此时还没有修改模型权重。
            loss.backward()

            # 限制过大的整体梯度，梯度出现无效数值时拒绝更新。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

            # 每个有有效助手目标的批次更新一次模型参数，不累积多批梯度。
            optimizer.step()

            # 累计本轮真正完成的更新次数。
            update_count += 1

            # 每十批和每轮最后一批打印当前批次损失。
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                # 输出实际轮次和批次，避免将当前损失误称为整轮平均。
                print(f"轮次={epoch + 1}/{args.epochs}, 批次={batch_index + 1}/{len(loader)}, loss={loss.item():.4f}", flush=True)

        # 整轮都没有助手目标时应说明数据或长度设置有问题，而不是保存未经微调的权重。
        if update_count == 0:
            # 提醒检查对话样本以及截断长度。
            raise ValueError("本轮没有完成任何更新，请检查对话数据是否包含助手回答，以及 max_length 是否截掉了全部回答。")

        # 每轮使用独立文件保存微调结果，文件名与预训练权重区分。
        weight_path = save_directory / f"sft_epoch_{epoch + 1}.pth"

        # 保存模型权重和结构配置，供后续正常创建模型再加载；不包含优化器或续训进度。
        torch.save({"model": model.state_dict(), "config": config.to_dict()}, weight_path)

        # 给出本轮保存的实际路径。
        print(f"已保存：{weight_path}", flush=True)


# 从项目根目录使用 python -m trainer.train_sft 启动，导入时不自动训练。
if __name__ == "__main__":
    # 使用标准解析器，不另建配置框架或训练管理器。
    parser = argparse.ArgumentParser(description="EdgeMind 单卡 GPU 对话微调")

    # 必须明确指定对话 JSONL 数据，避免默认使用小样本文件。
    parser.add_argument("--data_path", required=True, help="包含 conversations 字段的 SFT JSONL 文件")

    # 必须明确指定预训练权重，微调不从随机初始化开始。
    parser.add_argument("--init_from", required=True, help="train_pretrain.py 保存的 .pth 文件")

    # 默认将权重保存在 AutoDL 数据盘。
    parser.add_argument("--save_dir", default="/root/autodl-tmp/edgemind/checkpoints", help="模型权重保存目录")

    # 对话数据默认微调两轮，可按实际效果增减。
    parser.add_argument("--epochs", type=int, default=2, help="完整训练轮数")

    # 每批一条是当前模型的保守显存起点，可在 GPU 实测后调整。
    parser.add_argument("--batch_size", type=int, default=1, help="每批样本数")

    # 默认每条样本总长一千零二十四，对话通常比预训练段落更长。
    parser.add_argument("--max_length", type=int, default=1024, help="样本总长度")

    # 微调学习率小于预训练，避免大步子破坏已有语言能力。
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="固定学习率")

    # 两个读取进程准备数据，实际模型计算仍在单张 GPU 上执行。
    parser.add_argument("--num_workers", type=int, default=2, help="数据读取进程数")

    # 将基本设置交给唯一的训练函数。
    train(parser.parse_args())
