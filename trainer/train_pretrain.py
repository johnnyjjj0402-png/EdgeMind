# 参考 MiniMind 的基本训练顺序，先完成单卡 GPU 上最直接的预训练流程。
# 来源：https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/train_pretrain.py

# 使用标准库读取正式语料路径和基本训练设置。
import argparse

# 使用路径对象创建权重保存目录。
from pathlib import Path

# 复用 PyTorch 的计算、自动求导、优化器和权重保存能力。
import torch

# 使用标准读取器组批和打乱样本，不再手动管理样本编号。
from torch.utils.data import DataLoader

# 复用现有数据集，它已负责起止标记、注意力掩码和填充标签。
from dataset.lm_dataset import PretrainDataset

# 模型结构统一从正式配置类读取。
from model.configuration_edgemind import EdgeMindConfig

# 使用项目自己的模型，其第二个返回值是训练损失。
from model.modeling_edgemind import EdgeMindForCausalLM

# 加载已有分词器，不在训练脚本中训练或修改词表。
from model.tokenization_edgemind import load_tokenizer


# 只完成读取数据、更新模型和按轮保存权重三个任务。
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

    # 首版固定使用 BF16，需要服务器环境支持这种计算精度。
    if not torch.cuda.is_bf16_supported():
        # 环境不支持时直接说明原因。
        raise RuntimeError("当前 GPU 环境不支持 BF16，请检查 GPU 和 PyTorch 镜像。")

    # 固定初始化和样本打乱使用的基础随机种子。
    torch.manual_seed(42)

    # 直接采用正式模型配置，不另设小模型尺寸。
    config = EdgeMindConfig()

    # 加载并核对已有分词器。
    tokenizer = load_tokenizer(config)

    # 从指定的 JSONL 语料按需生成训练样本。
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 每次读取一批并打乱记录；批量和读取进程数由基本设置指定。
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 创建随机初始化的正式模型，并移动到 GPU。
    model = EdgeMindForCausalLM(config).to(device)

    # 直接使用 AdamW 的默认历史平均设置，明确关闭权重衰减，不进行参数分组。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    # 将保存位置转换为绝对路径。
    save_directory = Path(args.save_dir).resolve()

    # 创建输出目录，已有目录可以继续使用。
    save_directory.mkdir(parents=True, exist_ok=True)

    # 切换训练模式；真正修改权重仍由后面的 optimizer.step 完成。
    model.train()

    # 输出实际 GPU、模型参数量和样本数，便于确认训练对象。
    print(f"GPU={torch.cuda.get_device_name(device)}, 参数量={sum(p.numel() for p in model.parameters()):,}, 样本数={len(dataset)}", flush=True)

    # 每轮完整读取一次语料，不处理保存位置或跳过已读样本。
    for epoch in range(args.epochs):
        # 读取当前轮的每个批次，编号从零开始。
        for batch_index, batch in enumerate(loader):
            # 清除上一批的梯度，让当前批次独立决定这次更新。
            optimizer.zero_grad(set_to_none=True)

            # 把输入、注意力掩码和标签都移动到 GPU。
            batch = {name: value.to(device) for name, value in batch.items()}

            # 在前向和损失计算中使用 BF16 混合精度，模型权重仍保留 float32。
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 读取第二个返回值；模型内部已经完成下一位置预测和填充忽略。
                _, loss, _ = model(**batch)

            # 损失出现无效数值时停止，避免继续修改参数。
            if not torch.isfinite(loss).item():
                # 报告发生问题的轮次和批次。
                raise RuntimeError(f"第 {epoch + 1} 轮、第 {batch_index + 1} 批出现无效损失。")

            # 根据预测误差计算梯度，此时还没有修改模型权重。
            loss.backward()

            # 限制过大的整体梯度，梯度出现无效数值时拒绝更新。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

            # 每批更新一次模型参数，不累积多批梯度。
            optimizer.step()

            # 每十批和每轮最后一批打印当前批次损失。
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                # 输出实际轮次和批次，避免将当前损失误称为整轮平均。
                print(f"轮次={epoch + 1}/{args.epochs}, 批次={batch_index + 1}/{len(loader)}, loss={loss.item():.4f}", flush=True)

        # 每轮使用独立文件保存训练结果。
        weight_path = save_directory / f"pretrain_epoch_{epoch + 1}.pth"

        # 保存模型权重和结构配置，供后续正常创建模型再加载；不包含优化器或续训进度。
        torch.save({"model": model.state_dict(), "config": config.to_dict()}, weight_path)

        # 给出本轮保存的实际路径。
        print(f"已保存：{weight_path}", flush=True)


# 从项目根目录使用 python -m trainer.train_pretrain 启动，导入时不自动训练。
if __name__ == "__main__":
    # 使用标准解析器，不另建配置框架或训练管理器。
    parser = argparse.ArgumentParser(description="EdgeMind 单卡 GPU 预训练")

    # 必须明确指定正式 JSONL 语料，避免默认使用小样本文件。
    parser.add_argument("--data_path", required=True, help="包含 text 字段的预训练 JSONL 文件")

    # 默认将权重保存在 AutoDL 数据盘。
    parser.add_argument("--save_dir", default="/root/autodl-tmp/edgemind/checkpoints", help="模型权重保存目录")

    # 默认完整读取一次语料，可按实际训练计划调整轮数。
    parser.add_argument("--epochs", type=int, default=1, help="完整训练轮数")

    # 每批一条是当前正式模型的保守显存起点，可在 GPU 实测后调整。
    parser.add_argument("--batch_size", type=int, default=1, help="每批样本数")

    # 默认每条样本总长为五百一十二，包含起始和结束标记。
    parser.add_argument("--max_length", type=int, default=512, help="样本总长度")

    # 首版采用固定学习率，不增加预热或调度逻辑。
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="固定学习率")

    # 两个读取进程准备数据，实际模型计算仍在单张 GPU 上执行。
    parser.add_argument("--num_workers", type=int, default=2, help="数据读取进程数")

    # 将基本设置交给唯一的训练函数。
    train(parser.parse_args())
