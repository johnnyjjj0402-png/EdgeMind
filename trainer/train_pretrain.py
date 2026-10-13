# 参考 MiniMind 的单卡训练与检查点思路，支持从已有 Base 开始预训练，以及中断后恢复同一次训练。
# 来源：https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/train_pretrain.py
# 检查点与固定顺序恢复的参考：https://github.com/jingyaogong/minimind/blob/f659b55761b754d306bd140573493a6543cafd7f/trainer/trainer_utils.py

# 使用标准库读取正式语料路径和基本训练设置。
import argparse

# 使用标准库的圆周率和余弦函数，安排学习率的预热和衰减曲线。
import math

# 使用路径对象创建权重保存目录。
from pathlib import Path

# 复用 PyTorch 的计算、自动求导、优化器和权重保存能力。
import torch

# 使用标准读取器按指定的样本编号组批，正文读取和分词仍交给数据集。
from torch.utils.data import DataLoader

# 复用现有数据集，它已负责起止标记、注意力掩码和填充标签。
from dataset.lm_dataset import PretrainDataset

# 模型结构统一从正式配置类读取。
from model.configuration_edgemind import EdgeMindConfig

# 使用项目自己的模型，其第二个返回值是训练损失。
from model.modeling_edgemind import EdgeMindForCausalLM

# 加载已有分词器，不在训练脚本中训练或修改词表。
from model.tokenization_edgemind import load_tokenizer


# 先写完临时文件再替换正式文件；写入中断时，上一份完整文件仍然可以使用。
def save_checkpoint(state: dict, path: Path):
    # 临时文件与正式文件放在同一目录，保证完成后的替换发生在同一文件系统内。
    temporary_path = path.with_name(path.name + ".tmp")
    # 同步写入全部内容，失败时不会提前覆盖正式检查点。
    torch.save(state, temporary_path)
    # 写入成功后一次性替换正式文件，每种检查点只保留最新的完整内容。
    temporary_path.replace(path)


# 加载训练起点、读取正文、更新模型，并定期记录可以恢复的训练状态。
def train(args):
    # 基本训练设置与保存间隔必须为正，学习率还必须是有限数值。
    if args.epochs <= 0 or args.batch_size <= 0 or args.save_interval <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        # 在创建模型前指出基本设置错误。
        raise ValueError("epochs、batch_size、save_interval 必须大于零，learning_rate 必须是有限正数。")

    # 加载 Base 是开始新阶段，恢复检查点是继续旧阶段，两种起点不能同时指定。
    if args.init_from and args.resume:
        # 防止使用者误以为一个起点会自动覆盖另一个。
        raise ValueError("init_from 与 resume 只能选择一个。")

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

    # 固定初始随机种子；恢复训练时会在模型创建完成后还原保存的随机状态。
    torch.manual_seed(42)

    # 有起点时先在 CPU 读取本项目的检查点，避免读取文件时额外占用 GPU 显存。
    source_path = args.resume or args.init_from
    # 没有指定起点时仍允许从零训练，保留首版流程的复现入口。
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=True) if source_path else None
    # 恢复文件需要记录更新历史与进度，只有 Base 权重的文件不能当作恢复文件使用。
    if args.resume and "optimizer" not in checkpoint:
        # 明确区分两个文件的用途，避免出现难懂的字段缺失错误。
        raise ValueError("resume 需要 pretrain_resume.pth；只有模型权重的文件请使用 init_from。")
    # 已有权重的模型结构以保存时的配置为准，不用当前默认值重新猜测结构。
    config = EdgeMindConfig(**checkpoint["config"]) if checkpoint else EdgeMindConfig()

    # 加载并核对已有分词器。
    tokenizer = load_tokenizer(config)

    # 从指定的 JSONL 语料按需生成训练样本。
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 最后一批不足 batch_size 时仍训练，向上取整得到每轮的完整批次数。
    batches_per_epoch = (len(dataset) + args.batch_size - 1) // args.batch_size
    # 记录数据文件大小和修改时间，用于拒绝在恢复时悄悄换成另一份数据。
    data_stat = dataset.data_path.stat()
    # 恢复时这些设置必须相同，否则原来的样本位置或学习率进度就不再对应。
    training_settings = {
        # 固定数据位置、大小和修改时间，不需要额外遍历整份语料计算哈希。
        "data_path": str(dataset.data_path), "data_size": data_stat.st_size, "data_mtime_ns": data_stat.st_mtime_ns,
        # 样本数量、每批条数与截断长度共同决定每轮怎样读取正文。
        "samples": len(dataset), "batch_size": args.batch_size, "max_length": args.max_length,
        # 轮数与峰值学习率共同决定后面的学习率曲线。
        "epochs": args.epochs, "learning_rate": args.learning_rate,
    # 只记录恢复所必需的设置；读取进程数和保存间隔可以在重启时调整。
    }
    # 改变数据或训练安排会使精确恢复失效，在更新参数前明确停止。
    if args.resume and checkpoint["training_settings"] != training_settings:
        # 不自动改批次位置或重新安排曲线，避免丢失或重复学习部分样本。
        raise ValueError("恢复时数据文件、epochs、batch_size、max_length 和 learning_rate 必须与原训练一致。")

    # 正常创建模型，使包括位置角度表在内的内部数据完成初始化。
    model = EdgeMindForCausalLM(config)
    # 加载 Base 或恢复文件中的已有权重，所有参数名称与尺寸必须匹配。
    if checkpoint:
        # 严格加载，不忽略缺失或多出的参数。
        model.load_state_dict(checkpoint["model"], strict=True)
    # 将已经加载好权重的模型移动到 GPU。
    model = model.to(device)

    # 直接使用 AdamW 的默认历史平均设置，明确关闭权重衰减，不进行参数分组。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    # 每个批次更新一次参数，总步数是完整批次数乘以本阶段的轮数。
    total_steps = batches_per_epoch * args.epochs

    # 用本阶段前百分之一的更新逐渐增大步子，让已有 Base 平稳适应新的正文数据。
    warmup_steps = max(1, total_steps // 100)

    # 学习率先线性升到设定值，再按余弦曲线降到设定值的十分之一，收尾时步子变小让权重稳定。
    def lr_lambda(step):
        # step 从零开始，首批也要有正学习率，避免做完计算却没有更新模型权重。
        if step < warmup_steps:
            # 最后一批预热达到设定的峰值。
            return (step + 1) / warmup_steps
        # 预热后平滑减小步子；最后一次更新到达十分之一，曲线完成后保持在这个值。
        progress = min(1.0, (step - warmup_steps + 1) / max(1, total_steps - warmup_steps))
        # 余弦从 1 平滑降到 0，乘以 0.9 再加 0.1，最终保留设定值的十分之一。
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    # 把上面的曲线挂到优化器上，每完成一次参数更新就向前推进一步。
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 默认从本阶段的第一轮、第一批开始；这里的内部编号从零计数。
    start_epoch, start_batch = 0, 0
    # 恢复同一次训练时，还原参数更新历史、学习率位置和下一批位置。
    if args.resume:
        # 模型已经在 GPU 上，再加载更新历史，让优化器把所需状态放到对应设备。
        optimizer.load_state_dict(checkpoint["optimizer"])
        # 已创建相同曲线，再恢复曲线走到哪一步，避免重新预热。
        scheduler.load_state_dict(checkpoint["scheduler"])
        # next_batch 表示这一轮已完成的批数，也正是下一批的零起始编号。
        start_epoch, start_batch = checkpoint["epoch"], checkpoint["next_batch"]
        # 模型创建过程消耗过随机数，因此到此时再还原 CPU 的随机状态。
        torch.set_rng_state(checkpoint["torch_rng_state"])
        # 同时还原当前 GPU 的随机状态，使随机丢弃等操作接着原来的状态进行。
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        # 显示已经完成的位置，避免误以为恢复后又从第一批开始。
        print(f"恢复训练：第 {start_epoch + 1} 轮已完成 {start_batch} 批。", flush=True)
    # 新阶段只加载 Base 的学习成果，不继承旧阶段的优化器或学习率安排。
    elif args.init_from:
        # 在训练日志中留下实际起点，方便回溯实验。
        print(f"加载 Base：{Path(args.init_from).resolve()}", flush=True)
    # 起点状态已经复制到模型和优化器，释放读取检查点占用的 CPU 内存。
    del checkpoint

    # 将保存位置转换为绝对路径。
    save_directory = Path(args.save_dir).resolve()

    # 创建输出目录，已有目录可以继续使用。
    save_directory.mkdir(parents=True, exist_ok=True)
    # 完整恢复状态循环写入同一个文件，不为每个保存点累积一份大文件。
    resume_path = save_directory / "pretrain_resume.pth"
    # 新阶段不能覆盖提供初始权重的 Base 文件，误用旧保存目录时提前指出。
    if args.init_from and Path(args.init_from).resolve() in {(save_directory / f"pretrain_epoch_{index + 1}.pth").resolve() for index in range(args.epochs)} | {resume_path.resolve()}:
        # 使用单独目录即可保留旧 Base 作为训练前的对照。
        raise ValueError("保存目录会覆盖起始 Base，请为本阶段指定独立的 save_dir。")

    # 切换训练模式；真正修改权重仍由后面的 optimizer.step 完成。
    model.train()

    # 输出实际 GPU、模型参数量和样本数，便于确认训练对象。
    print(f"GPU={torch.cuda.get_device_name(device)}, 参数量={sum(p.numel() for p in model.parameters()):,}, 样本数={len(dataset)}", flush=True)

    # 从保存的位置开始，每一轮仍按固定、可重建的顺序读取全部正文。
    for epoch in range(start_epoch, args.epochs):
        # 单独使用 CPU 生成器打乱样本，不消耗模型随机丢弃使用的随机状态。
        data_generator = torch.Generator().manual_seed(42 + epoch)
        # 相同轮次得到相同顺序，恢复时无需把已完成批次再次读取和分词。
        indices = torch.randperm(len(dataset), generator=data_generator).tolist()
        # 只有恢复的第一轮需要跳过已完成批次，后面的轮次都从头读取。
        skip_batches = start_batch if epoch == start_epoch else 0
        # 每批条数固定，用完成的批数换算成已经读取的样本数量。
        remaining_indices = indices[skip_batches * args.batch_size:]
        # 只读取剩余编号；独立生成器也负责读取进程的种子，保持模型随机状态不受影响。
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=remaining_indices, num_workers=args.num_workers, generator=data_generator)

        # 批次编号延续原来的整轮位置，不把剩余数据重新编号成第一批。
        for batch_index, batch in enumerate(loader, start=skip_batches):
            # 清除上一批的梯度，让当前批次独立决定这次更新。
            optimizer.zero_grad(set_to_none=True)

            # 把输入、注意力掩码和标签都移动到 GPU。
            batch = {name: value.to(device) for name, value in batch.items()}
            # 在推进曲线之前记录本批真正使用的步子，后面打印它而不是下一批的学习率。
            current_lr = optimizer.param_groups[0]["lr"]

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

            # 参数更新后推进学习率曲线，让下一批使用新的学习率。
            scheduler.step()

            # 累计步数覆盖前面的完整轮次和本轮已完成批次，用于决定何时保存。
            global_step = epoch * batches_per_epoch + batch_index + 1
            # 每十批和每轮最后一批打印本批误差与实际学习率。
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == batches_per_epoch:
                # 恢复后仍显示整轮进度，便于与中断之前的日志衔接。
                print(f"轮次={epoch + 1}/{args.epochs}, 批次={batch_index + 1}/{batches_per_epoch}, loss={loss.item():.4f}, lr={current_lr:.2e}", flush=True)

            # 定期保存完整状态，最后一批也保存，避免必须等整轮结束才能保住训练成果。
            if global_step % args.save_interval == 0 or batch_index + 1 == batches_per_epoch:
                # 保存模型、更新历史、学习率、位置和随机状态，供下一次原样恢复。
                state = {
                    # 权重与配置也保留，恢复文件可以由现有续写测试脚本直接加载。
                    "model": model.state_dict(), "config": config.to_dict(),
                    # 两份历史记录保证恢复后接着原来的更新节奏进行。
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    # 记录本轮编号以及已完成的批次数，下一次直接读取后面的样本。
                    "epoch": epoch, "next_batch": batch_index + 1,
                    # 保存影响样本位置和学习率的设置，恢复时进行核对。
                    "training_settings": training_settings,
                    # 保存 CPU 与当前 GPU 的随机状态，不保存依赖外部库的随机对象。
                    "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(device),
                # 整个字典只包含本项目需要的普通数据与张量，可以安全方式读取。
                }
                # 完整写入后替换上一份恢复文件。
                save_checkpoint(state, resume_path)
                # 保存完成后释放临时字典，避免它额外持有模型或优化器状态的引用。
                del state
                # 说明本次已保存到哪一步，方便判断中断时最多会丢失多少批次。
                print(f"已保存恢复点：{resume_path}，累计更新 {global_step} 次。", flush=True)

        # 每轮使用独立文件保存训练结果。
        weight_path = save_directory / f"pretrain_epoch_{epoch + 1}.pth"

        # 整轮结果只保存权重与配置，继续兼容现有续写测试和 SFT 入口；恢复状态保存在另一份文件。
        save_checkpoint({"model": model.state_dict(), "config": config.to_dict()}, weight_path)

        # 给出本轮保存的实际路径。
        print(f"已保存：{weight_path}", flush=True)


# 从项目根目录使用 python -m trainer.train_pretrain 启动，导入时不自动训练。
if __name__ == "__main__":
    # 使用标准解析器，不另建配置框架或训练管理器。
    parser = argparse.ArgumentParser(description="EdgeMind 单卡 GPU 预训练")

    # 必须明确指定正式 JSONL 语料，避免默认使用小样本文件。
    parser.add_argument("--data_path", required=True, help="包含 text 字段的预训练 JSONL 文件")

    # 加载已有 Base 时开始一个新的预训练阶段；省略两个起点参数时才从随机权重开始。
    parser.add_argument("--init_from", help="本项目已有 Base 权重的 .pth 路径")

    # 恢复中断的同一次训练；数据、批量、长度、学习率和轮数须保持一致。
    parser.add_argument("--resume", help="完整 pretrain_resume.pth 的路径，与 init_from 互斥")

    # 默认使用独立目录，把这次全量预训练的结果与旧 Base、SFT 权重分开保存。
    parser.add_argument("--save_dir", default="/root/autodl-tmp/edgemind/checkpoints/pretrain_full", help="本阶段权重和恢复文件的保存目录")

    # 每隔一千次参数更新保存完整状态，最多回退到最近一次成功保存的位置。
    parser.add_argument("--save_interval", type=int, default=1000, help="完整恢复检查点的保存间隔，单位为参数更新次数")

    # 默认完整读取一次语料，可按实际训练计划调整轮数。
    parser.add_argument("--epochs", type=int, default=1, help="完整训练轮数")

    # SDPA 减少注意力中间结果的开销，每批八条作为起点，实际显存需要在 GPU 上验证。
    parser.add_argument("--batch_size", type=int, default=8, help="每批样本数")

    # 默认每条样本总长为五百一十二，包含起始和结束标记。
    parser.add_argument("--max_length", type=int, default=512, help="样本总长度")

    # 学习率指预热结束后的峰值，训练中会按余弦曲线自动降到它的十分之一。
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="预热后的峰值学习率")

    # 两个读取进程准备数据，实际模型计算仍在单张 GPU 上执行。
    parser.add_argument("--num_workers", type=int, default=2, help="数据读取进程数")

    # 将基本设置交给唯一的训练函数。
    train(parser.parse_args())
