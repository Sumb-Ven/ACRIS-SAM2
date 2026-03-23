import argparse
import copy
import json
from pathlib import Path
import os
import itertools  # [新增] 用于切分数据
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.commons import resume_from_checkpoint, trainable_state_dict
import datasets.samplers as samplers
from datasets import build_dataset
from models.acris_sam2.acris_sam2 import build_acris_sam2
from inference_fss import eval_fss
from engine import train_one_epoch
import opts
from util.commons import setup_logging, make_deterministic


# [新增] 包装类：用于修复 MetricLogger 的 len() 报错
class ChunkedLoader:
    """
    一个简单的包装类，用于给 itertools.islice 添加 __len__ 属性，
    以满足 engine.py 中 MetricLogger 的进度条打印需求。
    """
    def __init__(self, iterator, length):
        self.iterator = iterator
        self.length = length

    def __iter__(self):
        return self.iterator

    def __len__(self):
        return self.length


def main(args):
    utils.init_distributed_mode(args)
    rank = utils.get_rank()
    setup_logging(save_dir=args.output_dir, console="info", rank=rank)
    make_deterministic(args.seed + rank)  # fix the seed for reproducibility

    print(args)

    device = torch.device(args.device)
    # 注意：build_sansa 参数根据你之前的 opts.py 和 sansa.py 调整
    model = build_sansa(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device, clip_version=args.clip_version)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print(f'number of params: {n_parameters_tot}')

    head, fix = [], []
    for k, v in model_without_ddp.named_parameters():
        (head if v.requires_grad else fix).append(v)

    print(f'Trainable parameters: {sum(p.numel() for p in head)}')
    print(f'Parameters fixed: {sum(p.numel() for p in fix)}')

    param_list = [{'params': head, 'initial_lr': args.lr}]
    optimizer = torch.optim.AdamW(param_list, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), fused=True)

    cfg = copy.deepcopy(args)
    cfg.shots = args.J
    dataset_train = build_dataset(args.dataset_file, image_set='train', args=cfg)

    args.batch_size = int(args.batch_size / args.ngpu)
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(data_loader_train)
    )

    output_dir = Path(args.output_dir)
    
    # [新增] 最佳 mIoU 记录变量
    best_miou = 0.0
    start_epoch = 0

    if args.resume:
        model_without_ddp, optimizer, lr_scheduler = resume_from_checkpoint(args.resume, model_without_ddp, optimizer, lr_scheduler, args)
        start_epoch = args.start_epoch
        # 尝试从 checkpoint 中恢复 best_miou (如果你的 resume 函数支持返回 extra info 最好，否则这里重置为0)
        # 如果 checkpoint 里没有 best_miou，这行可能会需要调整，这里假设它是新的训练或不强求恢复 best 值
        pass 

    # =========================================================
    # [新增] 定义内部验证函数，包含 "Best Model" 保存逻辑
    # =========================================================
    def validate_and_save(current_epoch, step_suffix=""):
        nonlocal best_miou
        print(f"\n>> Start validation (Epoch {current_epoch} - {step_suffix})")
        
        # 验证
        val_stats = eval_fss(model, args) 
        # eval_fss 在 inference_fss.py 中返回的是 miou (float)
        current_miou = val_stats 

        if args.output_dir:
            # 1. 保存常规 Checkpoint (用于 Resume)
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 仅在 Epoch 结束时保存带编号的 checkpoint，节省空间
            if step_suffix == "end": 
                checkpoint_paths.append(output_dir / f'checkpoint{current_epoch:04}.pth')
            
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': trainable_state_dict(model_without_ddp),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': current_epoch,
                    'args': args,
                    'best_miou': best_miou, 
                }, checkpoint_path)

            # 2. [核心逻辑] 保存最佳模型
            if current_miou > best_miou:
                print(f"★ New Best mIoU: {current_miou:.2f} (Previous: {best_miou:.2f}). Saving checkpoint_best.pth...")
                best_miou = current_miou
                utils.save_on_master({
                    'model': trainable_state_dict(model_without_ddp),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': current_epoch,
                    'args': args,
                    'best_miou': best_miou,
                }, output_dir / 'checkpoint_best.pth')
            else:
                print(f"Current mIoU: {current_miou:.2f} is not better than best: {best_miou:.2f}")

    print("Start training")
    
    # [配置] 验证频率：每个 Epoch 验证 2 次 (即每 50% 数据验证一次)
    VAL_FREQ = 2 
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # 获取数据迭代器
        data_loader_iter = iter(data_loader_train)
        len_loader = len(data_loader_train)
        chunk_size = len_loader // VAL_FREQ
        
        # 将一个 Epoch 拆分为 VAL_FREQ 个阶段
        for chunk_idx in range(VAL_FREQ):
            
            # 计算当前 chunk 需要跑多少步
            if chunk_idx == VAL_FREQ - 1:
                # 最后一个 chunk 包含剩余所有数据
                steps_this_chunk = len_loader - (chunk_size * (VAL_FREQ - 1))
                step_suffix = "end" 
            else:
                steps_this_chunk = chunk_size
                step_suffix = f"part_{chunk_idx+1}"

            print(f"\nEpoch {epoch} [{chunk_idx+1}/{VAL_FREQ}]: Training for {steps_this_chunk} steps...")

            # [修改] 使用 islice 切分，并用 ChunkedLoader 包装以提供 __len__
            sub_loader_iter = itertools.islice(data_loader_iter, steps_this_chunk)
            sub_loader = ChunkedLoader(sub_loader_iter, steps_this_chunk)

            # 训练这个 Chunk
            # 注意：train_one_epoch 里的 log_every 现在会读取 len(sub_loader)
            train_stats = train_one_epoch(
                    model, sub_loader, optimizer, device, epoch,
                    args.clip_max_norm, lr_scheduler=lr_scheduler, args=args)
            
            # 记录日志
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'chunk': chunk_idx,
                         'n_parameters': n_parameters_tot}
            
            if utils.is_main_process():
                print(json.dumps(log_stats))

            # [执行验证与保存]
            if args.output_dir:
                validate_and_save(epoch, step_suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA training', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.name_exp)

    main(args)