import math
import random
import sys
from typing import Iterable, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler  # [新增] 引入混合精度

import util.misc as utils
# 引入之前封装好的 dice_loss，用于对抗 Few-Shot 下的小目标前景坍塌
from util.losses import loss_masks, dice_loss 
from util.promptable_utils import build_prompt_dict


def train_one_epoch(
    model: Module,
    data_loader: Iterable,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    args: Optional[object] = None,
) -> dict:

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # [新增] 初始化混合精度梯度缩放器
    scaler = GradScaler()

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        model.train()
        
        # --- 1. 解包数据 (Unpack Data) ---
        query_img = batch['query_img']         # [B, C, H, W]  (SAM Norm)
        query_mask = batch['query_mask']       # [B, H, W]
        support_imgs = batch['support_imgs']   # [B, S, C, H, W] (SAM Norm)
        support_masks = batch['support_masks'] # [B, S, H, W]
        class_names = batch['class_names']     # [B] List of strings
        
        clip_query_img = batch['clip_query_img']       # [B, C, 224, 224] (CLIP Norm)
        clip_support_imgs = batch['clip_support_imgs'] # [B, S, C, 224, 224] (CLIP Norm)
        clip_support_masks = batch['clip_support_masks'] # [B, S, H, W]

        if isinstance(class_names[0], tuple) or isinstance(class_names[0], list):
            batch_size = len(class_names[0])
            transposed_class_names = []
            
            for b in range(batch_size):
                sample_prompts = [prompt_tuple[b] for prompt_tuple in class_names]
                transposed_class_names.append(sample_prompts)
                
            class_names = transposed_class_names

        random_shot = args.shots
        # random_shot = random.randint(1, args.J)

        # --- 2. 构建 SAM2 输入序列 (Sequence Construction) ---
        samples = torch.cat([support_imgs[:, :random_shot], query_img.unsqueeze(1)], dim=1)   # [B, T, C, H, W]
        masks   = torch.cat([support_masks[:, :random_shot], query_mask.unsqueeze(1)], dim=1) # [B, T, H, W]

        # --- 3. 构建 CLIP 输入序列 ---
        clip_samples = torch.cat([
            clip_support_imgs[:, :random_shot], 
            clip_query_img.unsqueeze(1)
        ], dim=1) # Result: [B, T, 3, 224, 224]

        # --- 4. 构建 Prompt ---
        prompt_dict = build_prompt_dict(masks, args.prompt, n_shots=random_shot, train_mode=True, device=model.device)
        
        # --- 5. 送入设备 (To Device) ---
        samples = samples.to(device)
        clip_samples = clip_samples.to(device) 
        current_clip_masks = clip_support_masks[:, :random_shot].to(device)
        
        # --- 6. 模型前向传播 (Forward Pass) 加入 AMP 包装 ---
        # [修改] 利用 autocast 开启混合精度加速
        with autocast(dtype=torch.float16): 
            outputs = model(samples, prompt_dict, text_prompts=class_names, clip_inputs=clip_samples, clip_masks=current_clip_masks)

            # loss on last T or last (T-1) frames (skip first) depending on prompt
            T = samples.shape[1]
            use_frames = T if args.prompt != "mask" else 1
            losses_dict = loss_masks(outputs["pred_masks"], masks, num_frames=use_frames)

            # ================= [真正的深度监督辅助损失 - 动态衰减版] =================
            if outputs.get("aux_pseudo_mask") is not None:
                # aux_mask 的尺寸是较小的特征图尺寸（如 14x14 或 64x64）
                aux_mask = outputs["aux_pseudo_mask"] # [B, 1, H_feat, W_feat]
                
                # 获取 Query 的真实高分辨率掩码
                gt_query_mask = masks[:, -1:].to(device).float()  # [B, 1, H, W]
                
                # [核心优化 3] 尺度对齐损失：不要上采样 Logit，而是下采样 GT 掩码到特征图尺寸
                # 采用 nearest 最近邻下采样，确保降采样后依然是纯粹的 0/1 二值掩码
                gt_query_mask_down = F.interpolate(
                    gt_query_mask, 
                    size=aux_mask.shape[-2:], 
                    mode='nearest'
                )
                
                # 使用对齐后的小尺寸计算联合损失
                aux_bce = F.binary_cross_entropy_with_logits(aux_mask, gt_query_mask_down)
                aux_dice = dice_loss(aux_mask, gt_query_mask_down, num_boxes=gt_query_mask.shape[0])

                # --- 方案一：动态衰减逻辑 ---
                total_epochs = args.epochs if hasattr(args, 'epochs') else 30 
                progress = epoch / total_epochs
                
                if progress < 0.5:
                    decay_factor = 1.0
                else:
                    decay_factor = 1.0 - ((progress - 0.5) * 2) * 0.9 
                
                final_weight = 0.5 * decay_factor
                losses_dict['loss_vlp_aux'] = (aux_bce + aux_dice) * final_weight
            # ======================================================================

            # total loss
            loss = sum(losses_dict.values())

        # reduce for logging
        losses_reduced = utils.reduce_dict(losses_dict)
        loss_value = sum(losses_reduced.values()).item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(losses_reduced)
            sys.exit(1)

        # --- 7. 模型反向传播 (Backward Pass) 使用 AMP Scaler ---
        optimizer.zero_grad(set_to_none=True) # avoids unnecessary memory allocations for zero grad tensor
        
        # [修改] 混合精度缩放梯度与更新
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # 解码梯度，准备裁剪
        
        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **losses_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if math.isfinite(grad_total_norm.item()):
            metric_logger.update(grad_norm=grad_total_norm)
        # metric_logger.update(grad_norm=grad_total_norm)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}