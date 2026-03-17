import argparse
import sys
import random  # [新增]
import numpy as np  # [新增]
from os.path import join
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from models.acris_sam2.acris_sam2 import build_acris_sam2
from datasets import build_dataset
from util.commons import make_deterministic, setup_logging, resume_from_checkpoint
import util.misc as utils
from util.promptable_utils import build_prompt_dict
from util.metrics import AverageMeter, Evaluator


def main(args: argparse.Namespace) -> float:
    setup_logging(args.output_dir, console="info", rank=0)
    
    # 这里的设置保证独立运行时是确定性的
    make_deterministic(args.seed)
    print(args)

    model = build_acris_sam2(args.sam2_version, args.adaptformer_stages, args.channel_factor, args.device)
    device = torch.device(args.device)
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_from_checkpoint(args.resume, model)

    print(f"number of params: {n_parameters}")
    print('Start inference')

    mIoU = eval_fss(model, args)
    return mIoU


def eval_fss(model: torch.nn.Module, args: argparse.Namespace) -> float:
    """
    Evaluate SANSA on the few-shot segmentation benchmark.
    Computes and prints mIoU across the validation set.
    """
    
    # =========================================================================
    # [关键修改] 随机状态隔离：保存当前状态 -> 重置种子 -> 验证 -> 恢复状态
    # 目的：确保无论在训练的哪个阶段调用，验证集采样的 Episodes 都是完全一致的
    # =========================================================================
    
    # 1. 保存当前的随机状态 (防止打断训练的随机性)
    rng_state_torch = torch.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    rng_state_numpy = np.random.get_state()
    rng_state_random = random.getstate()

    # 2. 强制重置种子 (确保验证集的 Support/Query 采样顺序固定)
    # 这解决了 main.py 训练中验证指标与 inference_fss.py 不一致的问题
    print(f"| Eval Consistency | Resetting seed to {args.seed} for deterministic validation...")
    make_deterministic(args.seed) 
    
    # =========================================================================

    # load data
    validation_ds = 'coco' if args.dataset_file == 'multi' else args.dataset_file 
    print(f'Evaluating {validation_ds} - fold: {args.fold}')
    
    # 此时构建 Dataset，内部的 random/numpy 调用将基于重置后的种子
    ds = build_dataset(validation_ds, image_set='val', args=args)
    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    model.eval()
    average_meter = AverageMeter(args.dataset_file, ds.class_ids, ds.nclass)

    pbar = tqdm(dataloader, ncols=80, desc='runn avg.', disable=(utils.get_rank() != 0), file=sys.stderr, dynamic_ncols=True)
    for idx, batch in enumerate(pbar):
        query_img, query_mask = batch['query_img'], batch['query_mask']
        support_imgs, support_masks = batch['support_imgs'], batch['support_masks']

        clip_query_img = batch['clip_query_img']       # [B, 3, 224, 224]
        clip_support_imgs = batch['clip_support_imgs'] # [B, S, 3, 224, 224]
        clip_support_masks = batch['clip_support_masks'] # [B, S, H, W]
        class_names = batch['class_names']

        if isinstance(class_names[0], tuple) or isinstance(class_names[0], list):
            # class_names 目前结构: [ (S1_P1, S2_P1...), (S1_P2, S2_P2...) ]
            # 我们需要结构: [ [S1_P1, S1_P2...], [S2_P1, S2_P2...] ]
            
            batch_size = len(class_names[0])
            transposed_class_names = []
            
            for b in range(batch_size):
                # 收集第 b 个样本的所有提示
                sample_prompts = [prompt_tuple[b] for prompt_tuple in class_names]
                transposed_class_names.append(sample_prompts)
                
            class_names = transposed_class_names

        imgs = torch.cat([support_imgs[0], query_img]).unsqueeze(0) # b t c h w
        clip_imgs = torch.cat([clip_support_imgs[0], clip_query_img]).unsqueeze(0) # b t c h w
        img_h, img_w = imgs.shape[-2:]

        imgs = imgs.to(args.device)
        clip_imgs = clip_imgs.to(args.device)
        clip_support_masks = clip_support_masks.to(args.device)
        prompt_dict = build_prompt_dict(support_masks, args.prompt, n_shots=args.shots, train_mode=False, device=model.device)

        with torch.no_grad():
            # 6. 传入完整的参数
            outputs = model(
                imgs, 
                prompt_dict, 
                text_prompts=class_names,      
                clip_inputs=clip_imgs,         
                clip_masks=clip_support_masks  
            )

        pred_masks = outputs["pred_masks"].unsqueeze(0)  # [1, T, h, w]
        pred_masks = F.interpolate(pred_masks, size=(img_h, img_w), mode='bilinear', align_corners=False) 
        pred_masks = (pred_masks.sigmoid() > args.threshold)[0].cpu()

        area_inter, area_union = Evaluator.classify_prediction(pred_masks[-1:].float(), batch, device=imgs.device)
        average_meter.update(area_inter, area_union, batch['class_id'].cuda())

        if (idx + 1) % 50 == 0:
            miou, _, _ = average_meter.compute_iou()
            pbar.set_description(f"Runn. Avg mIoU = {miou:.1f}")

        if args.visualize:
            from util.visualization import visualize_episode
            visualize_episode(
                support_imgs=[support_imgs[0, i].cpu() for i in range(args.shots)],
                query_img=query_img[0].cpu(),
                query_gt=(query_mask[0].numpy() > 0),
                query_pred=pred_masks[-1].numpy(),
                prompt_dict=prompt_dict,
                out_dir=args.output_dir,
                idx=idx,
                src_size=model.sam.image_size,
                iou=area_inter/area_union,
            )
            
    average_meter.write_result(args.dataset_file)
    miou, fb_iou, _ = average_meter.compute_iou()
    print('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, miou, fb_iou.item()))
    print('==================== Finished Testing ====================')

    # =========================================================================
    # [关键修改] 恢复之前保存的随机状态
    # =========================================================================
    torch.set_rng_state(rng_state_torch)
    if rng_state_cuda is not None:
        torch.cuda.set_rng_state(rng_state_cuda)
    np.random.set_state(rng_state_numpy)
    random.setstate(rng_state_random)
    # =========================================================================

    return miou


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SANSA evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = join(args.output_dir, args.name_exp)
    main(args)