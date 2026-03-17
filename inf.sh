
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 inference_fss.py --dataset_file coco --fold 0 --resume /home/star/Code/MMSANSA/output/Train/coco/0/checkpoint_best.pth --name_exp eval --shot 1 --prompt mask --adaptformer_stages 2 3 --seed 0 


# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 inference_fss.py --dataset_file coco --fold 0 --resume weights/pascal-1shot-fold0/checkpoint_best.pth --name_exp eval --shot 1 --prompt mask --adaptformer_stages 2 3 --seed 0 