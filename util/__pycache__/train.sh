# COCO-20i, fold 0 (strict few-shot)
# sigle-gpu
# torchrun --nproc_per_node=2  main.py --batch_size 16 --name_exp train_edge_coco_f0_2node --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask

# torchrun --nproc_per_node=2  main.py --batch_size 16 --name_exp train_edge_coco_f1_2node --dataset_file coco --fold 1 --adaptformer_stages 2 3 --prompt mask

torchrun --nproc_per_node=2  main.py --batch_size 4 --name_exp decoder_3all2 --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask --sam2_version large --clip_version ViT-B/16

# torchrun --nproc_per_node=2  main.py --batch_size 8 --name_exp 0_w_dec_mem_adap_100coco --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask --sam2_version large --clip_version ViT-B/16

# torchrun --nproc_per_node=2  main.py --batch_size 8 --name_exp 0_w_dec_mem_adap_100coco --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask --sam2_version large --clip_version ViT-B/16

# torchrun --nproc_per_node=2  main.py --batch_size 8 --name_exp 0fusion-B/321 --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask --sam2_version large --clip_version ViT-B/32

# torchrun --nproc_per_node=2  main.py --batch_size 8 --name_exp 0fusion-B/322 --dataset_file coco --fold 0 --adaptformer_stages 2 3 --prompt mask --sam2_version large --clip_version ViT-B/32


