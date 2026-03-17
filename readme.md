

# ACRIS-SAM2 Official Guide

This repository contains the official implementation of ACRIS-SAM2, a Few-Shot Segmentation model built upon SAM2 and CLIP. Below are the detailed instructions for environment setup, data preparation, weight acquisition, and testing.

## 1. Environment Setup

* The underlying dependencies require PyTorch 2.7.1 with CUDA 12.6 support and its corresponding Torchvision 0.22.1.
* You can install all necessary dependencies in one click using the provided `requirements.txt` file. Run the following command:
```bash
  pip install -r requirements.txt

```


## 2. Dataset Download and Preparation

* The model supports evaluation on various public datasets, covering common visual segmentation datasets (`coco`, `pascal_voc`, etc.).
* **Please refer to `./docs/data.md` for detailed instructions on dataset download, formatting, and preparation.** * By default, the root directory for datasets is expected to be `../data`. If your datasets are located elsewhere, you can override this by specifying the `--data_root` parameter during execution.

## 3. Weight Acquisition

Before running evaluations, you need to download and correctly place the pre-trained weights and model checkpoints:

* **SAM2 Weights:** Please place the downloaded foundational SAM2 visual encoder weights (default is `large`) into the `pretrain/` directory.
* **CLIP Weights:** Please place the CLIP model weights (default is `ViT-B/16`) into the `~/.cache/clip/` directory.
* **Result Weights (Checkpoints):** You can download the trained model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1gyac8nUg4NPQFGTEDAgERYLkEE6mKP0B?usp=drive_link). After downloading (e.g., `checkpoint_best.pth`), please place them inside the `weights/` directory in your project root.

## 4. Testing and Inference

* The evaluation process is handled by `inference_fss.py`, which calculates the mIoU metric for few-shot segmentation tasks.
* You can use the `torchrun` utility to launch the test, which supports distributed evaluation. Below is an example command for running a 1-shot evaluation on fold 0 of the `coco` dataset:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 inference_fss.py \
    --dataset_file {coco|pascal_voc} \
    --fold 0 \
    --resume weights/{coco|pascal_voc}-1shot/fold0/checkpoint_best.pth \
    --name_exp eval \
    --shot 1 \
    --prompt mask \
    --adaptformer_stages 2 3 \
    --seed 0 

```


* **Core Arguments:**
* `--dataset_file`: Specifies the dataset to evaluate on (e.g., `coco`).
* `--fold`: Specifies the cross-validation fold (e.g., `0`).
* `--resume`: Path to your downloaded model checkpoint (e.g., `weights/.../checkpoint_best.pth`).
* `--shot`: Number of support frames for few-shot learning.
* `--prompt`: The type of prompt to use for support frames. Choices include `mask`, `scribble`, `box`, `point`, or `multi`.
* `--adaptformer_stages`: Specifies which AdaptFormer stages to enable for feature fusion (e.g., `2 3`).
* `--visualize`: Append this flag to the command if you wish to save the generated segmentation masks for qualitative analysis.


