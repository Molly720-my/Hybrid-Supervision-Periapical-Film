# IGLC-MAE Pre-Training

This folder contains the pre-training code for IGLC-MAE, a masked autoencoder that uses Intensity, Gradient, and Local Contrast adaptive masking for unlabeled periapical film images.


## Installation

Install PyTorch and torchvision for the CUDA version of the target machine, then install the remaining packages:

```bash
pip install -r requirements.txt
```

## Dataset format

Prepare a flat image directory:

```text
unlabeled_images/
  image_0001.jpg
  image_0002.jpg
  ...
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`.

## Pre-Training

```bash
python train_iglc_mae.py \
  --data_path /path/to/unlabeled_images \
  --output_dir outputs/iglc_mae_pretrain \
  --model iglc_mae_vit_base_patch16 \
  --input_size 224 \
  --mask_ratio 0.75 \
  --norm_pix_loss \
  --epochs 400 \
  --batch_size 64
```


They can be changed with command-line arguments, for example:

```bash
python train_iglc_mae.py \
  --data_path /path/to/unlabeled_images \
  --masking_w_var 0.15 \
  --masking_w_grad 0.35 \
  --masking_w_contrast 0.50
```

Use `--resume /path/to/checkpoint.pth` to resume a full pre-training run. Use `--pretrained /path/to/checkpoint.pth` only to initialize compatible model tensors from an existing checkpoint.

## Downstream Use

After pre-training, pass the encoder checkpoint to the downstream Mask2Former code through:

```bash
MODEL.VITMAE.CHECKPOINT_PATH /path/to/iglc_mae_encoder_or_checkpoint.pth
```

No pretrained weights are committed in this repository.

## Optional Visualization

The optional script `visualize_iglc_mask_top10.py` visualizes IGLC-based mask selection on locally supplied images:

```bash
python visualize_iglc_mask_top10.py \
  --image_dir /path/to/example_images \
  --out_dir outputs/iglc_mask_visualization
```
