# Periapical Film Segmentation with Mask2Former

This repository contains the downstream Mask2Former training and inference code used for seven-class periapical film segmentation.


## Dataset format

Install PyTorch and Detectron2 according to the CUDA version of the target machine, then install the remaining Python packages:

```bash
pip install -r requirements.txt
```

Prepare a dataset root with this structure:

```text
dataset_root/
  classes.txt
  images/
    train/
    val/
  annotations/
    train/
    val/
```

Each annotation mask should be a PNG/JPG file with the same basename as its image. Pixel values should use:

```text
0 background
1 tooth
2 pulp
3 caries
4 fillings
5 root_canal_fillings
6 periapical_lesion
7 crown
```

## Training

```bash
python train_net.py \
  --num-gpus 1 \
  --config-file configs/periapical/mask2former_iglc_mae_vit_base_7class.yaml \
  --dataset-root /path/to/dataset_root \
  MODEL.VITMAE.CHECKPOINT_PATH /path/to/local/iglc_mae_encoder.pth \
  OUTPUT_DIR outputs/periapical_experiment
```

Use `MODEL.WEIGHTS /path/to/local/downstream_checkpoint.pth` only when resuming or fine-tuning a full Mask2Former segmentation checkpoint. Use `--val-dataset-root` if the validation data are stored under a different root.

## Evaluation

Detectron2's semantic segmentation evaluator is used by default when running:

```bash
python train_net.py \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/periapical/mask2former_iglc_mae_vit_base_7class.yaml \
  --dataset-root /path/to/dataset_root \
  MODEL.WEIGHTS /path/to/local/model.pth
```

## Image inference

```bash
python inference.py \
  --config-file configs/periapical/mask2former_iglc_mae_vit_base_7class.yaml \
  --input "/path/to/images/*.jpg" \
  --output outputs/inference \
  --weights /path/to/local/model.pth \
  --save-masks
```

The `--save-masks` flag writes argmax semantic masks as PNG files in addition to visualizations.

## Adaptive semi-supervised training

After obtaining a supervised initialization checkpoint, run teacher-student training with labeled and unlabeled data:

```bash
python semi_supervised/train_adaptive_teacher_student.py \
  --num-gpus 1 \
  --config-file configs/periapical/mask2former_adaptive_teacher_student.yaml \
  --dataset-root /path/to/labeled_dataset_root \
  --unlabeled-root /path/to/unlabeled_images \
  MODEL.WEIGHTS /path/to/local/supervised_checkpoint.pth \
  OUTPUT_DIR outputs/periapical_adaptive_teacher_student
```

The default semi-supervised objective uses a `0.85:0.15` supervised-to-pseudo loss ratio, EMA decay `0.999`, confidence center `0.8`, and temperature `0.10`. These values can be changed with config overrides, for example:

```bash
SEMI_SUPERVISED.CONFIDENCE_CENTER 0.9 SEMI_SUPERVISED.TEMPERATURE 0.10
```

## Checkpoints

No model weights are committed. Put local checkpoints under `CHECKPOINTS/` or pass them explicitly through `MODEL.VITMAE.CHECKPOINT_PATH`, `MODEL.WEIGHTS`, or `--weights`.
