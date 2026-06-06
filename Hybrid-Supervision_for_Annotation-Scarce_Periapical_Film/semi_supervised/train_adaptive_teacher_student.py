import argparse
import copy
import glob
import json
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, PROJECT_ROOT)

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import create_ddp_model, default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import Instances
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from PIL import Image
from torch.nn import functional as F

from mask2former import add_maskformer2_config
from semantic_dataset import register_periapical_metadata, register_semantic_dataset
from train_net import Trainer


def add_adaptive_teacher_student_config(cfg):
    cfg.SEMI_SUPERVISED = CN()
    cfg.SEMI_SUPERVISED.EMA_DECAY = 0.999
    cfg.SEMI_SUPERVISED.SUPERVISED_LOSS_WEIGHT = 0.85
    cfg.SEMI_SUPERVISED.PSEUDO_LOSS_WEIGHT = 0.15
    cfg.SEMI_SUPERVISED.CONFIDENCE_CENTER = 0.8
    cfg.SEMI_SUPERVISED.TEMPERATURE = 0.10
    cfg.SEMI_SUPERVISED.MIN_PSEUDO_WEIGHT = 0.05
    cfg.SEMI_SUPERVISED.LOG_PERIOD = 20
    cfg.SEMI_SUPERVISED.SAVE_TEACHER = True


def get_parser():
    parser = default_argument_parser()
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Labeled dataset root with images/<split> and annotations/<split>.",
    )
    parser.add_argument(
        "--val-dataset-root",
        default=None,
        help="Optional validation dataset root. Defaults to --dataset-root.",
    )
    parser.add_argument(
        "--unlabeled-root",
        required=True,
        help="Unlabeled image root. The script checks images/<split>, <split>, then the root itself.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--unlabeled-split", default="train")
    parser.add_argument("--train-dataset-name", default="my_dataset_train2")
    parser.add_argument("--val-dataset-name", default="my_dataset_val2")
    parser.add_argument("--unlabeled-dataset-name", default="periapical_unlabeled_train")
    return parser


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_adaptive_teacher_student_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def register_datasets(args):
    val_root = args.val_dataset_root or args.dataset_root
    register_semantic_dataset(args.train_dataset_name, args.dataset_root, args.train_split)
    register_semantic_dataset(args.val_dataset_name, val_root, args.val_split)
    register_unlabeled_dataset(args.unlabeled_dataset_name, args.unlabeled_root, args.unlabeled_split)


def find_unlabeled_image_dir(root, split):
    candidates = [
        os.path.join(root, "images", split),
        os.path.join(root, split),
        root,
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find unlabeled image directory under {root}")


def register_unlabeled_dataset(name, root, split):
    image_dir = find_unlabeled_image_dir(root, split)
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pattern)))
    image_files = sorted(set(image_files))
    if not image_files:
        raise FileNotFoundError(f"No unlabeled images found in {image_dir}")

    def load_unlabeled_records():
        records = []
        for idx, file_name in enumerate(image_files):
            with Image.open(file_name) as image:
                width, height = image.size
            records.append(
                {
                    "file_name": file_name,
                    "image_id": idx,
                    "height": height,
                    "width": width,
                }
            )
        return records

    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, load_unlabeled_records)
    register_periapical_metadata(name)


class UnlabeledImageDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.size_divisibility = cfg.INPUT.SIZE_DIVISIBILITY
        self.augmentations = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            self.augmentations.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        if is_train:
            self.augmentations.append(T.RandomFlip())

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        aug_input, _ = T.apply_transform_gens(self.augmentations, aug_input)
        image = aug_input.image

        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()

        dataset_dict["image"] = image
        dataset_dict["height"] = image.shape[-2]
        dataset_dict["width"] = image.shape[-1]
        return dataset_dict


def build_unlabeled_train_loader(cfg, dataset_name):
    unlabeled_cfg = cfg.clone()
    unlabeled_cfg.defrost()
    unlabeled_cfg.DATASETS.TRAIN = (dataset_name,)
    unlabeled_cfg.freeze()
    mapper = UnlabeledImageDatasetMapper(unlabeled_cfg, is_train=True)
    return build_detection_train_loader(unlabeled_cfg, mapper=mapper)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def semantic_mask_to_instances(sem_seg, ignore_label):
    sem_seg = sem_seg.to(torch.long)
    height, width = sem_seg.shape
    classes = torch.unique(sem_seg)
    classes = classes[classes != ignore_label]

    instances = Instances((height, width))
    instances.gt_classes = classes.to(torch.int64)
    if len(classes) == 0:
        instances.gt_masks = torch.zeros((0, height, width), dtype=torch.bool)
    else:
        instances.gt_masks = torch.stack([sem_seg == class_id for class_id in classes])
    return instances


@torch.no_grad()
def initialize_teacher(student, teacher):
    teacher.load_state_dict(unwrap_model(student).state_dict(), strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def update_ema_teacher(student, teacher, decay):
    student_state = unwrap_model(student).state_dict()
    teacher_state = teacher.state_dict()
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key].detach()
        if torch.is_floating_point(teacher_value):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


@torch.no_grad()
def build_pseudo_batch(teacher, unlabeled_batch, cfg):
    ignore_label = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
    center = cfg.SEMI_SUPERVISED.CONFIDENCE_CENTER
    temperature = cfg.SEMI_SUPERVISED.TEMPERATURE
    min_weight = cfg.SEMI_SUPERVISED.MIN_PSEUDO_WEIGHT

    teacher.eval()
    outputs = teacher(unlabeled_batch)

    pseudo_batch = []
    pseudo_weights = []
    kept_pixel_ratios = []

    for record, output in zip(unlabeled_batch, outputs):
        image = record["image"]
        target_size = image.shape[-2:]
        logits = output["sem_seg"]
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        probs = torch.softmax(logits, dim=0)
        confidence, pseudo_mask = torch.max(probs, dim=0)
        adaptive_weight = torch.sigmoid((confidence - center) / temperature)
        keep_mask = adaptive_weight >= min_weight

        pseudo_mask = pseudo_mask.to(torch.long)
        pseudo_mask[~keep_mask] = ignore_label
        instances = semantic_mask_to_instances(pseudo_mask.cpu(), ignore_label)

        pseudo_record = copy.copy(record)
        pseudo_record["sem_seg"] = pseudo_mask.cpu()
        pseudo_record["instances"] = instances
        pseudo_record["pseudo_weight"] = adaptive_weight.mean().detach()
        pseudo_batch.append(pseudo_record)
        pseudo_weights.append(adaptive_weight.mean().detach())
        kept_pixel_ratios.append(keep_mask.float().mean().detach())

    return pseudo_batch, torch.stack(pseudo_weights).mean(), torch.stack(kept_pixel_ratios).mean()


def reduce_loss_dict(loss_dict):
    detached = {key: value.detach() for key, value in loss_dict.items()}
    return comm.reduce_dict(detached)


def write_metrics(output_dir, metrics):
    if not comm.is_main_process():
        return
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "semi_supervised_log.jsonl")
    with open(metrics_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def save_teacher_checkpoint(cfg, teacher, iteration):
    if not cfg.SEMI_SUPERVISED.SAVE_TEACHER or not comm.is_main_process():
        return
    path = os.path.join(cfg.OUTPUT_DIR, f"teacher_model_{iteration:07d}.pth")
    torch.save({"model": teacher.state_dict(), "iteration": iteration}, path)


def train(args, cfg):
    logger = logging.getLogger("mask2former.semi_supervised")
    device = torch.device(cfg.MODEL.DEVICE)

    student = Trainer.build_model(cfg)
    optimizer = Trainer.build_optimizer(cfg, student)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)
    checkpointer = DetectionCheckpointer(
        student,
        save_dir=cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    start_iter = checkpoint.get("iteration", -1) + 1 if args.resume else 0

    teacher = Trainer.build_model(cfg).to(device)
    initialize_teacher(student, teacher)

    student = create_ddp_model(student, broadcast_buffers=False)
    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        cfg.SOLVER.CHECKPOINT_PERIOD,
        max_iter=cfg.SOLVER.MAX_ITER,
    )

    labeled_loader = Trainer.build_train_loader(cfg)
    unlabeled_loader = build_unlabeled_train_loader(cfg, args.unlabeled_dataset_name)
    labeled_iter = cycle(labeled_loader)
    unlabeled_iter = cycle(unlabeled_loader)

    max_iter = cfg.SOLVER.MAX_ITER
    log_period = cfg.SEMI_SUPERVISED.LOG_PERIOD
    eval_period = cfg.TEST.EVAL_PERIOD

    logger.info("Starting adaptive teacher-student training from iteration %s", start_iter)
    for iteration in range(start_iter, max_iter):
        start_time = time.time()
        student.train()

        labeled_batch = next(labeled_iter)
        unlabeled_batch = next(unlabeled_iter)

        supervised_loss_dict = student(labeled_batch)
        supervised_loss = sum(supervised_loss_dict.values()) * cfg.SEMI_SUPERVISED.SUPERVISED_LOSS_WEIGHT

        pseudo_batch, pseudo_weight, kept_ratio = build_pseudo_batch(teacher, unlabeled_batch, cfg)
        pseudo_loss_dict = student(pseudo_batch)
        pseudo_loss = (
            sum(pseudo_loss_dict.values())
            * cfg.SEMI_SUPERVISED.PSEUDO_LOSS_WEIGHT
            * pseudo_weight.to(device)
        )

        total_loss = supervised_loss + pseudo_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        update_ema_teacher(student, teacher, cfg.SEMI_SUPERVISED.EMA_DECAY)

        if iteration % log_period == 0 or iteration == max_iter - 1:
            supervised_reduced = reduce_loss_dict(supervised_loss_dict)
            pseudo_reduced = reduce_loss_dict(pseudo_loss_dict)
            metrics = {
                "iteration": iteration,
                "lr": optimizer.param_groups[0]["lr"],
                "total_loss": float(total_loss.detach().cpu()),
                "supervised_loss": float(sum(supervised_reduced.values()).cpu()),
                "pseudo_loss": float(sum(pseudo_reduced.values()).cpu()),
                "pseudo_weight": float(pseudo_weight.cpu()),
                "kept_pixel_ratio": float(kept_ratio.cpu()),
                "time": time.time() - start_time,
            }
            write_metrics(cfg.OUTPUT_DIR, metrics)
            if comm.is_main_process():
                logger.info(
                    "iter=%d total=%.4f sup=%.4f pseudo=%.4f pseudo_w=%.3f keep=%.3f lr=%.6g",
                    iteration,
                    metrics["total_loss"],
                    metrics["supervised_loss"],
                    metrics["pseudo_loss"],
                    metrics["pseudo_weight"],
                    metrics["kept_pixel_ratio"],
                    metrics["lr"],
                )

        periodic_checkpointer.step(iteration)
        if iteration > start_iter and cfg.SOLVER.CHECKPOINT_PERIOD > 0:
            if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                save_teacher_checkpoint(cfg, teacher, iteration)

        if eval_period > 0 and iteration > start_iter and iteration % eval_period == 0:
            student.eval()
            Trainer.test(cfg, student)
            comm.synchronize()

    save_teacher_checkpoint(cfg, teacher, max_iter)
    return {"final_iteration": max_iter}


def main(args):
    register_datasets(args)
    cfg = setup(args)
    return train(args, cfg)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="auto",
        args=(args,),
    )
