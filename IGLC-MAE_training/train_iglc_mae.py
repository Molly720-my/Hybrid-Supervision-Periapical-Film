# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for IGLC-MAE pre-training.

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted(
            [
                os.path.join(image_dir, name)
                for name in os.listdir(image_dir)
                if name.lower().endswith(IMG_EXTENSIONS)
            ]
        )
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
        return image, torch.tensor(0)


def get_args_parser():
    parser = argparse.ArgumentParser("IGLC-MAE pre-training")
    parser.add_argument("--data_path", required=True, type=str, help="Directory containing unlabeled training images.")
    parser.add_argument("--output_dir", default="outputs/iglc_mae_pretrain", help="Directory for checkpoints and logs.")
    parser.add_argument("--log_dir", default=None, help="TensorBoard log directory. Defaults to output_dir.")

    parser.add_argument("--model", default="iglc_mae_vit_base_patch16", type=str, metavar="MODEL")
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument("--masking_w_var", default=0.15, type=float)
    parser.add_argument("--masking_w_grad", default=0.35, type=float)
    parser.add_argument("--masking_w_contrast", default=0.50, type=float)
    parser.add_argument("--masking_temp", default=1.0, type=float)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR")
    parser.add_argument("--blr", type=float, default=1e-3, metavar="LR")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="Resume from a full pre-training checkpoint.")
    parser.add_argument("--pretrained", default="", help="Optional compatible MAE/IGLC-MAE checkpoint for initialization.")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--save_freq", default=50, type=int)

    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def build_optimizer(model, weight_decay, lr):
    import timm.optim.optim_factory as optim_factory

    if hasattr(optim_factory, "param_groups_weight_decay"):
        param_groups = optim_factory.param_groups_weight_decay(model, weight_decay)
    else:
        param_groups = optim_factory.add_weight_decay(model, weight_decay)
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def load_pretrained_weights(model, checkpoint_path):
    if not checkpoint_path:
        return
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        state_dict = checkpoint

    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value

    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in cleaned.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded initialization checkpoint: {checkpoint_path}. "
        f"Matched {len(compatible)} tensors; missing={len(missing)}, unexpected={len(unexpected)}."
    )


def main(args):
    import models_iglc_mae
    from engine_pretrain import train_one_epoch

    misc.init_distributed_mode(args)

    if args.log_dir is None:
        args.log_dir = args.output_dir

    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(str(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset_train = FlatImageDataset(args.data_path, transform=transform_train)
    print(f"Loaded {len(dataset_train)} images from {args.data_path}")

    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
        global_rank = misc.get_rank()
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        global_rank = 0

    log_writer = None
    if global_rank == 0 and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        if SummaryWriter is not None:
            log_writer = SummaryWriter(log_dir=args.log_dir)
        else:
            print("TensorBoard is not installed; continuing without TensorBoard logging.")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.model not in models_iglc_mae.__dict__:
        available = sorted(name for name in models_iglc_mae.__dict__ if name.startswith("iglc_mae"))
        raise ValueError(f"Unknown model '{args.model}'. Available models: {available}")

    model = models_iglc_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        masking_w_var=args.masking_w_var,
        masking_w_grad=args.masking_w_grad,
        masking_w_contrast=args.masking_w_contrast,
        masking_temp=args.masking_temp,
    )
    load_pretrained_weights(model, args.pretrained)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {n_parameters / 1e6:.2f}M")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    args.lr = float(args.lr)
    print(f"Effective batch size: {eff_batch_size}")
    print(f"Learning rate: {args.lr:.2e}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    optimizer = build_optimizer(model_without_ddp, args.weight_decay, args.lr)
    loss_scaler = NativeScaler()

    if args.resume:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and misc.is_main_process():
            if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
