import argparse
import glob
import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from semantic_dataset import register_periapical_metadata, register_semantic_dataset


WINDOW_NAME = "periapical mask2former demo"


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.DATASETS.TEST = (args.dataset_name,)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Periapical Mask2Former image inference")
    parser.add_argument(
        "--config-file",
        default="configs/periapical/mask2former_iglc_mae_vit_base_7class.yaml",
        metavar="FILE",
        help="Path to config file.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image paths or a single glob pattern such as 'images/*.jpg'.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory or file path for visualization output.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a local model checkpoint. No weights are distributed with this repository.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root for registering class metadata from classes.txt.",
    )
    parser.add_argument("--dataset-name", default="my_dataset_val2")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save argmax semantic masks as PNG files.",
    )
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Directory for semantic mask PNGs. Defaults to <output>/masks when --output is a directory.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using KEY VALUE pairs.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def expand_inputs(inputs):
    if len(inputs) == 1:
        expanded = glob.glob(os.path.expanduser(inputs[0]))
        if expanded:
            return sorted(expanded)
    return inputs


def save_semantic_mask(predictions, path, output_dir):
    if "sem_seg" not in predictions:
        return
    os.makedirs(output_dir, exist_ok=True)
    sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.uint8)
    base_name = os.path.splitext(os.path.basename(path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}.png"), sem_seg)


def register_metadata(args):
    if args.dataset_root:
        register_semantic_dataset(args.dataset_name, args.dataset_root, args.split)
    else:
        register_periapical_metadata(args.dataset_name)


if __name__ == "__main__":
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    register_metadata(args)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    input_paths = expand_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError("No input images found.")

    mask_output = args.mask_output
    if args.save_masks and mask_output is None:
        mask_output = os.path.join(args.output, "masks") if os.path.isdir(args.output) else "masks"

    for path in tqdm.tqdm(input_paths, disable=False):
        image = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(image)
        logger.info("%s: finished in %.2fs", path, time.time() - start_time)

        if os.path.isdir(args.output):
            out_filename = os.path.join(args.output, os.path.basename(path))
        else:
            if len(input_paths) != 1:
                raise ValueError("--output must be a directory when using multiple inputs.")
            out_filename = args.output

        os.makedirs(os.path.dirname(out_filename) or ".", exist_ok=True)
        visualized_output.save(out_filename)

        if args.save_masks:
            save_semantic_mask(predictions, path, mask_output)
