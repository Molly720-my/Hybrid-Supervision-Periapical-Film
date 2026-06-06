import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "demo"))

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


def get_parser():
    parser = argparse.ArgumentParser(description="Run periapical Mask2Former inference on images.")
    parser.add_argument(
        "--config-file",
        default="configs/periapical/mask2former_iglc_mae_vit_base_7class.yaml",
        metavar="FILE",
        help="Path to the Mask2Former config file.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input image paths or one glob pattern, for example images/*.jpg.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory or file path for visualization outputs.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained segmentation checkpoint.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root. If provided, classes.txt is used when available.",
    )
    parser.add_argument("--dataset-name", default="my_dataset_val2")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save argmax semantic label masks as PNG files.",
    )
    parser.add_argument(
        "--mask-output",
        default=None,
        help="Directory for semantic mask PNGs. Defaults to <output>/masks.",
    )
    parser.add_argument(
        "--opts",
        help="Additional config options in KEY VALUE format.",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.DATASETS.TEST = (args.dataset_name,)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.freeze()
    return cfg


def register_metadata(args):
    if args.dataset_root:
        register_semantic_dataset(args.dataset_name, args.dataset_root, args.split)
    else:
        register_periapical_metadata(args.dataset_name)


def expand_inputs(inputs):
    if len(inputs) == 1:
        expanded = glob.glob(os.path.expanduser(inputs[0]))
        if expanded:
            return sorted(expanded)
    return inputs


def output_path_for(input_path, output_arg, num_inputs):
    if os.path.isdir(output_arg) or num_inputs > 1:
        os.makedirs(output_arg, exist_ok=True)
        return os.path.join(output_arg, os.path.basename(input_path))

    os.makedirs(os.path.dirname(output_arg) or ".", exist_ok=True)
    return output_arg


def save_semantic_mask(predictions, image_path, output_dir):
    if "sem_seg" not in predictions:
        return

    os.makedirs(output_dir, exist_ok=True)
    sem_seg = predictions["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.uint8)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}.png"), sem_seg)


def main():
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: %s", args)

    register_metadata(args)
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    input_paths = expand_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError("No input images found.")

    mask_output = args.mask_output
    if args.save_masks and mask_output is None:
        mask_output = os.path.join(args.output, "masks") if os.path.isdir(args.output) else "masks"

    for image_path in tqdm.tqdm(input_paths):
        image = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(image)
        logger.info("%s: %.2fs", image_path, time.time() - start_time)

        visualized_output.save(output_path_for(image_path, args.output, len(input_paths)))
        if args.save_masks:
            save_semantic_mask(predictions, image_path, mask_output)


if __name__ == "__main__":
    main()
