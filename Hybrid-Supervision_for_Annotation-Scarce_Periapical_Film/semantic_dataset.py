import os
import random

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


PERIAPICAL_CLASSES = [
    "background",
    "tooth",
    "pulp",
    "caries",
    "fillings",
    "root_canal_fillings",
    "periapical_lesion",
    "crown",
]

PERIAPICAL_COLORS = [
    [0, 0, 0],
    [135, 206, 250],
    [143, 0, 255],
    [50, 255, 50],
    [140, 140, 140],
    [255, 179, 71],
    [0, 71, 171],
    [227, 66, 52],
]


def _read_classes(root):
    classes_file = os.path.join(root, "classes.txt")
    if PathManager.exists(classes_file):
        with PathManager.open(classes_file, "r") as handle:
            return [line.strip() for line in handle if line.strip()]
    return PERIAPICAL_CLASSES


def get_semantic_dataset(root, split):
    image_dir = os.path.join(root, "images", split)
    seg_dir = os.path.join(root, "annotations", split)

    image_files = [
        name
        for name in PathManager.ls(image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    dataset_dicts = []
    for image_file in sorted(image_files):
        image_path = os.path.join(image_dir, image_file)
        seg_filename = os.path.splitext(image_file)[0] + ".png"
        seg_path = os.path.join(seg_dir, seg_filename)

        if not PathManager.exists(seg_path):
            raise FileNotFoundError(f"Segmentation mask not found: {seg_path}")

        dataset_dicts.append(
            {
                "file_name": image_path,
                "sem_seg_file_name": seg_path,
            }
        )

    return dataset_dicts


def register_periapical_metadata(name, classes=None, colors=None):
    MetadataCatalog.get(name).set(
        stuff_classes=classes or PERIAPICAL_CLASSES,
        stuff_colors=colors or PERIAPICAL_COLORS,
        ignore_label=255,
        evaluator_type="sem_seg",
    )


def register_semantic_dataset(name, root, split):
    classes = _read_classes(root)

    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda root=root, split=split: get_semantic_dataset(root, split))

    register_periapical_metadata(name, classes=classes)


def verify_dataset(dataset_name, num_samples=3):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Number of images: {len(dataset_dicts)}")
    print(f"Classes: {metadata.stuff_classes}")

    for record in random.sample(dataset_dicts, min(num_samples, len(dataset_dicts))):
        print("Image:", record["file_name"])
        print("Segmentation:", record["sem_seg_file_name"])
        if not os.path.exists(record["file_name"]):
            raise FileNotFoundError(record["file_name"])
        if not os.path.exists(record["sem_seg_file_name"]):
            raise FileNotFoundError(record["sem_seg_file_name"])
