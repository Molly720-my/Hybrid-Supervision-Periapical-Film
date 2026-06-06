import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


def _assert_strides_are_log2_contiguous(strides):
    for stride, next_stride in zip(strides, strides[1:]):
        if next_stride != 2 * stride:
            raise ValueError(f"Feature strides must be log2-contiguous, got {strides}.")


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "teacher", "student"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _load_vit_checkpoint(vit, checkpoint_path):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _clean_state_dict(_extract_checkpoint_state(checkpoint))
    vit_state = vit.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in vit_state and vit_state[key].shape == value.shape
    }
    missing, unexpected = vit.load_state_dict(compatible, strict=False)
    print(
        f"Loaded MAE backbone weights from {checkpoint_path}. "
        f"Matched {len(compatible)} tensors; missing={len(missing)}, unexpected={len(unexpected)}."
    )


class TimmMAEWrapper(Backbone):
    def __init__(self, model_name, pretrained=False, checkpoint_path="", out_feature="last_feat"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        _load_vit_checkpoint(self.vit, checkpoint_path)

        self.patch_size = self._get_patch_size()
        self.embed_dim = self._get_embed_dim()
        self._out_feature = out_feature
        self._out_feature_channels = {out_feature: self.embed_dim}
        self._out_feature_strides = {out_feature: self.patch_size}
        self._out_features = [out_feature]

        if hasattr(self.vit, "pos_embed") and self.vit.pos_embed is not None:
            self.register_buffer("original_pos_embed", self.vit.pos_embed.detach().clone())
            self.vit.pos_embed = None

        if hasattr(self.vit.patch_embed, "strict_img_size"):
            self.vit.patch_embed.strict_img_size = False

    def _get_patch_size(self):
        if hasattr(self.vit.patch_embed, "patch_size"):
            patch_size = self.vit.patch_embed.patch_size
            return patch_size[0] if isinstance(patch_size, tuple) else patch_size
        if hasattr(self.vit.patch_embed.proj, "kernel_size"):
            return self.vit.patch_embed.proj.kernel_size[0]
        raise ValueError("Cannot infer patch size from the MAE backbone.")

    def _get_embed_dim(self):
        if hasattr(self.vit, "embed_dim"):
            return self.vit.embed_dim
        if len(self.vit.blocks) > 0:
            return self.vit.blocks[0].mlp.fc1.in_features
        raise ValueError("Cannot infer embedding dimension from the MAE backbone.")

    def _interpolate_pos_encoding(self, x, height_patches, width_patches):
        if not hasattr(self, "original_pos_embed"):
            return 0

        cls_pos = self.original_pos_embed[:, :1]
        patch_pos = self.original_pos_embed[:, 1:]
        original_size = int(math.sqrt(patch_pos.shape[1]))
        patch_pos = patch_pos.reshape(1, original_size, original_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(height_patches, width_patches),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height_patches * width_patches, -1)
        return torch.cat([cls_pos, patch_pos], dim=1).to(x.device)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        pad_width = (self.patch_size - width % self.patch_size) % self.patch_size
        pad_height = (self.patch_size - height % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_width, 0, pad_height))

        height_patches = (height + pad_height) // self.patch_size
        width_patches = (width + pad_width) // self.patch_size

        x = self.vit.patch_embed(x)
        if hasattr(self.vit, "cls_token"):
            cls_token = self.vit.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        pos_embed = self._interpolate_pos_encoding(x, height_patches, width_patches)
        if isinstance(pos_embed, torch.Tensor):
            x = x + pos_embed

        x = self.vit.pos_drop(x)
        for block in self.vit.blocks:
            x = block(x)
        x = self.vit.norm(x)

        if hasattr(self.vit, "cls_token"):
            x = x[:, 1:]

        x = x.permute(0, 2, 1).reshape(batch_size, -1, height_patches, width_patches)
        return {self._out_feature: x}

    def output_shape(self):
        return {
            self._out_feature: ShapeSpec(
                channels=self.embed_dim,
                stride=self.patch_size,
            )
        }


class SimpleFeaturePyramid(Backbone):
    def __init__(
        self,
        vit,
        in_feature="last_feat",
        out_features=("res2", "res3", "res4", "res5"),
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="GN",
    ):
        super().__init__()
        self.vit = vit
        self.in_feature = in_feature
        self._out_features = list(out_features)

        input_shape = vit.output_shape()[in_feature]
        strides = [int(input_shape.stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shape.channels
        self.stages = nn.ModuleList()
        for scale in scale_factors:
            layers = []
            if scale == 4.0:
                layers.extend(
                    [
                        nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                        get_norm(norm, dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    ]
                )
                out_dim = dim // 4
            elif scale == 2.0:
                layers.append(nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2))
                out_dim = dim // 2
            elif scale == 1.0:
                layers.append(nn.Identity())
                out_dim = dim
            elif scale == 0.5:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = dim
            else:
                raise ValueError(f"Unsupported scale factor: {scale}")

            layers.extend(
                [
                    Conv2d(out_dim, out_channels, 1, norm=get_norm(norm, out_channels)),
                    Conv2d(out_channels, out_channels, 3, padding=1, norm=get_norm(norm, out_channels)),
                ]
            )
            self.stages.append(nn.Sequential(*layers))

        self._out_feature_strides = {
            name: stride for name, stride in zip(self._out_features, strides)
        }
        self._out_feature_channels = {
            name: out_channels for name in self._out_features
        }

    def forward(self, x):
        features = self.vit(x)[self.in_feature]
        outputs = [stage(features) for stage in self.stages]
        return {name: output for name, output in zip(self._out_features, outputs)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
class VITMAE(SimpleFeaturePyramid):
    def __init__(self, cfg, input_shape):
        vit_cfg = cfg.MODEL.VITMAE
        vit = TimmMAEWrapper(
            model_name=vit_cfg.MODEL_NAME,
            pretrained=vit_cfg.PRETRAINED,
            checkpoint_path=vit_cfg.CHECKPOINT_PATH,
            out_feature=vit_cfg.IN_FEATURE,
        )
        super().__init__(
            vit=vit,
            in_feature=vit_cfg.IN_FEATURE,
            out_features=vit_cfg.OUT_FEATURES,
            out_channels=vit_cfg.OUT_CHANNELS,
            scale_factors=vit_cfg.SCALE_FACTORS,
            norm=vit_cfg.NORM,
        )

    @property
    def size_divisibility(self):
        return self.vit.patch_size
