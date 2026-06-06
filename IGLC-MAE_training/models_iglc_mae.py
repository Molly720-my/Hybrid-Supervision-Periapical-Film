# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified for IGLC-MAE pre-training.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed

from util.pos_embed import get_2d_sincos_pos_embed


class IGLCAdaptiveMAEViT(nn.Module):
    """
    Masked Autoencoder with Intensity, Gradient, and Local Contrast adaptive masking.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        masking_w_var=0.15,
        masking_w_grad=0.35,
        masking_w_contrast=0.50,
        masking_temp=1.0,
    ):
        super().__init__()

        self.masking_w_var = masking_w_var
        self.masking_w_grad = masking_w_grad
        self.masking_w_contrast = masking_w_contrast
        self.masking_temp = masking_temp

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("sobel_x_kernel", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y_kernel", sobel_y.view(1, 1, 3, 3), persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        weight = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        channels = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], channels, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * channels))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(self.num_patches**0.5)
        assert h * w == self.num_patches
        assert x.shape[-1] % (p**2) == 0
        channels = x.shape[-1] // (p**2)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * p, w * p))
        return imgs

    def calculate_patch_stats(self, imgs):
        batch_size, channels, height, width = imgs.shape
        p = self.patch_embed.patch_size[0]
        patches_h = height // p
        patches_w = width // p
        num_patches = patches_h * patches_w

        imgs_gray = imgs.mean(dim=1, keepdim=True) if channels > 1 else imgs

        unfolded_gray = F.unfold(imgs_gray, kernel_size=p, stride=p).reshape(
            batch_size,
            1,
            p * p,
            num_patches,
        )
        patch_var = torch.var(unfolded_gray, dim=2, unbiased=False).squeeze(1)
        patch_mean = torch.mean(unfolded_gray, dim=2).squeeze(1)

        grad_x = F.conv2d(imgs_gray, self.sobel_x_kernel.to(imgs_gray.dtype), padding=1)
        grad_y = F.conv2d(imgs_gray, self.sobel_y_kernel.to(imgs_gray.dtype), padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12)
        unfolded_grad = F.unfold(grad_mag, kernel_size=p, stride=p).reshape(
            batch_size,
            1,
            p * p,
            num_patches,
        )
        patch_grad = torch.mean(unfolded_grad, dim=2).squeeze(1)

        patch_mean_map = patch_mean.reshape(batch_size, 1, patches_h, patches_w)
        patch_mean_map = F.pad(patch_mean_map, pad=(1, 1, 1, 1), mode="replicate")
        neighborhood_mean = F.avg_pool2d(patch_mean_map, kernel_size=3, stride=1, padding=0)
        neighborhood_mean = neighborhood_mean.reshape(batch_size, num_patches)
        patch_contrast = torch.abs(patch_mean - neighborhood_mean)

        return patch_var, patch_grad, patch_contrast

    @staticmethod
    def _normalize_stat(stat):
        return (stat - stat.mean(dim=1, keepdim=True)) / (stat.std(dim=1, keepdim=True) + 1e-6)

    def adaptive_masking(self, x, imgs, mask_ratio):
        batch_size, num_patches, dim = x.shape
        len_keep = int(num_patches * (1 - mask_ratio))
        num_mask = num_patches - len_keep

        patch_var, patch_grad, patch_contrast = self.calculate_patch_stats(imgs)
        score = (
            self.masking_w_var * self._normalize_stat(patch_var)
            + self.masking_w_grad * self._normalize_stat(patch_grad)
            + self.masking_w_contrast * self._normalize_stat(patch_contrast)
        )
        weights = F.softmax(score / self.masking_temp, dim=1)
        ids_mask = torch.multinomial(weights, num_samples=num_mask, replacement=False)

        mask = torch.zeros(batch_size, num_patches, device=x.device, dtype=torch.long)
        mask.scatter_(dim=1, index=ids_mask, value=1)

        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs, mask_ratio):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.adaptive_masking(x, imgs, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        num_patches_total = ids_restore.shape[1]
        num_keep = x.shape[1] - 1
        num_masked = num_patches_total - num_keep

        mask_tokens = self.mask_token.repeat(x.shape[0], num_masked, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed

        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def iglc_mae_vit_base_patch16_dec512d8b(**kwargs):
    return IGLCAdaptiveMAEViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def iglc_mae_vit_large_patch16_dec512d8b(**kwargs):
    return IGLCAdaptiveMAEViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def iglc_mae_vit_huge_patch14_dec512d8b(**kwargs):
    return IGLCAdaptiveMAEViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


iglc_mae_vit_base_patch16 = iglc_mae_vit_base_patch16_dec512d8b
iglc_mae_vit_large_patch16 = iglc_mae_vit_large_patch16_dec512d8b
iglc_mae_vit_huge_patch14 = iglc_mae_vit_huge_patch14_dec512d8b
