# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D, AudioVisionPatchEmbed3D
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_xy, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_
from src.masks.utils import apply_masks

from logging import getLogger

logger = getLogger()


class AudioVisionTransformer(nn.Module):
    """ Audio Vision Transformer """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        # Tokenize pixels with convolution
        if self.is_video:
            self.patch_embed = AudioVisionPatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size // patch_size)
                * (img_size // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim)
            self.num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )

        # Position embedding
        self.uniform_power = uniform_power
        self.video_pos_embed = None
        logger.info(f'num_patches is: {self.num_patches}')
        self.video_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=False)
        self.audio_pos_embed = nn.Parameter(
            torch.zeros(1, 96, embed_dim), # based on current calculation method, always 96 tokens
            requires_grad=False)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # ------ initialize weights
        if self.video_pos_embed is not None:
            self._init_video_pos_embed(self.video_pos_embed.data)  # sincos pos-embed
            logger.info("video positional embedding intialized")
        if self.audio_pos_embed is not None:
            self._init_audio_pos_embed(self.audio_pos_embed.data)
            logger.info("audio positional embedding initialized")
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_video_pos_embed(self, video_pos_embed):
        embed_dim = video_pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        video_pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_audio_pos_embed(self, audio_pos_embed):
        embed_dim = audio_pos_embed.size(-1)
        # based on current implementation, audiospectrogram is always 128 by 192
        grid_h = 128 // self.patch_size
        grid_w = 192 // self.patch_size
        
        sincos = get_2d_sincos_pos_embed_xy(
            embed_dim,
            grid_h,
            grid_w,
            cls_token=False
        )
        
        audio_pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {}

    def forward(self, x, y, masks=None):
        """
        :param x: input image/video
        :param masks: indices of patch tokens to mask (remove)
        """
        v_masks = None
        a_masks = None
        if masks is not None:
            v_masks = masks[0]
            a_masks = masks[1]
        if v_masks is not None and not isinstance(v_masks, list):
            v_masks = [v_masks]
        if a_masks is not None and not isinstance(a_masks, list):
            a_masks = [a_masks]

        logger.info(f'Input x shape is: {x.shape}')
        logger.info(f'Input y shape is: {y.shape}')

        # Tokenize input
        video_pos_embed = self.video_pos_embed
        audio_pos_embed = self.audio_pos_embed
        if video_pos_embed is not None:
            video_pos_embed = self.interpolate_pos_encoding(x, video_pos_embed)

        video_tokens, audio_tokens = self.patch_embed(x, y)
        video_tokens += video_pos_embed
        audio_tokens += audio_pos_embed

        logger.info(f'video_tokens shape is: {video_tokens.shape}')
        logger.info(f'audio_tokens shape is: {audio_tokens.shape}')

        # Mask away unwanted tokens (if masks provided)
        if masks is not None:
            video_tokens = apply_masks(video_tokens, v_masks)
            audio_tokens = apply_masks(audio_tokens, a_masks)
            #masks = torch.cat(masks, dim=0)

        logger.info(f'post masking video_tokens shape is: {video_tokens.shape}')
        logger.info(f'post masking audio_tokens shape is: {audio_tokens.shape}')

        x = torch.cat([video_tokens, audio_tokens], dim=1) # combine into multimodal input
        logger.info(f'x shape is: {x.shape}')

        # Fwd prop
        logger.info(f'starting forward prop...')
        logger.info(f'masks type is: {type(masks)}')
        if masks:
            logger.info(f'masks len is: {len(masks)}')
            logger.info(f'masks[0] shape is: {masks[0].shape}')
            logger.info(f'masks[1] shape is: {masks[1].shape}')
        outs = []
        logger.info(f'Starting x shape is: {x.shape}')
        for i, blk in enumerate(self.blocks):
            logger.info(f'block {i}')
            x = blk(x, mask=masks)
            logger.info(f'block {i} post x shape is: {x.shape}')
            if self.out_layers is not None and i in self.out_layers:
                outs.append(self.norm(x))

        if self.out_layers is not None:
            return outs

        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed


def vit_tiny(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_giant(patch_size=16, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_gigantic(patch_size=14, **kwargs):
    model = AudioVisionTransformer(
        patch_size=patch_size, embed_dim=1664, depth=48, num_heads=16, mpl_ratio=64/13,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model


VIT_EMBED_DIMS = {
    'vit_tiny': 192,
    'vit_small': 384,
    'vit_base': 768,
    'vit_large': 1024,
    'vit_huge': 1280,
    'vit_giant': 1408,
    'vit_gigantic': 1664,
}
