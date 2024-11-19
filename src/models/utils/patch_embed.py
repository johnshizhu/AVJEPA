# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from logging import getLogger
logger = getLogger()


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class AudioVisionPatchEmbed3D(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(
        self,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        logger.info(f'in_chans: {in_chans}')
        logger.info(f'embed_dim: {embed_dim}')
        logger.info(f'kernel_size: {(tubelet_size, patch_size, patch_size)}')
        logger.info(f'stride: {(tubelet_size, patch_size, patch_size)}')

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.audio_proj = nn.Conv2d(
            in_channels = 1,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size)
        )

    def forward(self, x, y, **kwargs):
        B, C, T, H, W = x.shape
        #logger.info(f'x shape is: {x.shape}')
        #logger.info(f'y shape is: {y.shape}')
        x = self.proj(x).flatten(2).transpose(1, 2)
        y = self.audio_proj(y).flatten(2).transpose(1, 2)
        # logger.info(f'x embed forward output type:{type(x)}')
        # logger.info(f'x embed forward output shape:{x.shape}')
        # logger.info(f'y embed forward output type:{type(y)}')
        # logger.info(f'y embed forward output shape:{y.shape}')
        out = torch.cat([x, y], dim=1)
        #logger.info(f'out shape is: {out.shape}')
        return out
