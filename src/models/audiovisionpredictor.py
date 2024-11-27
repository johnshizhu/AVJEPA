import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_xy, get_3d_sincos_pos_embed
from src.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)
from src.masks.utils import apply_masks, target_apply_masks

from logging import getLogger
logger = getLogger()

class AudioVisionTransformerPredictor(nn.Module):
    """ Vision Transformer """
    def __init__(
        self,
        img_size=224,
        a_size=(128,192),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        use_mask_tokens=False,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        **kwargs
    ):
        super().__init__()
        # Map input to predictor dimension
        self.predictor_embed_v = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.predictor_embed_a = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for i in range(num_mask_tokens)
            ])

        # Determine positional embedding
        self.input_size = img_size
        self.patch_size = patch_size
        # --
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1

        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        a_height = a_size[0] // self.patch_size
        a_width  = a_size[1] // self.patch_size

        self.num_patches = num_patches = (
            (num_frames // tubelet_size)
            * (img_size // patch_size)
            * (img_size // patch_size)
        )

        num_patches_a = a_height * a_width

        # Position embedding
        self.uniform_power = uniform_power
        self.predictor_pos_embed_v = None
        self.predictor_pos_embed_a = None
        self.predictor_pos_embed_v = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False)
        self.predictor_pos_embed_a = nn.Parameter(
            torch.zeros(1, num_patches_a, predictor_embed_dim),
            requires_grad=False)

        # Attention Blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                grid_size=grid_size,
                grid_depth=grid_depth,
                norm_layer=norm_layer)
            for i in range(depth)])

        # Normalize & project back to input dimension
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # ------ initialize weights
        if self.predictor_pos_embed_v is not None:
            self._init_video_pos_embed(self.predictor_pos_embed_v.data)  # sincos pos-embed
        if self.predictor_pos_embed_a is not None:
            self._init_audio_pos_embed(self.predictor_pos_embed_a.data)
        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
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

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):

        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # Sample diffusion time step
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * torch.randn(x.shape, device=x.device)
        return x

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt, mask_index=1):
        """
        :param ctxt: context tokens
        :param tgt: target tokens
        :param masks_ctxt: indices of context tokens in input
        :params masks_tgt: indices of target tokens in input
        """
        # -- video and audio tokens extraction
        assert (masks_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        ctxt_v, ctxt_a = ctxt
        tgt_v,  tgt_a  = tgt

        logger.info(f'mask_tokens status: {self.mask_tokens}')

        logger.info(f'ctxt_v.shape: {ctxt_v.shape}')
        logger.info(f'ctxt_a.shape: {ctxt_a.shape}')
        logger.info(f'tgt_v.shape: {tgt_v.shape}')
        logger.info(f'tgt_a.shape: {tgt_a.shape}')

        masks_ctxt_v, masks_ctxt_a = masks_ctxt[0], masks_ctxt[1]
        masks_tgt_v, masks_tgt_a = masks_tgt[0], masks_tgt[1]

        for i, m in enumerate(masks_ctxt_v):
            logger.info(f'masks_ctxt_v[{i}] shape: {m.shape}')
        for i, m in enumerate(masks_ctxt_a):
            logger.info(f'masks_ctxt_a[{i}] shape: {m.shape}')
        for i, m in enumerate(masks_tgt_v):
            logger.info(f'masks_tgt_v[{i}] shape: {m.shape}')
        for i, m in enumerate(masks_tgt_a):
            logger.info(f'masks_tgt_a[{i}] shape: {m.shape}')

        if not isinstance(masks_ctxt_v, list):
            masks_ctxt_v = [masks_ctxt_v]
        if not isinstance(masks_ctxt_a, list):
            masks_ctxt_a = [masks_ctxt_a]
        if not isinstance(masks_ctxt, list):
            masks_tgt_v = [masks_tgt_v]
        if not isinstance(masks_tgt_a, list):
            masks_tgt_a = [masks_tgt_a]

        # Batch Size
        # B = len(ctxt_v) // len(masks_ctxt_v)
        B = 24

        # Map context tokens to predictor dimensions
        x_v = self.predictor_embed_v(ctxt_v)
        x_a = self.predictor_embed_a(ctxt_a)
        logger.info(f"x_v: {x_v.shape}")
        logger.info(f"x_a: {x_a.shape}")
        _, N_ctxt, D = x_v.shape

        # Add positional embedding to ctxt tokens
        if self.predictor_pos_embed_v is not None:
            ctxt_pos_embed_v = self.predictor_pos_embed_v.repeat(B, 1, 1)
            ctxt_pos_embed_a = self.predictor_pos_embed_a.repeat(B, 1, 1)
            x_v += target_apply_masks(ctxt_pos_embed_v, masks_ctxt_v)
            x_a += target_apply_masks(ctxt_pos_embed_a, masks_ctxt_a)

        logger.info(f"x_v: {x_v.shape}")
        logger.info(f"x_a: {x_a.shape}")
        logger.info(f'type x_v: {type(x_v)}')
        logger.info(f'type x_a: {type(x_a)}')

        # Map target tokens to predictor dimensions & add noise (fwd diffusion)
        if self.mask_tokens is None:
            pred_tokens_v = self.predictor_embed_v(tgt_v)
            pred_tokens_a = self.predictor_embed_a(tgt_a)
            pred_tokens_v = self.diffusion(pred_tokens_v)
            pred_tokens_a = self.diffusion(pred_tokens_a)
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
            pred_tokens_v = target_apply_masks(pred_tokens, masks_tgt_v[0])
            pred_tokens_a = target_apply_masks(pred_tokens, masks_tgt_a[0])
            logger.info(f'pred_tokens: {pred_tokens.shape}')
            logger.info(f'pred_tokens_v: {pred_tokens_v.shape}')
            logger.info(f'pred_tokens_a: {pred_tokens_a.shape}')


        # Add positional embedding to target tokens
        if self.predictor_pos_embed_v is not None:
            pos_embs = self.predictor_pos_embed_v.repeat(B, 1, 1)
            pos_embs = target_apply_masks(pos_embs, masks_tgt_v[0])
            # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt_v))
            logger.info(f'left: {pred_tokens_v.shape}')
            logger.info(f'right: {pos_embs.shape}')
            pred_tokens_v += pos_embs

        if self.predictor_pos_embed_a is not None:
            pos_embs = self.predictor_pos_embed_a.repeat(B, 1, 1)
            pos_embs = target_apply_masks(pos_embs, masks_tgt_a[0])
            # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_ctxt_a))
            pred_tokens_a += pos_embs

        # Concatenate context & target tokens

        logger.info(f'x_v: {x_v.shape}')
        x_v = x_v.repeat(len(masks_tgt_v), 1, 1)
        logger.info(f'x_v: {x_v.shape}')
        logger.info(f'pred_tokens_v: {pred_tokens_v.shape}')
        x_v = torch.cat([x_v, pred_tokens_v], dim=1)
        logger.info(f'x_v: {x_v.shape}')

        logger.info(f'x_a: {x_a.shape}')
        logger.info(f'pred_tokens_a: {pred_tokens_a.shape}')
        x_a = torch.cat([x_a, pred_tokens_a], dim=1)
        logger.info(f'x_a: {x_a.shape}')

        


        # FIXME: this implementation currently assumes masks_ctxt and masks_tgt
        # are alligned 1:1 (ok with MultiMask wrapper on predictor but
        # otherwise will break)
        # POLO: TODO  do we need mask here
        x = torch.cat([x_v, x_a], dim=1)
        masks = None
        # masks_ctxt = torch.cat(masks_ctxt, dim=0)
        # masks_tgt = torch.cat(masks_tgt, dim=0)
        # masks = torch.cat([masks_ctxt, masks_tgt], dim=1)

        # Fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, mask=masks)
        x = self.predictor_norm(x)

        # Return output corresponding to target tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x


def vit_avpredictor(**kwargs):
    model = AudioVisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
