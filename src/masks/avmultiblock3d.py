# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class AVMaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        super(AVMaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _AVMaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=m.get('spatial_scale'),
                temporal_pred_mask_scale=m.get('temporal_scale'),
                aspect_ratio=m.get('aspect_ratio'),
                npred=m.get('num_blocks'),
                max_context_frames_ratio=m.get('max_temporal_keep', 1.0),
                max_keep=m.get('max_keep', None),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):

        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred_v, collated_masks_enc_v = [], []
        collated_masks_pred_a, collated_masks_enc_a = [], []

        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc_v, masks_enc_a, masks_pred_v, masks_pred_a = mask_generator(batch_size)
            collated_masks_enc_v.append(masks_enc_v)
            collated_masks_pred_v.append(masks_pred_v)
            collated_masks_enc_a.append(masks_enc_a)
            collated_masks_pred_a.append(masks_pred_a)

        return collated_batch, collated_masks_enc_v, collated_masks_enc_a, collated_masks_pred_v, collated_masks_pred_a


class _AVMaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        a_size = (128, 192),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        super(_AVMaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.a_height = a_size[0] // spatial_patch_size
        self.a_width = a_size[1] // spatial_patch_size
        self.a_size = a_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(
        self,
        generator,
        temporal_scale,
        spatial_scale,
        aspect_ratio_scale
    ):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask_v(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        v_mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        v_mask[start:start+t, top:top+h, left:left+w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            v_mask[self.max_context_duration:, :, :] = 0

        # --
        return v_mask
    
    def _sample_block_mask_a(self, h=4, w=6):
        top = torch.randint(0, self.a_height - h + 1, (1,))
        left = torch.randint(0, self.a_width - w + 1, (1,))

        mask = torch.ones((self.a_height, self.a_width), dtype=torch.int32)
        mask[top:top+h, left:left+w] = 0

        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred_v, collated_masks_enc_v = [], []
        collated_masks_pred_a, collated_masks_enc_a = [], []
        #min_keep_enc_v = min_keep_pred_v = self.duration * self.height * self.width
        #min_keep_enc_a = min_keep_pred_a = self.a_height * self.a_width
        # for _ in range(batch_size)
        #for _ in range(self.npred): # number of masks per batch

        mask_e_v = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask_e_a = torch.ones((self.a_height, self.a_width), dtype=torch.int32)
        mask_e_v *= self._sample_block_mask_v(p_size)
        mask_e_a *= self._sample_block_mask_a()
        
        mask_e_v = mask_e_v.flatten()
        mask_e_a = mask_e_a.flatten()

        # predictor mask -- vector of INDICES where the mask is 0
        mask_p_v = torch.argwhere(mask_e_v == 0).squeeze()
        mask_p_a = torch.argwhere(mask_e_a == 0).squeeze()
        
        # encoder mask -- vector of INDICES where mask is 1
        mask_e_v = torch.nonzero(mask_e_v).squeeze()
        mask_e_a = torch.nonzero(mask_e_a).squeeze()

        # replicate along batch size
        mask_p_v = mask_p_v.unsqueeze(0).repeat(batch_size, 1)
        mask_p_a = mask_p_a.unsqueeze(0).repeat(batch_size, 1)
        mask_e_v = mask_e_v.unsqueeze(0).repeat(batch_size, 1)
        mask_e_a = mask_e_a.unsqueeze(0).repeat(batch_size, 1)

        #collated_masks_pred_v = torch.utils.data.default_collate(collated_masks_pred_v)
        #collated_masks_pred_a = torch.utils.data.default_collate(collated_masks_pred_a)
        #collated_masks_enc_v = torch.utils.data.default_collate(collated_masks_enc_v)
        #collated_masks_enc_a = torch.utils.data.default_collate(collated_masks_enc_a)

        #logger.info(f'collated_masks_pred_v shape: {collated_masks_pred_v.shape}')
        #logger.info(f'collated_masks_pred_a shape: {collated_masks_pred_a.shape}')
        #logger.info(f'collated_masks_enc_v shape: {collated_masks_enc_v.shape}')
        #logger.info(f'collated_masks_enc_a shape: {collated_masks_enc_a.shape}')

        #logger.info(f'mask_e_v shape: {mask_e_v.shape}')
        #logger.info(f'mask_e_a shape: {mask_e_a.shape}')
        #logger.info(f'mask_p_v shape: {mask_p_v.shape}')
        #logger.info(f'mask_p_a shape: {mask_p_a.shape}')


        return mask_e_v, mask_e_a, mask_p_v, mask_p_a
