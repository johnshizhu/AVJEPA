# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from logging import getLogger

logger = getLogger()


def apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    # -- masks are a tensor of INDICES of tokens that should be masked
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        tmp = [torch.gather(x, dim=1, index=mask_keep)]
        all_x += tmp
    if not concat:
        return all_x
    out = torch.cat(all_x, dim=1)
    return out

def target_apply_masks(x, masks, concat=True):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors of shape [B, K] containing indices of K patches in [N] to keep
    """
    # Create a combined mask
    f_masks = torch.cat([m.view(-1) for m in masks])
    c_mask = torch.tile(torch.unique(f_masks), (24, 1))
    mask_keep = c_mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
    return torch.gather(x, dim=1, index=mask_keep)

def get_pred_masks(enc_masks, modal):
    '''
        enc_masks: LIST of (batch_size, mask_size), mask is a vector of indices of tokens to be masked
    '''
    batch_size, _ = enc_masks[0].shape
    pred_masks = []
    device = enc_masks[0].device

    # -- full range of indices depends on modality of mask
    if modal == 0: # video
        full = torch.arange(0, 1568, device=device)
    if modal == 1: # audio
        full = torch.arange(0, 96, device=device)
    full = full.unsqueeze(0).expand(batch_size, -1)

    for m in enc_masks:
        batch = []

        for i in range(batch_size):
            s_mask = m[i]
            s_full = full[i]
            p_indices = s_full[~torch.isin(s_full, s_mask)]
            batch.append(p_indices)

        batch = torch.stack(batch)
        pred_masks.append(batch)

    return pred_masks