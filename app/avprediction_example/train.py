# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn import DataParallel

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.avmultiblock3d import AVMaskCollator as AVMB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.avprediction.utils import (
    load_checkpoint,
    init_audio_video_model,
    init_probe_opt
)
from app.vjepa.transforms import make_transforms
from src.models.prediction_probes import LinearProbe, AttentiveProbe, FactorizedProbe, ConvolutionalProbe, ProgressiveProbe, AttentionProbe, ConvTemporalProbe, PoolingProbe
from app.avprediction.utils import rebuild_tokens

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    # -- init model
    # logger.info("Initializing models")
    # logger.info(f"uniform_power: {uniform_power}")
    # logger.info(f'use_mask_tokens: {use_mask_tokens}')
    # logger.info(f'num_mask_tokens: {len(cfgs_mask)}')
    # logger.info(f'zero_init_mask_tokens: {zero_init_mask_tokens}')
    # logger.info(f'device: {device}')
    # logger.info(f'patch_size: {patch_size}')
    # logger.info(f'num_frames: {num_frames}')
    # logger.info(f'tubelet_size: {tubelet_size}')
    # logger.info(f'model_name: {model_name}')
    # logger.info(f'crop_size: {crop_size}')
    # logger.info(f'pred_depth: {pred_depth}')
    # logger.info(f'pred_embed_dim: {pred_embed_dim}')
    # logger.info(f'use_sdpa: {use_sdpa}')
    # print(1/0)
    encoder, predictor = init_audio_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = AVMB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size)

    # -- init data-loaders/samplers
    print(f"dataset_type is: {dataset_type}")
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=True,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    # -- init optimizer and scheduler
    encoder = DataParallel(encoder)
    predictor = DataParallel(predictor)
    target_encoder = DataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    logger.info("LOADING CHECKPOINTS")
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()
    
    logger.info("Freezing encoder and predictor weights")
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False
    
    # -- Initializing probe
    probe_checkpoint = torch.load(r"C:\Users\johns\OneDrive\Desktop\jepa_logs\avjepaSmall-prediction-testing-FIRSTTRY-latest.pth.tar", map_location='cpu')
    probe = AttentionProbe(emb_dim=384)
    pretrained_dict = probe_checkpoint['probe']    
    msg = probe.load_state_dict(pretrained_dict)
    logger.info(f'msg: {msg}')
    probe = probe.to(device)
    logger.info("Freezing encoder and predictor weights")
    for p in probe.parameters():
        p.requires_grad = False

    logger.info(f'AttentionProbe Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    logger.info(f'Encoder Trainable Parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}')
    logger.info(f'Predictor Trainable Parameters: {sum(p.numel() for p in predictor.parameters() if p.requires_grad)}')

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            'probe': probe.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')

    # -- probe optimizer
    logger.info(f'Initializing Probe Optimizer')
    optimizer, scaler, scheduler, wd_scheduler = init_probe_opt(
        probe=probe,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)
    
    # -- TRAINING LOOP
    epoch = 0
    logger.info('Beginning Training Loop...')
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()
            try:
                udata, masks_enc_v, masks_enc_a, masks_pred_v, masks_pred_a = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc_v, masks_enc_a, masks_pred_v, masks_pred_a = next(loader)
            assert len(masks_enc_v) == len(masks_pred_v), \
                'Currently require num encoder masks = num predictor masks'
            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)

                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc_v, _masks_pred_v = [], []
                _masks_enc_a, _masks_pred_a = [], []

                for _me, _mp in zip(masks_enc_v, masks_pred_v):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc_v.append(_me)
                    _masks_pred_v.append(_mp)
                for _me, _mp in zip(masks_enc_a, masks_pred_a):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc_a.append(_me)
                    _masks_pred_a.append(_mp)
            
                return (clips, _masks_enc_v, _masks_enc_a, _masks_pred_v, _masks_pred_a)

            clips, masks_enc_v, masks_enc_a, masks_pred_v, masks_pred_a = load_clips()

            # At this point the video has not been altered in any way

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc_v[_i][0].size(-1))

            # Audiospectrogram Data Edit
            asgram = udata[3].unsqueeze(1).to(device)

            def train_step():

                def forward_frozen(c, a):
                    
                    z = encoder(c, a, masks_enc) # provide masked input to encoder
                    
                    #-- splitting video and audio tokens post-encoder-embedding
                    v_size = []
                    a_size = []
                    for i in masks_enc:
                        _, v_size_e = i[0].shape
                        _, a_size_e = i[1].shape
                        v_size.append(v_size_e)
                        a_size.append(a_size_e)
                    z_t = []
                    for index, i in enumerate(z):
                        z_v, z_a = torch.split(i, [v_size[index], a_size[index]], dim=1)
                        z_t.append((z_v, z_a))

                    d = predictor(z_t, z_t, masks_enc, masks_pred) # preditor reads in embedded non-masked tokens and predicts masked tokens
                    
                    return z, d

                def forward_probe(z):
                    res_v = []
                    res_a = []
                    for i in z:
                        v, a = probe(i)
                        res_v.append(v)
                        res_a.append(a)
                    return res_v, res_a

                # Step 1. Forward
                masks_enc = list(zip(masks_enc_v, masks_enc_a))
                masks_pred = list(zip(masks_pred_v, masks_pred_a))
                logger.info(f'masks_pred[0] video: {masks_pred[0][0]}')
                logger.info(f'masks_pred[0] audio: {masks_pred[0][1]}')
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    ctxt, pred = forward_frozen(clips, asgram)
                    emb_tokens = rebuild_tokens(ctxt, pred, masks_enc, masks_pred)
                    o_v, o_a = forward_probe(emb_tokens)
                    import matplotlib.pyplot as plt
                    import numpy as np

                    def plot_video_and_audio(clips, o_v, asgram, o_a, selected_frame=5):
                        clips = clips.to(dtype=torch.float32).detach().cpu().squeeze(0)
                        asgram = asgram.to(dtype=torch.float32).detach().cpu().squeeze(0).squeeze(0)
                        o_v = [tensor.to(dtype=torch.float32).detach().cpu() for tensor in o_v]
                        o_a = [tensor.to(dtype=torch.float32).detach().cpu() for tensor in o_a]
                        
                        # Prepare the fifth frame and spectrograms for plotting
                        clips_fifth = clips[:, selected_frame, :, :]
                        o_v_0_fifth = o_v[0][:, selected_frame, :, :]
                        clips_fifth = clips_fifth.permute(1, 2, 0).numpy()
                        o_v_0_fifth = o_v_0_fifth.permute(1, 2, 0).numpy()
                        
                        # Create figure with subplots
                        fig = plt.figure(figsize=(12, 6))
                        
                        # Plot the original audio spectrogram
                        plt.subplot(2, 2, 1)
                        plt.imshow(asgram, cmap='gray')
                        plt.title("Original Audio Spectrogram")
                        plt.axis("off")
                        
                        # Plot the reconstructed audio spectrogram
                        plt.subplot(2, 2, 2)
                        plt.imshow(o_a[0].squeeze(0), cmap='gray')
                        plt.title("Reconstructed Audio Spectrogram")
                        plt.axis("off")
                        
                        # Plot the original video fifth frame
                        plt.subplot(2, 2, 3)
                        plt.imshow(clips_fifth)
                        plt.title(f"Original Video")
                        plt.axis("off")
                        
                        # Plot the reconstructed video fifth frame
                        plt.subplot(2, 2, 4)
                        plt.imshow(o_v_0_fifth)
                        plt.title(f"Reconstructed Video")
                        plt.axis("off")
                        
                        plt.tight_layout()
                        plt.show()
                        return fig
                    
                    plot_video_and_audio(clips, o_v[0], asgram, o_a[0])
            train_step()