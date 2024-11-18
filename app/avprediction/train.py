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
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
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
    init_opt,
    init_probe_opt
)
from app.vjepa.transforms import make_transforms
from src.models.prediction_probes import LinearProbe, AttentiveProbe, FactorizedProbe, ConvolutionalProbe, ProgressiveProbe, AttentionProbe, ConvTemporalProbe, PoolingProbe

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
        mask_collator = MB3DMaskCollator(
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
    #logger.info(f'init_data complete...')
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
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
    # probe = LinearProbe()
    # logger.info(f'Linear Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    # probe = AttentiveProbe()
    # logger.info(f'Attentive Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    # probe = FactorizedProbe()
    # logger.info(f'Factorized Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    # probe = ProgressiveProbe()
    # probe = probe.to(device)
    # logger.info(f'Progressive Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    
    probe_checkpoint = torch.load(r"C:\Users\johns\OneDrive\Desktop\jepa_logs\avjepaSmall-prediction-v3-latest.pth.tar", map_location='cpu')
    probe = AttentionProbe(emb_dim=384)
    pretrained_dict = probe_checkpoint['probe']    
    msg = probe.load_state_dict(pretrained_dict)
    logger.info(f'msg: {msg}')
    probe = probe.to(device)

    logger.info("Freezing encoder and predictor weights")
    for p in probe.parameters():
        p.requires_grad = False

    logger.info(f'AttentionProbe Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    # probe = PoolingProbe()
    # logger.info(f'PoolingProbe Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    # probe = ConvTemporalProbe()
    # logger.info(f'ConvTemporalProbe Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
    #probe = ConvolutionalProbe()
    #logger.info(f'Convolutional Parameters: {sum(p.numel() for p in probe.parameters() if p.requires_grad)}')
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
    logger.info(f'Initializing Probe Optimizers')
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
    #logger.info('Beginning Training Loop...')
    # -- TRAINING LOOP
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
            #logger.info(f'Beginning Iteration {itr}...')
            itr_start_time = time.time()
            try:
                udata, masks_enc, masks_pred = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'
            def load_clips():
                # Normalize clips to [0,1] range
                clips = torch.cat([normalize_tensor(u.to(device, non_blocking=True)) for u in udata[0]], dim=0)

                # Process masks
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips) 
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)

            def normalize_tensor(tensor):
                tensor = tensor - tensor.min()
                tensor = tensor / (tensor.max() + 1e-8)
                return tensor

            # logger.info(f'udata length is: {len(udata)}')
            # logger.info(f'udata[0] len is: {len(udata[0])}')
            # logger.info(f'udata[1] shape is: {udata[1].shape}')
            # logger.info(f'udata[2] len is: {len(udata[2])}')
            # logger.info(f'udata[0][0] shape is: {udata[0][0].shape}')
            # logger.info(f'udata[2][0] shape is: {udata[2][0].shape}')
            # logger.info(f'udata[3] shape is: {udata[3].shape}')

            # logger.info(f'masks_enc type is: {type(masks_enc)}')
            # logger.info(f'masks_pred type is: {type(masks_pred)}')


            clips, masks_enc, masks_pred = load_clips()


            # print("\n")
            # logger.info(f'clips shape is: {clips.shape}')
            # logger.info(f'masks_enc length is: {len(masks_enc)}')
            # logger.info(f'masks_enc[0] shape is: {masks_enc[0].shape}')
            # logger.info(f'masks_enc[1] shape is: {masks_enc[1].shape}')
            # logger.info(f'masks_pred length is: {len(masks_pred)}')
            # logger.info(f'masks_pred[0] shape is: {masks_pred[0].shape}')
            # logger.info(f'masks_pred[1] shape is: {masks_pred[1].shape}')
            # print("\n")

            # At this point the video has not been altered in any way

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            # Audiospectrogram Data Edit
            asgram = udata[3].unsqueeze(1)

            def train_step():
                #logger.info("Beginning Train Step...")
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_frozen(c, a):
                    
                    # Pass the clip and associated data through the encoder
                    #logger.info("Running encoder...")
                    z = encoder(c, a, masks_enc)

                    # Pass the encoded features through the predictor
                    #logger.info("Running predictor...")
                    z = predictor(z, None, masks_enc, masks_pred)
                    return z[0]

                def forward_probe(z):
                    return probe(z)

                def loss_fn(output, original):
                    #logger.info(f"output shape is: {output.shape}")
                    #logger.info(f'original shape is: {original.shape}')
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    loss = F.mse_loss(output, original)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1. Forward
                #logger.info('Beginning Forward Pass...')
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    #logger.info(f'c type is: {type(clips)}')
                    #logger.info(f'c shape is: {clips.shape}\n')
                    #logger.info(f'clips shape is: {clips.shape}')
                    #logger.info(f'asgram shape is: {asgram.shape}')
                    z = forward_frozen(clips, asgram)
                    #logger.info(f'z shape is: {z.shape}')

                    o = forward_probe(z)
                    loss_jepa = loss_fn(o, clips) 
                    #logger.info(f'o type is: {type(o)}')
                    #logger.info(f'o shape is: {o.shape}')

                    # #logger.info(f'forward_context done')
                    # loss_jepa = loss_fn(o, clips)  # jepa prediction loss
                    # logger.info(f'loss is: {loss_jepa}')
                    logger.info(f'masks_enc len is: {len(masks_enc)}')
                    logger.info(f'masks_pred len is: {len(masks_pred)}')
                    logger.info(f'masks_enc[0] shape is: {masks_enc[0].shape}')
                    logger.info(f'masks_pred[0] shape is: {masks_pred[0].shape}')
                    logger.info(f'masks_enc[1] shape is: {masks_enc[1].shape}')
                    logger.info(f'masks_pred[1] shape is: {masks_pred[1].shape}')
                    
                    import matplotlib.pyplot as plt
                    import numpy as np

                    def normalize_tensor(tensor):
                        tensor = tensor - tensor.min()
                        tensor = tensor / (tensor.max() + 1e-8)
                        return tensor

                    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

                    for i in range(2):
                        # Get images
                        image1 = clips[i, :, 0, :, :].permute(1, 2, 0).detach().cpu().numpy()
                        image2 = o[i, :, 0, :, :].permute(1, 2, 0).detach().cpu().float().numpy()
                        
                        # Create masked version by setting masked patches to black
                        mask = masks_enc[0][i, 0].bool().cpu().numpy()
                        masked_img = image1.copy()
                        masked_img[mask] = 0  # Set to black to represent removed tokens
                        
                        # Get normalized versions
                        image1_norm = normalize_tensor(torch.tensor(image1)).numpy()
                        image2_norm = normalize_tensor(torch.tensor(image2)).numpy()
                        
                        axes[i, 0].imshow(image1)
                        axes[i, 0].axis('off')
                        axes[i, 0].set_title(f"Vid{i} Original")
                        
                        axes[i, 1].imshow(masked_img) 
                        axes[i, 1].axis('off')
                        axes[i, 1].set_title(f"Vid{i} Masked (Black = Removed)")
                        
                        axes[i, 2].imshow(image2)
                        axes[i, 2].axis('off')
                        axes[i, 2].set_title(f"Vid{i} Prediction")
                        
                        axes[i, 3].imshow(image1_norm)
                        axes[i, 3].axis('off')
                        axes[i, 3].set_title(f"Vid{i} Original (Norm)")
                        
                        axes[i, 4].imshow(image2_norm)
                        axes[i, 4].axis('off')
                        axes[i, 4].set_title(f"Vid{i} Prediction (Norm)")

                    plt.tight_layout()
                    plt.show()


                    print(1/0)
                    pstd_z = reg_fn(z)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg
                #logger.info(f'Loss is: {loss}')
                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg,
                           wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr,
                               optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min,
                               optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min,
                               optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max,
                               grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats_pred.first_layer,
                               grad_stats_pred.last_layer,
                               grad_stats_pred.min,
                               grad_stats_pred.max,
                               grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint
        logger.info('avg. loss %.3f' % loss_meter.avg)
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f'{tag}-e{epoch}.pth.tar'
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)