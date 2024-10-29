# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pathlib
import warnings

from logging import getLogger

import numpy as np
import pandas as pd

from decord import VideoReader, cpu

import ffmpeg
import librosa

import torch

from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


def make_audiovideodataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    frame_step=4,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    duration=None,
    log_dir=None,
):
    dataset = AudioVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=duration,
        shared_transform=shared_transform,
        transform=transform)

    logger.info('AudioVideoDataset dataset created')
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset.sample_weights,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0)
    logger.info('AudioVideoDataset unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class AudioVideoDataset(torch.utils.data.Dataset):
    """ Audio Video Dataset """

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,  # duration in seconds
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration

        if VideoReader is None:
            raise ImportError('Unable to import "decord" which is required to read videos.')

        # Load video paths and labels
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:

            if data_path[-4:] == '.csv':
                data = pd.read_csv(data_path, header=None, delimiter=" ")
                samples += list(data.values[:, 0])  # VIDEO
                labels += list(data.values[:, 1])
                num_samples = len(data)
                self.num_samples_per_dataset.append(num_samples)

            elif data_path[-4:] == '.npy':
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                self.num_samples_per_dataset.append(len(data))

        # [Optional] Weights for each sample to be used by downstream
        # weighted video sampler
        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample = self.samples[index]

        # Keep trying to load video/audio until you find a valid sample
        loaded_video = False
        while not loaded_video:
            buffer, clip_indices, sgram = self.loadaudiovideo_decord(sample)  # [T H W 3]
            loaded_video = len(buffer) > 0
            if not loaded_video:
                index = np.random.randint(self.__len__())
                sample = self.samples[index]

        #logger.info(f'buffer shape: {buffer.shape}')
        #logger.info(f'sgram shape: {sgram.shape}')

        # Label/annotations for video
        label = self.labels[index]

        def split_into_clips(video):
            """ Split video into a list of clips """
            fpc = self.frames_per_clip
            nc = self.num_clips
            return [video[i*fpc:(i+1)*fpc] for i in range(nc)]

        # Parse video into frames & apply data augmentations
        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)
        buffer = split_into_clips(buffer)
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]

        #logger.info(f'buffer type: {type(buffer)}')
        #logger.info(f'label type: {type(label)}')
        #logger.info(f'clip_indices type: {type(clip_indices)}')
        #print("\n")
        return buffer, label, clip_indices, sgram

    def loadaudiovideo_decord(self, sample):
        """ Load video AND audio content using Decord and ffmpeg """

        fname = sample
        if not os.path.exists(fname):
            warnings.warn(f'data path not found {fname=}')
            return [], None, None

        _fsize = os.path.getsize(fname)
        if _fsize < 1 * 1024:  # avoid hanging issue
            warnings.warn(f'video too short {fname=}')
            return [], None, None
        if _fsize > self.filter_long_videos:
            warnings.warn(f'skipping long video of size {_fsize=} (bytes)')
            return [], None, None

        # VIDEO - Get fps immediately after creating VideoReader
        try:
            vr = VideoReader(fname, num_threads=-1, ctx=cpu(0))
            fps = vr.get_avg_fps()  # Get fps here
            if fps <= 0 or np.isnan(fps):
                warnings.warn(f'Invalid fps value: {fps}')
                return [], None, None
        except Exception:
            return [], None, None
        
        # AUDIO
        try:
            stream = ffmpeg.input(fname)
            audio_stream = stream.audio
            # Get audio as 16-bit PCM with original sample rate
            audio_array, err = (
                audio_stream
                .output('pipe:', format='f32le', acodec='pcm_f32le')
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            # Convert binary audio data to numpy array
            audio_data = np.frombuffer(audio_array, np.float32)
            
            # Get the audio sample rate from the input file
            probe = ffmpeg.probe(fname)
            audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            sr = int(audio_info['sample_rate'])
        
        except Exception as e:
            warnings.warn(f'Failed to load audio: {str(e)}')
            return [], None, None

        fpc = self.frames_per_clip
        fstp = self.frame_step
        if self.duration is not None:
            try:
                fstp = int(self.duration * fps / fpc)
            except Exception as e:
                warnings.warn(str(e))
                return [], None, None
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f'skipping video of length {len(vr)}')
            return [], None, None

        vr.seek(0)  # Go to start of video before sampling frames

        # Partition video into equal sized segments and sample each clip
        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        try:
            for i in range(self.num_clips):
                if partition_len > clip_len:
                    # If partition_len > clip len, then sample a random window of
                    # clip_len frames within the segment
                    end_indx = clip_len
                    if self.random_clip_sampling:
                        end_indx = np.random.randint(clip_len, partition_len)
                    start_indx = end_indx - clip_len
                    indices = np.linspace(start_indx, end_indx, num=fpc)
                    indices = np.clip(indices, start_indx, end_indx-1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    if not self.allow_clip_overlap:
                        indices = np.linspace(0, partition_len, num=partition_len // fstp)
                        indices = np.concatenate((indices, np.ones(fpc - partition_len // fstp) * partition_len,))
                        indices = np.clip(indices, 0, partition_len-1).astype(np.int64)
                        indices = indices + i * partition_len
                    else:
                        sample_len = min(clip_len, len(vr)) - 1
                        indices = np.linspace(0, sample_len, num=sample_len // fstp)
                        indices = np.concatenate((indices, np.ones(fpc - sample_len // fstp) * sample_len,))
                        indices = np.clip(indices, 0, sample_len-1).astype(np.int64)
                        clip_step = 0
                        if len(vr) > clip_len:
                            clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                        indices = indices + i * clip_step

                clip_indices.append(indices)
                all_indices.extend(list(indices))

            buffer = vr.get_batch(all_indices).asnumpy()
        except Exception as e:
            warnings.warn(f'Failed to sample video frames: {str(e)}')
            return [], None, None

        try:
            # Audio processing with guaranteed fps
            frame_duration_sec = 1 / fps
            start_frame = int(clip_indices[0][0])
            end_frame = int(clip_indices[-1][-1])

            start_time = start_frame * frame_duration_sec
            end_time = (end_frame + 1) * frame_duration_sec 

            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            if end_sample > len(audio_data):
                warnings.warn('Audio clip exceeds audio data length')
                return [], None, None
                
            audio_clip = audio_data[start_sample:end_sample]

            if len(audio_clip) == 0:
                warnings.warn('Empty audio clip')
                return [], None, None

            # Compute spectrogram
            S = librosa.stft(audio_clip, 
                            n_fft=2048,
                            hop_length=512,
                            win_length=None,
                            window='hann')
            
            # Convert to mel spectrogram
            mel_S = librosa.feature.melspectrogram(S=np.abs(S), 
                                                sr=sr,
                                                n_mels=128)
            
            # Convert to log scale
            S_dB = librosa.power_to_db(mel_S, ref=np.max, top_db=80)
            #logger.info(f'buffer type is: {type(buffer)}')
            #logger.info(f'buffer shape is: {buffer.shape}')

            # cropping / padding audiospectrogram to constant shape
            sgram_shape = S_dB.shape

            if (sgram_shape[1] > 184):
                sgram = S_dB[:, :184]

            if (sgram_shape[1] < 184):
                sgram = np.pad(S_dB, ((0, 0), (0, 184 - sgram_shape[1])), mode='constant')

            #logger.info(f'sgram type is: {type(S_dB)}')
            #logger.info(f'sgram shape is: {S_dB.shape}')

            return buffer, clip_indices, sgram
        
        except Exception as e:
            warnings.warn(f'Failed to process audio: {str(e)}')
            return [], None, None
    
    def __len__(self):
        return len(self.samples)
