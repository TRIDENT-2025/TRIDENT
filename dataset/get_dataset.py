import os
import re
import librosa
import torch
import argparse
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from dataset.get_transform import get_img_transform, get_rf_transform, get_audio_transform

# Helper function for sorting numerically
def numerical_sort(value):
    parts = re.split('(\d+)', value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Dataset class
class DroneFusionDataset(Dataset):
    def __init__(self, audio_root, video_root, rf_root, dataset_type='Train', transform=None, n_mfcc=40, n_fft=2048, hop_length=512, sample_rate=44100, duration=10, segment_length=0.25):
        self.audio_root = audio_root
        self.video_root = video_root
        self.rf_root = rf_root
        self.dataset_type = dataset_type.lower()
        self.transform = transform
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_length = segment_length  # Length of each segment in seconds
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        categories = ['Drone', 'Background']
        video_folder_name = 'Images_Extracted' if self.dataset_type == 'train' else 'Images'
        rf_folder_name = 'Images_Spectrograms' if self.dataset_type == 'train' else 'Images'

        for category in categories:
            audio_category_path = os.path.join(self.audio_root, category)
            video_category_path = os.path.join(self.video_root, category)
            rf_category_path = os.path.join(self.rf_root, category)

            scenarios = sorted(os.listdir(audio_category_path), key=numerical_sort)
            for scenario in scenarios:
                audio_scenario_path = os.path.join(audio_category_path, scenario)
                video_scenario_path = os.path.join(video_category_path, scenario, video_folder_name)
                rf_scenario_path = os.path.join(rf_category_path, scenario, rf_folder_name)

                audio_files = sorted([f for f in os.listdir(audio_scenario_path) if f.endswith('.wav')], key=numerical_sort)
                for audio_file in audio_files:
                    audio_file_number = re.findall('\d+', audio_file)[0]
                    audio_path = os.path.join(audio_scenario_path, audio_file)
                    label = 0 if category == "Drone" else 1

                    num_segments = int(self.duration / self.segment_length)

                    for segment in range(num_segments):
                        start_time = segment * self.segment_length
                        end_time = (segment + 1) * self.segment_length

                        # Calculate frame start and end times for video and RF
                        video_frame_start = int(start_time * 28) + 1
                        video_frame_end = int(end_time * 28)

                        # 4 frames per second for RF
                        rf_frame_start = int(start_time * 4) + 1
                        rf_frame_end = int(end_time * 4)

                        # Ensure frame numbers are within valid range
                        video_frame_start = max(1, video_frame_start)
                        video_frame_end = min(video_frame_end, 300)  
                        rf_frame_start = max(1, rf_frame_start)
                        rf_frame_end = min(rf_frame_end, 40) 

                        video_frames = [os.path.join(video_scenario_path, f"{audio_file_number}_frame_{i}.jpg") for i in range(video_frame_start, video_frame_end + 1)]
                        rf_frames = [os.path.join(rf_scenario_path, f"{audio_file_number}_frame_{i}.jpg") for i in range(rf_frame_start, rf_frame_end + 1)]

                        samples.append({
                            'audio_path': audio_path,
                            'video_frame_paths': video_frames,
                            'video_sequence_label': label,
                            'rf_frame_paths': rf_frames,
                            'rf_sequence_label': label,
                            'label': label,
                            'start': start_time,
                            'end': end_time
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sr = librosa.load(sample['audio_path'], sr=self.sample_rate, offset=sample['start'], duration=self.segment_length)
        
        # Apply audio-specific transformations
        if 'audio' in self.transform:
            waveform, sr = self.transform['audio'](waveform, sr)
            
        waveform = np.asfortranarray(waveform[::3])  # Consider if this decimation is necessary with shorter segments
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        pad_width = max(0, self.n_mfcc - mfcc.shape[1])
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        video_frames = [Image.open(path).convert("RGB") for path in sample['video_frame_paths']]
        # Apply video-specific transformations
        if 'video' in self.transform:
            video_frames_tensor = torch.stack([self.transform['video'](frame) for frame in video_frames]) if video_frames else torch.empty(0)
        else:
            video_frames_tensor = torch.empty(0)

        rf_frames = [Image.open(path).convert("RGB") for path in sample['rf_frame_paths']]
        # Apply RF-specific transformations
        if 'rf' in self.transform:
            rf_frames_tensor = torch.stack([self.transform['rf'](frame) for frame in rf_frames]) if rf_frames else torch.empty(0)
        else:
            rf_frames_tensor = torch.empty(0)

        return audio_tensor, video_frames_tensor, rf_frames_tensor, sample['label']

# DataLoader function
def get_loader(args, dataset, distributed=False, is_train=True):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None) and is_train,
        sampler=sampler,
        drop_last=False,
        num_workers=1,
        pin_memory=True
    )
    return loader, sampler


# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=r"TRIDENT/Dataset")
    parser.add_argument("--distributed-train", action='store_true', help="Enable distributed training")
    parser.add_argument("--distributed-test", action='store_true', help="Enable distributed test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--scale-size", type=int, default=640)
    args = parser.parse_args()

    train_transform = {
        'audio': get_audio_transform(args, is_training=True),
        'video': get_img_transform(args, is_training=True),
        'rf': get_rf_transform(args, is_training=True)
    }
    test_transform = {
        'audio': get_audio_transform(args, is_training=True),
        'video': get_img_transform(args, is_training=True),
        'rf': get_rf_transform(args, is_training=True)
    }

    audio_root = os.path.join(args.dataset_dir, 'Audio')
    video_root = os.path.join(args.dataset_dir, 'Video')
    rf_root = os.path.join(args.dataset_dir, 'RF_Spectrograms')

    train_dataset = DroneFusionDataset(audio_root+'/Train', video_root+'/Train', rf_root+'/Train', dataset_type='Train', transform=train_transform)
    test_dataset = DroneFusionDataset(audio_root+'/Test', video_root+'/Test', rf_root+'/Test', dataset_type='Test', transform=test_transform)

    train_loader, train_sampler = get_loader(args, train_dataset, distributed=args.distributed_train, is_train=True)
    test_loader, test_sampler = get_loader(args, test_dataset, distributed=args.distributed_test, is_train=False)

    for audio, video, rf, label in train_loader:
        print("Train Loader:")
        print(audio.size(), video.size(), rf.size(), label.size())
        break

    for audio, video, rf, label in test_loader:
        print("Test Loader:")
        print(audio.size(), video.size(), rf.size(), label.size())
        break
