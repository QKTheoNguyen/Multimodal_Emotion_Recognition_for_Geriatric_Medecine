from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchaudio
import os
from transformers import Wav2Vec2Processor, WhisperProcessor

class_mapping = {
    'Untrained': 0,
    'Beginner': 1,
    'Intermediate': 2,
    'Expert': 3
}

class_mapping_emo = {
    'enterface': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
    },
    'emodb': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'neutral': 5,
        'boredom': 6,
    },
    'oreau': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6,
    },
    'french': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6,
    },
    'french_2': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6,
    },
    'cafe': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6,
    },
    'ravdess': {
        'joy': 0,
        'sadness': 1,
        'anger': 2,
        'fear': 3,
        'disgust': 4,
        'surprise': 5,
        'neutral': 6,
        'calm': 7
    }
}

class EmoDataset(Dataset):
    def __init__(self, config, metadata_file, data_dir, transform, device, random_sample, model_name, mode):
        self.metadata = pd.read_csv(metadata_file)
        self.config = config
        self.data_dir = data_dir
        self.device = device
        self.random_sample = random_sample
        self.model_name = model_name
        self.transform = transform
        self.mode = mode

        self.n_mels = self.config["n_mels"]
        self.n_fft = self.config["n_fft"]
        self.hop_length = self.config["hop_length"]
        self.target_sr = self.config["target_sr"]
        self.full_audio_length = self.config["full_audio_length"]
        self.augmentation = self.config["augmentation"]
        self.RCS = self.config["RCS"]
        self.database = self.config["database"]

        if self.config["n_frames"] is not None and self.config["duration"] is not None:
            raise ValueError("You can't specify both n_frames and duration")
        elif self.config["n_frames"] is not None:
            self.n_frames = self.config["n_frames"]
            self.n_samples = (self.n_frames - 1) * self.hop_length + self.n_fft
            self.duration = self.n_samples / self.target_sr
        elif self.config["duration"] is not None:
            self.duration = self.config["duration"]
            self.n_samples = self.duration * self.target_sr
            self.n_frames = 1 + (self.n_samples - self.n_fft) // self.hop_length

        if self.model_name != "MusicRecNet":
            self.n_mels = 227
            self.n_frames = 227
            self.n_freq = 227
            self.n_fft = self.n_freq * 2 - 1
            self.hop_length = self.n_fft // 2
            self.n_samples = self.n_frames * self.hop_length + self.n_fft - 1
            self.duration = self.n_samples / self.target_sr

        print(f'Loading audio on length {self.n_samples}, duration {"{:.2f}".format(self.duration)}s, sample rate {self.target_sr}Hz')

        self.transformation = self._create_transformation()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        label = self.metadata.iloc[index].emotion
        label_int = class_mapping_emo[self.database][label]
        # audio_path = self.metadata.iloc[index].filepath
        audio_path = os.path.join(self.data_dir, self.metadata.iloc[index].filepath)
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # convert to mono
        signal = signal.to(self.device)

        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)(signal)

        if self.augmentation and self.mode == "train":
            signal = self._augment(signal)

        if self.full_audio_length:
            onset = self.metadata.iloc[index].onset
            offset = self.metadata.iloc[index].offset

        if signal.size(1) != self.n_samples and not self.full_audio_length:
            signal = self._reshape_signal(signal, random_sample=self.random_sample)

        if self.full_audio_length:
            signal = signal[:, onset:offset]
            if signal.size(1) < self.n_samples:
                signal = torch.nn.functional.pad(signal, (0, self.n_samples - signal.size(1)))

        # normalize the signal (peak normalization)
        mean = signal.mean(dim=1, keepdim=True)
        peak = torch.max(torch.max(signal), torch.min(signal))
        signal = (signal - mean) / (1e-10 + peak)

        signal = self.transformation(signal)

        if self.model_name != "MusicRecNet":
            signal = self._Delta_channels(signal)
        
        signal = self._normalize(signal)

        if self.RCS != 0 and self.mode == "train":
            signal = self._RCS(signal, self.RCS)

        return signal, label_int
    
    def _reshape_signal(self, signal, random_sample):
        len_sig = signal.size(1)
        if len_sig < self.n_samples:
            signal = torch.nn.functional.pad(signal, (0, self.n_samples - len_sig))
        else:
            if random_sample:
                max_offset = len_sig - self.n_samples
                offset = torch.randint(0, max_offset, (1,)).item()
                signal = signal[:, offset:offset + self.n_samples]
            else:
                signal = signal[:, :self.n_samples]
        return signal
    
    def _create_transformation(self):
        if self.transform == "mel_spectrogram":
            transformation = torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sr,
                                                                n_fft=self.n_fft,
                                                                hop_length=self.hop_length,
                                                                n_mels=self.n_mels,
                                                                normalized=False,
                                                                center=False)
        elif self.transform == "log_mel_spectrogram":
            transformation = torchaudio.transforms.MelSpectrogram(sample_rate=self.target_sr,
                                                                n_fft=self.n_fft,
                                                                hop_length=self.hop_length,
                                                                n_mels=self.n_mels,
                                                                normalized=False,
                                                                center=False)
            transformation = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)(transformation)

        elif self.transform == "spectrogram":
            transformation = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
                                                            hop_length=self.hop_length, 
                                                            power=1, 
                                                            normalized=True, 
                                                            center=False)
        elif self.transform == "power_spectrogram":
            transformation = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, 
                                                            hop_length=self.hop_length, 
                                                            power=2, 
                                                            normalized=True, 
                                                            center=False)
        elif self.transform == "mfcc":
            transformation = torchaudio.transforms.MFCC(sample_rate=self.target_sr,
                                                        n_mfcc=self.n_mels,
                                                        melkwargs={"n_fft": self.n_fft, 
                                                                    "hop_length": self.hop_length, 
                                                                    "n_mels": self.n_mels,
                                                                    "normalized": False,
                                                                    "center": False})

        return transformation.to(self.device)
    
    def _augment(self, signal, noise_level_range=(0.005, 0.1)):

        # Randomly apply noise
        p = 0.5
        if torch.rand(1).item() > p:
            return signal
        else:
            noise_level = torch.FloatTensor(1).uniform_(*noise_level_range).item()
            noise = torch.randn_like(signal, device=self.device) * noise_level
            signal = signal + noise

        # Randomly shift the signal
        p = 0.5
        if torch.rand(1).item() > p:
            return signal
        else:
            shift = torch.randint(low=0, high=signal.size(1), size=(1,)).item()
            signal = torch.roll(signal, shifts=shift, dims=1)

        return signal
    
    def _RCS(self, signal, RCS):

        C, F, T = signal.shape
        shift = torch.randint(low=1, high=F-1, size=(RCS,))
        shifted = torch.zeros((RCS, C, F, T), device=self.device)
        for i in range(RCS):
            s = shift[i]
            shifted[i, :, :, s:] = signal[:, :, 0:T-s]
            shifted[i, :, :, 0:s] = signal[:, :, T-s:]
            
        return shifted
    
    def _normalize(self, signal):

        if self.transform == "mel_spectrogram" or self.transform == "mfcc":

            ## normal distribution
            mean = signal.mean(dim=2, keepdim=True)
            stdev = signal.std(dim=2, keepdim=True)
            signal = (signal - mean) / (stdev + 1e-10) 

        
        elif self.transform == "spectrogram" or self.transform == "power_spectrogram":
            signal = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)(signal)

            for i in range(signal.shape[0]):
                channel = signal[i]

                ## Normalize each channel to the range [0, 1]
                min_val = torch.min(channel)
                max_val = torch.max(channel)
                signal[i] = (channel - min_val) / (max_val - min_val + 1e-8)  # Add epsilon to avoid division by zero
            
            # to integer
            signal = torch.round(signal * 255)

        return signal
    
    def _Delta_channels(self, signal):
        delta = torchaudio.transforms.ComputeDeltas()(signal)
        delta_delta = torchaudio.transforms.ComputeDeltas()(delta)
        signal = torch.cat((signal, delta, delta_delta), dim=0)
        return signal
    
    def _get_item_from_path(self, audio_path):
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = signal.to(self.device)
        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)(signal)
        if signal.size(1) != self.n_samples and not self.full_audio_length:
            signal = self._reshape_signal(signal, random_sample=False)
        if self.full_audio_length:
            signal = signal[:, self.onset:self.offset]
            if signal.size(1) < self.n_samples:
                signal = torch.nn.functional.pad(signal, (0, self.n_samples - signal.size(1)))
        # normalize the signal (peak normalization)
        mean = signal.mean(dim=1, keepdim=True)
        peak = torch.max(torch.max(signal), torch.min(signal))
        signal = (signal - mean) / (1e-10 + peak)
        signal = self.transformation(signal)
        if self.model_name != "MusicRecNet":
            signal = self._Delta_channels(signal)
        signal = self._normalize(signal)
        return signal
    
class EmoDataset_Wav2Vec2(Dataset):
    def __init__(self, config, metadata_file, data_dir, device, random_sample, model_name, mode, wav2vec2_path=None):
        self.metadata = pd.read_csv(metadata_file)
        self.config = config
        self.data_dir = data_dir
        self.device = device
        self.random_sample = random_sample
        self.model_name = model_name
        self.mode = mode

        if self.model_name in ["wav2vec2", "hubert", "wavlm"]:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        elif self.model_name in ["whisper-tiny", "whisper-base"]:
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")

        self.target_sr = self.config["target_sr"]
        self.full_audio_length = self.config["full_audio_length"]
        self.augmentation = self.config["augmentation"]
        self.RCS = self.config["RCS"]
        self.database = self.config["database"]
        self.duration = self.config["duration"]
        self.n_samples = int(self.duration * self.target_sr)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        label = self.metadata.iloc[index].emotion
        label_int = class_mapping_emo[self.database][label]
        audio_path = os.path.join(self.data_dir, self.metadata.iloc[index].filepath)
        signal, sr = torchaudio.load(audio_path, normalize=True)
        signal = signal.to(self.device)

        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        
        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)(signal)

        if signal.size(1) != self.n_samples and not self.full_audio_length:
            signal = self._reshape_signal(signal, random_sample=self.random_sample)
        
        signal = signal.squeeze(0)

        if self.augmentation:
            signal = self._augment(signal)

        if self.model_name in ["wav2vec2", "hubert", "wavlm"]:
            
            inputs = self.processor(
                signal,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=False
            )

            input_values = inputs.input_values.squeeze(0)
        
        elif self.model_name in ["whisper-tiny", "whisper-base"]:

            signal = signal.cpu().numpy()  # Convert to numpy array for Whisper processor
            inputs = self.processor(
                audio=signal,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=False
            )
            input_values = inputs.input_features.squeeze(0)
            input_values = input_values.to(self.device)

        return input_values, label_int
    
    def _reshape_signal(self, signal, random_sample):
        len_sig = signal.size(1)
        if len_sig < self.n_samples:
            signal = torch.nn.functional.pad(signal, (0, self.n_samples - len_sig))
        else:
            if random_sample:
                max_offset = len_sig - self.n_samples
                offset = torch.randint(0, max_offset, (1,)).item()
                signal = signal[:, offset:offset + self.n_samples]
            else:
                signal = signal[:, :self.n_samples]
        return signal
    
    def _augment(self, signal, noise_level_range=(0.005, 0.1)):

        # Randomly apply noise
        p = 0.5
        if torch.rand(1).item() > p:
            return signal
        else:
            noise_level = torch.FloatTensor(1).uniform_(*noise_level_range).item()
            noise = torch.randn_like(signal, device=self.device) * noise_level
            signal = signal + noise

        # Randomly shift the signal
        p = 0.5
        if torch.rand(1).item() > p:
            return signal
        else:
            shift = torch.randint(low=0, high=signal.size(0), size=(1,)).item()
            signal = torch.roll(signal, shifts=shift, dims=0)

        return signal
    
    def _get_item_from_path(self, audio_path):
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = signal.to(self.device)
        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)(signal)
        if signal.size(1) != self.n_samples and not self.full_audio_length:
            signal = self._reshape_signal(signal, random_sample=False)

        signal = signal.squeeze(0)
        
        inputs = self.processor(
            signal,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False
        )
        input_values = inputs.input_values.squeeze(0)
        input_values = input_values.to(self.device)

        return input_values

        

    #TODO: define __getitem__ for wav2vec
        

if __name__ == "__main__":

    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--nfft", type=int, default=None)
    parser.add_argument("--n_frames", type=int, default=None)
    parser.add_argument("--hop_length", type=int, default=None)
    parser.add_argument("-t","--transform", type=str, default=None)
    parser.add_argument("-m","--model_name", type=str, default=None)

    args = parser.parse_args()

    def load_config(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    metadata_file = 'data/metadata_emo.csv'
    data_dir = os.path.join("/home","tnguyen","Documents","Emotion_recognition","Multimodal_Emotion_Recognition_for_Geriatric_Medecine","data","EmoData")
    config_path = "config/emo_config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sr = config["target_sr"]
    n_fft = config["n_fft"] if args.nfft is None else args.nfft
    hop_length = config["hop_length"] if args.hop_length is None else args.hop_length
    n_frames = config["n_frames"] if args.n_frames is None else args.n_frames
    n_mels = config["n_mels"]
    transform = config["transform"] if args.transform is None else args.transform
    model_name = config["model_name"] if args.model_name is None else args.model_name


    # dataset = EmoDataset(config, metadata_file, data_dir, transform, device, random_sample=True, model_name=model_name, mode="train")
    dataset = EmoDataset_Wav2Vec2(config, metadata_file, data_dir, device, random_sample=True, model_name=model_name, mode="train")

    train_loader = DataLoader(dataset=dataset, 
                            batch_size=5, 
                            shuffle=False)
    
    (data, target) = next(iter(train_loader))

    print(f"Data shape : {data.size()}")
    print(f"Target shape : {target.size()}")
    print(f"Target : {target}")
    print(f'Data min : {data.min()}, max : {data.max()}')
    print(f'Data mean : {data.mean()}, std : {data.std()}')
    print(f'device : {device}')

    # # save data to check
    # import matplotlib.pyplot as plt
    # import numpy as np
    
    # # data_test_path = "/home/tnguyen/Documents/Emotion_recognition/Multimodal_Emotion_Recognition_for_Geriatric_Medecine/data/test_data"

    # for i in range(data.size(0)):
    #     plt.imshow(data[i,1].cpu().numpy(), cmap='viridis', aspect='auto')
    #     plt.colorbar()
    #     plt.xlabel('Time')
    #     plt.ylabel('Frequency')
    #     plt.gca().invert_yaxis()
    #     plt.title(f'Target : {target[i]}')
    #     plt.savefig(os.path.join(data_test_path, f'test_{i}.png'))
    #     plt.axis('off')
    #     plt.close()