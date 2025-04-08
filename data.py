from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import torchaudio
import os

class_mapping = {
    'Untrained': 0,
    'Beginner': 1,
    'Intermediate': 2,
    'Expert': 3
}

class_mapping_emo = {
    'joy': 0,
    'sadness': 1,
    'anger': 2,
    'fear': 3,
    'disgust': 4,
    'surprise': 5,
    'neutral': 6
}

class EmoDataset(Dataset):
    def __init__(self, config, metadata_file, data_dir, transform, device, random_sample, model_name):
        self.metadata = pd.read_csv(metadata_file)
        self.config = config
        self.data_dir = data_dir
        self.device = device
        self.random_sample = random_sample
        self.model_name = model_name

        self.n_mels = self.config["n_mels"]
        self.n_fft = self.config["n_fft"]
        self.hop_length = self.config["hop_length"]
        self.target_sr = self.config["target_sr"]
        self.full_audio_length = self.config["full_audio_length"]
        self.transform = transform
        
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

        if self.model_name == "AlexNet":
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
        label_int = class_mapping_emo[label]
        audio_path = self.metadata.iloc[index].filepath
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # convert to mono
        signal = signal.to(self.device)
        if self.full_audio_length:
            onset = self.metadata.iloc[index].onset
            offset = self.metadata.iloc[index].offset
        if sr != self.target_sr:
            signal = torchaudio.transforms.Resample(sr, self.target_sr)(signal)

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

        if self.transform == "mel_spectrogram" or self.transform == "mfcc":

            ## normalize the output

            ## normal distribution
            mean = signal.mean(dim=2, keepdim=True)
            stdev = signal.std(dim=2, keepdim=True)
            signal = (signal - mean) / (stdev + 1e-10) 

            ## log normalization
            # signal = torch.log(signal)
        
        elif self.transform == "spectrogram":
            signal = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=80)(signal)

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
    data_dir = "data/EmoData"
    config_path = "config/emo_config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_sr = config["target_sr"]
    n_fft = config["n_fft"] if args.nfft is None else args.nfft
    hop_length = config["hop_length"] if args.hop_length is None else args.hop_length
    n_frames = config["n_frames"] if args.n_frames is None else args.n_frames
    n_mels = config["n_mels"]
    transform = config["transform"] if args.transform is None else args.transform
    model_name = args.model_name


    dataset = EmoDataset(config, metadata_file, data_dir, transform, device, random_sample=True, model_name=model_name)

    train_loader = DataLoader(dataset=dataset, 
                            batch_size=5, 
                            shuffle=False)
    
    (data, target) = next(iter(train_loader))

    print(f"Data shape : {data.size()}")
    print(f"Target shape : {target.size()}")
    print(f"Target : {target}")
    print(f'Data min : {data.min()}, max : {data.max()}')
    print(f'Data mean : {data.mean()}, std : {data.std()}')