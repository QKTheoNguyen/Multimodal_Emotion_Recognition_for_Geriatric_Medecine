import torch
import torchaudio
import os
import yaml
import datetime
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import *
from torch.utils.data import DataLoader
# from data import EmoDataset

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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(model,
                   model_dir,
                   metadata_file_test, 
                   config,
                   transformation,
                   transformation_type,
                   device,
                   show_matrix=False):
    
    model.eval()

    
    metadata = pd.read_csv(metadata_file_test)
    duration = config["duration"]
    data_dir = config["data_dir"]
    target_sr = config["target_sr"]
    n_samples = target_sr * duration
    correct = 0
    confusion_matrix = torch.zeros(7, 7)

    for song_index in tqdm(range(len(metadata))):

        filepath = metadata.filepath[song_index]
        label = metadata.emotion[song_index]
        label_int = class_mapping_emo[label]
        audio_path = filepath
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)  # convert to mono
        signal = signal.to(device)

        if sr != target_sr:
            signal = torchaudio.transforms.Resample(sr, target_sr)(signal)

        # normalize the signal (peak normalization)
        mean = signal.mean(dim=1, keepdim=True)
        peak = torch.max(torch.max(signal), torch.min(signal))
        signal = (signal - mean) / (1e-10 + peak)

        len_sig = signal.size(1)
        if len_sig < n_samples:
            signal = torch.nn.functional.pad(signal, (0, n_samples - len_sig))
        else:
            signal = signal[:, :n_samples]

        signal_data = signal.unsqueeze(0)
        signal_data = transformation(signal_data)

        with torch.no_grad():
            prediction = model(signal_data)
            total_prediction = torch.mean(prediction, dim=0)

            _, predicted = torch.max(total_prediction, dim=0)

            if predicted == label_int:
                correct += 1

            confusion_matrix[label_int, predicted] += 1

    accuracy = correct / len(metadata)

    print(f'Accuracy of the network on the test songs: {accuracy:.3f}')

    ### Plot confusion matrix ###
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="viridis")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(torch.arange(7))
    ax.set_yticks(torch.arange(7))
    ax.set_xticklabels(class_mapping_emo.keys())
    ax.set_yticklabels(class_mapping_emo.keys())

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(7):
        for j in range(7):
            text = ax.text(j, i, int(confusion_matrix[i, j].item()),
                        ha="center", va="center", color="w")

    ax.set_title(f"Model confusion matrix, Accuracy: {accuracy:.3f}")
    fig.tight_layout()
    if show_matrix:
        plt.show()

    # save figure to file
    fig.savefig(os.path.join("trained", model_dir, "confusion_matrix.png"))
            

    return


if __name__ == "__main__":

    # test model: emorec/20250403-184723
    # test model: emorec/20250403-185500

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model", type=str, required=True)

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config file and set hyperparameters
    model_dir = args.model
    config_path = os.path.join("trained", model_dir, "config.yaml")
    config = load_config(config_path)

    data_dir = config["data_dir"]
    target_sr = config["target_sr"]
    duration = config["duration"]
    transform = config["transform"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    n_frames = config["n_frames"]
    filters = config["filters"]
    add_dropout = config["add_dropout"]
    model_name = config["model_name"]
    metadata_file_test = "data/test_metadata_emo.csv"

    
    if config["n_frames"] is not None:
        n_frames = config["n_frames"]
        n_samples = (n_frames - 1) * hop_length + n_fft
        duration = n_samples / target_sr
    elif config["duration"] is not None:
        duration = config["duration"]
        n_samples = duration * target_sr
        n_frames = 1 + (n_samples - n_fft) // hop_length


    # Create transform

    if transform == "mel_spectrogram":

        transformation = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              n_mels=n_mels,
                                                              normalized=False,
                                                              center=False)
    
    elif transform == "mfcc":
        transformation = torchaudio.transforms.MFCC(sample_rate=target_sr,
                                                    n_mfcc=n_mels,
                                                    melkwargs={"n_fft": n_fft, 
                                                                "hop_length": hop_length, 
                                                                "n_mels": n_mels,
                                                                "normalized": False,
                                                                "center": False})

    
    # Define model

    if model_name == "CNN_Network":
        raise NotImplementedError("CNN_Network is not implemented")
    elif model_name == "MusicRecNet":
        model = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=filters, add_dropout=add_dropout).to(device)
    elif model_name == "CNN_new":
        raise NotImplementedError("CNN_new is not implemented")
    else:
        raise ValueError("Model not found")
    
    model.load_state_dict(torch.load(os.path.join("trained", model_dir, "model.pth")))

    print(f'model name: {model_name}')

    evaluate_model(model, model_dir, metadata_file_test, config, transformation, transform, device, show_matrix=True)