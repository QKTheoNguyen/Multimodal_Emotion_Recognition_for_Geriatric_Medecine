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
from data import *
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
    'neutral': 6,
    'boredom': 7,
}

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def evaluate_model(model,
                   model_dir,
                   metadata_file_test, 
                   config,
                   show_matrix=False):
    
    model.eval()

    print(f"Evaluating {config['model_name']} model...")

    metadata = pd.read_csv(metadata_file_test)
    data_dir = config["data_dir"]
    correct = 0
    confusion_matrix = torch.zeros(8, 8)
    test_dataset = EmoDataset(config, 
                               metadata_file_test, 
                               data_dir, 
                               transform, 
                               device, 
                               random_sample=True, 
                               model_name=model_name)


    for wav_index in tqdm(range(len(metadata))):

        filepath = metadata.filepath[wav_index]
        label = metadata.emotion[wav_index]
        label_int = class_mapping_emo[label]
        audio_path = os.path.join(data_dir, filepath)
        signal_data = test_dataset._get_item_from_path(audio_path)
        signal_data = signal_data.unsqueeze(0)

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
    ax.set_xticks(torch.arange(8))
    ax.set_yticks(torch.arange(8))
    ax.set_xticklabels(class_mapping_emo.keys())
    ax.set_yticklabels(class_mapping_emo.keys())

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(8):
        for j in range(8):
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

    # test model: emorec/20250411-163737
    # test model: emorec/20250411-164740

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
    metadata_file_test = "data/test_metadata_emo_new.csv"
    pretrained = False
    fine_tune = False

    
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

    if model_name == "MusicRecNet":
        model = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=filters, add_dropout=add_dropout).to(device)
    else:
        model = get_model(model_name, num_classes=8, pretrained=pretrained, fine_tune=fine_tune).to(device)
        model.to(device)
    
    model.load_state_dict(torch.load(os.path.join("trained", model_dir, "model.pth")))

    print(f'model name: {model_name}')

    evaluate_model(model, model_dir, metadata_file_test, config, show_matrix=True)