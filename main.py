import torch
import torchaudio
import os
import yaml
import datetime
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
# tensorboard --logdir "C:\Users\quang\Desktop\Deep Learning Project"
from data import EmoDataset
from model import *
from train import train, load_config


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/emo_config.yaml")
    parser.add_argument("-e","--epochs", type=int, default=None)
    parser.add_argument("-bs","--batch_size", type=int, default=None)
    parser.add_argument("-lr","--learning_rate", type=int, default=None)
    parser.add_argument("-v","--verbose", action="store_true", default=False)
    parser.add_argument("-t","--transform", type=str, default=None)

    args = parser.parse_args()


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load config file and set hyperparameters
    config_path = args.config
    config = load_config(config_path)

    metadata_file_train = config["metadata_file_train"]
    metadata_file_valid = config["metadata_file_valid"]
    data_dir = config["data_dir"]
    target_sr = config["target_sr"]
    duration = config["duration"]
    epochs = args.epochs if args.epochs is not None else config["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    lr = args.learning_rate if args.learning_rate is not None else config["learning_rate"]
    verbose = args.verbose if args.verbose is not None else config["verbose"]
    transform = args.transform if args.transform is not None else config["transform"]
    n_fft = config["n_fft"]
    hop_length = config["hop_length"]
    n_mels = config["n_mels"]
    n_frames = config["n_frames"]
    filters = config["filters"]
    add_dropout = config["add_dropout"]
    model_name = config["model_name"]

    if config["n_frames"] is not None:
        n_frames = config["n_frames"]
        n_samples = (n_frames - 1) * hop_length + n_fft
        duration = n_samples / target_sr
    elif config["duration"] is not None:
        duration = config["duration"]
        n_samples = duration * target_sr
        n_frames = 1 + (n_samples - n_fft) // hop_length
        
    # Define loss function, model and optimizer
    loss_fn = nn.CrossEntropyLoss()

    if model_name == "CNN_Network":
        raise NotImplementedError("CNN_Network is not implemented yet")
    elif model_name == "MusicRecNet":
        model = MusicRecNet(n_mels=n_mels, n_frames=n_frames, filters=filters, add_dropout=add_dropout).to(device)
    elif model_name == "CNN_new":
        raise NotImplementedError("CNN_new is not implemented yet")
    else:
        raise ValueError("Model not found")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)

    # Create train and validation datasets and dataloaders
    train_dataset = EmoDataset(config, metadata_file_train, data_dir, transform, device, random_sample=True, model_name=model_name)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True)
    
    valid_dataset = EmoDataset(config, metadata_file_valid, data_dir, transform, device, random_sample=True, model_name=model_name)
    valid_loader = DataLoader(dataset=valid_dataset, 
                              batch_size=batch_size, 
                              shuffle=False)
    
    # Train the model
    train(model, train_loader, valid_loader, config, loss_fn, optimizer, device, epochs, verbose)