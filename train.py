import torch
import torchaudio
import os
import yaml
import datetime
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
# tensorboard --logdir "/home/tnguyen/Documents/Emotion_recognition/Multimodal_Emotion_Recognition_for_Geriatric_Medecine"
from torch.utils.tensorboard import SummaryWriter
from data import EmoDataset
from model import *
from callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def train(model, train_loader, valid_loader, config, loss_fn, optimizer, device, epochs, verbose=False):

    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "trained/emorec/" + date_time
    if verbose:
        print(f"Training {date_time} started")
    

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimizer, device, verbose)
        valid_loss, valid_accuracy = validate_single_epoch(model, valid_loader, loss_fn, device, verbose)

        print(f"Train Loss: {train_loss}, Validation Loss: {valid_loss}, Validation Accuracy: {valid_accuracy}")

        if epoch == 0:
            early_stopping = EarlyStopping(patience=20)
            reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5)
            tensorboard = TensorBoard(log_dir=log_dir, config=config)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            checkpoint = ModelCheckpoint(model, save_path=log_dir)

        early_stopping(valid_loss)
        reduce_lr(valid_loss, optimizer)
        tensorboard(epoch, train_loss, valid_loss)
        checkpoint(valid_loss, epoch)

        if reduce_lr.reduce_lr:
            print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if checkpoint.save:
            print("Model saved")

    tensorboard.close()

    print("Training completed")

def train_single_epoch(model, train_loader, loss_fn, optimizer, device, verbose=False):
    
    for (data, target) in tqdm(train_loader, leave=False, ncols=80):

        if len(data.size()) == 5:
            B, RCS, C, T, F = data.size()
            data = data.reshape(B*RCS, C, T, F)
            target = target.repeat_interleave(RCS)

        if verbose:
            print(f'data size: {data.size()}')

        data = data.to(device)
        target = target.to(device)
        
        prediction = model(data)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(f"Loss: {loss.item()}")
    return loss.item()

def validate_single_epoch(model, valid_loader, loss_fn, device, verbose=False):

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for (data, target) in valid_loader:

            data = data.to(device)
            target = target.to(device)

            prediction = model(data)
            if verbose:
                print(f'prediction: {prediction}')
                print(f'target: {target}')

            loss = loss_fn(prediction, target)
            # print(f'loss: {loss.item()}')
            val_loss += loss.item()

            _, predicted = torch.max(prediction, 1)
            if verbose:
                print(f'predicted final: {predicted}')
                print(f'valid predictions {predicted == target}')

            total += target.size(0)
            correct += (predicted == target).sum().item()

            if verbose:
                print(f'correct: {correct}')
                print(f'total: {total}')



    accuracy = correct / total
    val_loss /= len(valid_loader)
    # print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}")

    model.train()

    return val_loss, accuracy