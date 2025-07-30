import os
import yaml
import warnings
import torch
import torchaudio
from transformers import Wav2Vec2Processor, WhisperProcessor
from model import *

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

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_model_for_inference(model_dir, device):

    config_path = os.path.join("trained", model_dir, "config.yaml")
    config = load_config(config_path)

    model_name = config["model_name"]
    database = config["database"]
    num_classes = len(class_mapping_emo[database])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        model = get_model(model_name, num_classes=num_classes, pretrained=False, fine_tune=False).to(device)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join("trained", model_dir, "model.pth")))

    return model

def process_input(audio_path, target_sr, n_samples, model_name, device):

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file {audio_path} does not exist.")
    
    # define processor based on model type
    if model_name in ["wav2vec2", "hubert", "wavlm"]:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    elif model_name in ["whisper-tiny", "whisper-base"]:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")

    # process wav file only

    if not audio_path.endswith('.wav'):
        raise ValueError("Only .wav files are supported for inference.")
    else:
        signal, sr = torchaudio.load(audio_path, normalize=True)
        if signal.size(0) > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        signal = signal.to(device)
        if sr != target_sr:
            signal = torchaudio.transforms.Resample(sr, target_sr).to(device)(signal)
        if signal.size(1) != n_samples:
            len_signal = signal.size(1)
            if len_signal < n_samples:
                padding = n_samples - len_signal
                signal = torch.nn.functional.pad(signal, (0, padding))
            else:
                signal = signal[:, :n_samples]
        signal = signal.squeeze(0)


        inputs = processor(
            signal,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=False
        )
        input_values = inputs.input_values.squeeze(0)
        input_values = input_values.to(device)
        input_values = input_values.unsqueeze(0)

        return input_values

def infer(model, input_data, device):
    input_data = input_data.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Emotion Recognition Inference")
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("-a", "--audio_path", type=str, required=True, help="Path to the audio file for inference")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to run the inference on (cpu or cuda)")
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.model_dir, device)
    
    config_path = os.path.join("trained", args.model_dir, "config.yaml")
    config = load_config(config_path)
    target_sr = config["target_sr"]
    n_samples = config["duration"] * target_sr
    model_name = config["model_name"]
    input_data = process_input(args.audio_path, target_sr, n_samples, model_name, device)

    output = infer(model, input_data, device)
    predicted_class = torch.argmax(output, dim=1).item()
    database = config["database"]
    class_mapping = class_mapping_emo[database]
    predicted_emotion = [emotion for emotion, idx in class_mapping.items() if idx == predicted_class][0]
    print(f"Predicted Emotion: {predicted_emotion} (Class Index: {predicted_class})")
    print(f"Output Tensor: {output}")
    print(f"Output Shape: {output.shape}")



    