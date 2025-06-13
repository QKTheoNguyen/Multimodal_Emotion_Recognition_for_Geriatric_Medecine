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

def infer_long_audio(audio_path,
                     model,
                     target_sr, 
                     n_samples, 
                     overlap, 
                     model_name, 
                     device, 
                     save_path,
                     image_path):

    """
    Infer a long audio file by splitting it into overlapping segments.
    
    Args:
        audio_path (str): Path to the audio file.
        model (torch.nn.Module): Pre-trained model for inference.
        target_sr (int): Target sampling rate.
        n_samples (int): Number of samples per segment.
        overlap (float): Overlap between segments in percentage (0 to 1).
        model_name (str): Name of the model to determine the processor.
        device (torch.device): Device to run the inference on (cpu or cuda).
        save_path (str): Path to save the processed segments.
        image_path (str): Path to save image.

    Returns:
        Tensor: Processed input tensor of size (n_labels, n_segments)
    """

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
        signal = signal.squeeze(0)
  
        len_signal = int(signal.size(0))
        segment_hop = int(n_samples * (1 - overlap))
        n_segments = (len_signal - n_samples) // segment_hop + 1

        ###
        print(f'len_signal: {len_signal}, n_samples: {n_samples}, segment_hop: {segment_hop}, n_segments: {n_segments}')

        for i in range(n_segments):
            start = i * segment_hop
            end = start + n_samples
            segment = signal[start:end]
            if segment.size(0) < n_samples:
                padding = n_samples - segment.size(0)
                segment = torch.nn.functional.pad(segment, (0, padding))
            inputs = processor(
                segment,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=False
            )
            input_values = inputs.input_values.squeeze(0)
            input_values = input_values.to(device)
            input_values = input_values.unsqueeze(0)

            inferred_output = infer(model, input_values, device)
            if i == 0:
                all_outputs = inferred_output
            else:
                all_outputs = torch.cat((all_outputs, inferred_output), dim=0)

    print(f"Processed {n_segments} segments from the audio file.")

    if save_path:
        torch.save(all_outputs, save_path)

    if image_path:
        image = all_outputs.cpu().numpy()
        # from (n_samples, n_segments) to (n_segments, n_samples)
        image = image.T
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 2.5))
        plt.imshow(image, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title("Inferred Outputs")
        plt.xlabel("Time (s)")
        seconds_hop = segment_hop / target_sr
        # plt.xticks(range(0, n_segments, 10), [f"{i * seconds_hop:.2f}s" for i in range(n_segments, 10)])
        plt.xticks(range(0, n_segments, 10), [f"{int(i * seconds_hop)}" for i in range(0, n_segments, 10)])
        plt.ylabel("Classes")
        plt.yticks(range(len(class_mapping_emo[config['database']])), list(class_mapping_emo[config['database']].keys()))
        plt.tight_layout()
        plt.savefig(image_path)

    return all_outputs

def infer(model, input_data, device):
    input_data = input_data.to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output

def infer_all_audio(data_dir,                  
                    model,
                    target_sr, 
                    n_samples, 
                    overlap, 
                    model_name, 
                    device):

    our_dirs = ["Untrained", "Begginer", "Intermediate", "Expert"]

    for dir in os.listdir(data_dir):
        if dir in our_dirs:
            dir_path = os.path.join(data_dir, dir)
            for speaker in os.listdir(dir_path):
                speaker_path = os.path.join(dir_path, speaker)
                for task in os.listdir(speaker_path):
                    task_path = os.path.join(speaker_path, task)
                    for audio_file in os.listdir(task_path):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(task_path, audio_file)
                            print(f"Processing {audio_path}...")
                            image_path = audio_path.replace('.wav', '.png')
                            save_path = audio_path.replace('.wav', '_outputs.pt')
                            output_data = infer_long_audio(
                                audio_path=audio_path,
                                model=model,
                                target_sr=target_sr,
                                n_samples=n_samples,
                                overlap=overlap,
                                model_name=model_name,
                                device=device,
                                save_path=save_path,
                                image_path=image_path
                            )
                                    
    return

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Emotion Recognition Inference")
    parser.add_argument("-m", "--model_dir", type=str, required=True, help="Directory of the trained model")
    parser.add_argument("-p", "--dir_path", type=str, required=True, help="Path to the data directory for inference")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to run the inference on (cpu or cuda)")
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.model_dir, device)
    
    config_path = os.path.join("trained", args.model_dir, "config.yaml")
    config = load_config(config_path)
    target_sr = config["target_sr"]
    n_samples = config["duration"] * target_sr
    model_name = config["model_name"]
    # replace .wav with .png for image_path
    infer_all_audio(
        data_dir=args.dir_path,
        model=model,
        target_sr=target_sr,
        n_samples=n_samples,
        overlap=0.5,
        model_name=model_name,
        device=device
    )
