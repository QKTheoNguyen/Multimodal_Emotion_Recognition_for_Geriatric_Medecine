# Speech Emotion Recognition applied to “Humanitude” Geriatric Medicine

Welcome to the Speech Emotion Recognition project applied to Humanitude-based geriatric care.  
This project aims to characterize the quality of the Humanitude interaction between caregivers and elderly patients by analyzing emotions expressed through speech as well as other speech features. Our focus lies in understanding how emotional cues reflect the presence of “Humanitude” principles in real-life caregiving settings.

> Data was collected from numerous geriatric health establishments during natural caregiver-patient interactions. The subjects are composed of 23 female caregivers, of different levels of training Humanitude care.

---

## 🧠 Overview

The project is organized into two main phases:

### 1. Training a Speech Emotion Recognition (SER) System
- Train neural network models to classify emotions from speech signals.
- Several model architectures can be used, including CNN vision models and transformers models for audio.
- Public emotional speech datasets can be used (e.g., EmoDB, RAVDESS, CaFE) for pretraining or transfer learning.

### 2. Analyzing Emotion and Speech Features from Geriatric Care Audio
- Extract speech-based features from caregiver-patient interactions.
- Apply a trained SER model to detect emotional states over time. Note that the size of analysis windows must match the size of the training window.
- Generate structured annotations for further analysis.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/QKTheoNguyen/Multimodal_Emotion_Recognition_for_Geriatric_Medecine.git
cd Multimodal_Emotion_Recognition_for_Geriatric_Medecine
```

2. Create a virtual environment (optional but recommended):

```
python -m venv humanitude_project
source humanitude_project/bin/activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. If you have access to the data of the project, copy the data folder from `/data/ACAPELA_data_Nguyen` to your working folder

```
rsync -a /data/ACAPELA_data_Nguyen/data .
```

> Note: If you do not have access to this dataset, you can use a public emotional speech dataset (e.g., RAVDESS, EMO-DB, CaFE) and adapt the preprocessing accordingly with `preprocessing.ipynb`.

## Usage

1. Train your SER model

```
python3 main.py
```

Model checkpoints will be saved in trained/emorec/ directory under a unique name.


2. Evaluate your SER model

```
python3 evaluate.py -m emorec/name-of-your-model
```

This will generate metrics such as accuracy, confusion matrix, and optionally class-wise F1 scores.



3. Analyze audio features

Extract audio features and analyze them using the provided notebooks. Feature vectors may include emotion probabilities (emotion vector), pitch (F0), energy levels, and speech activity ratios.

Notebooks include:

- time dependent analysis (raw emotion vectors, statistical descriptors of pitch and energy level)
- time invariant analysis (statistical descriptors of emotion vectors, pitch and energy level)
- delta analysis (relative difference between statistical descriptors of one subject, during a control phase and a care phase)

## Project structure

```

├── config/ # Configuration files for model and training
├── data/ # Emotion datasets, Humanitude datasets and extracted features
├── tools/ # Utility functions (harmof0 for pitch estimation, timber_toolbox for speech features analysis)

├── callbacks.py # Custom training callbacks (e.g., early stopping, logging)
├── data.py # Dataset loading
├── evaluate.py # Evaluation script for trained models
├── inference.py # Run inference on a single audio file
├── inference_all_audio.py # Run inference on a full dataset or batch of audio files
├── model.py # Model architecture definitions
├── train.py # Training loop and model fitting
├── main.py # Main function for training models with specified config file

├── preprocessing.ipynb # Notebook for extracting and saving metadatas from emotion datasets
├── analysis_time_dependant.ipynb # Notebook for time-dependent analysis
├── analysis_time_invariant.ipynb # Notebook for time-invariant analysis
├── analysis_delta.ipynb # Notebook for analyzing relative differences

├── requirements.txt # List of required Python packages
└── README.md

```
