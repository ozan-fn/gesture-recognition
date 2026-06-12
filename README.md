# PSC2 - BISINDO Sign Language Recognition

Real-time Indonesian Sign Language (BISINDO) recognition using MediaPipe and LSTM.

## Project Structure

```
psc2/
├── 1_extract_features.py   # Extract keypoints from videos using MediaPipe
├── 2_train_model.py        # Train LSTM model on extracted features
├── 3_realtime_inference.py # Real-time inference with webcam
├── requirements.txt        # Python dependencies
├── bisindo_model.h5        # Trained model (generated after training)
├── class_names.pkl         # Class labels (generated after training)
├── scaler.pkl              # Feature scaler (generated after training)
├── Dataset/                # Raw video dataset
│   └── raw_video/          # Organized by class folders
└── MP_Data/                # Extracted keypoints (generated)
```

## Installation

```bash
python -m venv .venv -p 3.10
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Extract Features from Videos

Place your videos in `Dataset/raw_video/<class_name>/` then run:

```bash
python 1_extract_features.py
```

This extracts pose and hand keypoints, normalizes them relative to shoulder position, and saves sequences to `MP_Data/`.

### 2. Train Model

```bash
python 2_train_model.py
```

Trains an LSTM model on the extracted features. Outputs `bisindo_model.h5`, `class_names.pkl`, and `scaler.pkl`.

### 3. Real-time Inference

```bash
python 3_realtime_inference.py
```

Uses webcam for real-time sign language prediction.

## Features

- **Scale-invariant**: Normalizes landmarks relative to shoulder distance
- **Efficient**: Only tracks critical pose points (shoulders, elbows, wrists)
- **Hand shape**: Local hand shape normalized relative to wrist
- **Frame cropping**: Automatically crops to frames where hands are active

## Dependencies

- mediapipe==0.10.11
- opencv-python==4.9.0.80
- tensorflow==2.15.0
- numpy==1.26.4
- scikit-learn==1.4.2
- tqdm==4.66.4
