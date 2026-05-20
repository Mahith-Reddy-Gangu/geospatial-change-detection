<div align="center">
  <img src="https://img.icons8.com/color/96/000000/satellite-in-orbit.png" alt="Logo">
  <h1>Geospatial Change Detection AI</h1>
  <p>
    <strong>A high-performance Siamese U-Net pipeline for automated building footprint and geospatial change detection.</strong>
  </p>
  <p>
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a>
  </p>
</div>

---

## 🌍 Overview

This project implements a state-of-the-art Deep Learning pipeline for detecting geospatial changes—specifically focusing on urban development and new building footprints—from satellite imagery. By comparing historical baselines (e.g., Esri) with recent captures (e.g., Google Maps), the system automatically identifies, masks, and quantifies construction changes.

Built with an emphasis on **real-time inference** and **domain shift resilience**, the system features a robust PyTorch backend served via FastAPI, accompanied by a sleek, interactive React-style frontend for easy demonstration.

## ✨ Features

- **Siamese U-Net Architecture**: Utilizes a dual-encoder Siamese U-Net to extract and compare feature maps from two temporally separated satellite images, concatenating them at the bottleneck for precise spatial localization.
- **Domain Shift Mitigation**: Incorporates automatic Histogram Matching (`scikit-image`) to normalize brightness, contrast, and color balance between different satellite sensors (e.g., Esri vs. Google), drastically reducing false positives.
- **Advanced Object Analytics**: Uses contour mapping and convex hull calculations via OpenCV to accurately estimate building footprint areas and count distinct structural changes.
- **High-Performance API**: A stateless, highly responsive FastAPI backend that handles image processing, model inference, and real-time Esri tile fetching natively.
- **Interactive Dashboard**: A beautiful, vanilla JS frontend tailored for technical demonstrations, featuring interactive maps, side-by-side image comparisons, and dynamic analytic readouts.

## 🧠 Architecture

The core of the system is the `SiameseUNet` (implemented in `kaggle_training/model.py`). 
1. **Shared Encoder**: Both the "Before" and "After" images pass through identical ResNet/VGG-style convolutional blocks, sharing weights to ensure feature extraction is consistent across both images.
2. **Feature Concatenation**: The deep feature representations are concatenated at the bottleneck.
3. **Decoder**: A standard U-Net decoder with skip connections reconstructs the spatial resolution, outputting a single-channel binary mask highlighting the changed pixels.

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/geospatial-change-detection.git
cd geospatial-change-detection
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model Weights
Place your trained PyTorch weights (`siamese_unet_weights.pth`) in the root directory of the project. 
*(Note: If weights are missing, the inference server will run in demonstration mode using untrained weights).*

## 💻 Usage

### Starting the Server

The easiest way to start the server on Windows is to use the provided batch script:
```bash
start_geochange.bat
```

Alternatively, you can run the FastAPI server directly:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Accessing the Dashboard

Once the server is running, open your web browser and navigate to:
```
http://localhost:8000
```
From the dashboard, you can:
1. **Upload Images Manually**: Upload a "Before" and "After" image to see the detected changes.
2. **Use Coordinates**: Enter a Latitude and Longitude to automatically fetch the Esri baseline and compare it against your uploaded "After" image.
3. **View Analytics**: Real-time readouts of distinct building changes and estimated pixel area.

## 📁 Project Structure

```text
├── api/
│   ├── main.py          # FastAPI application & endpoints
│   └── inference.py     # Siamese U-Net predictor & OpenCV pipeline
├── frontend/
│   ├── index.html       # Interactive Dashboard UI
│   ├── style.css        # Dashboard styling
│   └── test/            # Sample images for testing/demo
├── kaggle_training/
│   ├── model.py         # PyTorch Siamese U-Net definition
│   ├── dataset.py       # Custom PyTorch Dataset loader
│   └── train.py         # Training loop & evaluation metrics
├── start_geochange.bat  # Windows startup script
├── requirements.txt     # Python dependencies
└── README.md            # This document
```

## 📈 Model Training

To retrain the model, navigate to the `kaggle_training` directory. The codebase is optimized for execution on platforms like Kaggle or Google Colab using GPUs. 
Ensure your dataset is structured properly and modify paths in `train.py` before execution.

```bash
cd kaggle_training
python train.py
```
