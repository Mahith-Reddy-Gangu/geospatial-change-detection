<div align="center">
  <img src="https://img.icons8.com/color/96/000000/satellite-in-orbit.png" alt="Logo" width="80" height="80">
  <h1>🌍 Geospatial Change Detection AI</h1>
  <p>
    <strong>A Siamese U-Net computer vision pipeline for automated building footprint extraction and urban development tracking.</strong>
  </p>
  <p>
    <a href="#-executive-summary">Executive Summary</a> •
    <a href="#-technical-architecture">Architecture</a> •
    <a href="#-installation--usage">Usage</a>
  </p>
</div>

---

## 🎯 Executive Summary

This project implements a Deep Learning pipeline designed to solve a geospatial intelligence problem: **automatically detecting urban development from satellite imagery.** 

By comparing historical baselines (e.g., Esri) with recent captures, the system identifies new construction and draws semantic masks over building footprints in real-time. 

**Key Technical Achievements:**
- Designed and trained a custom **Siamese U-Net** on the LEVIR-CD dataset, achieving an **~80% F1-Score** on high-resolution validation data.
- **Illumination & Seasonal Invariance**: The model inherently learns to ignore superficial environmental differences (e.g., shadows, seasons, lighting angles), isolating only true structural development.
- Engineered a robust backend using **FastAPI** to handle live inference and real-time Esri tile fetching natively.
- Developed an interactive, state-free **React-style frontend dashboard** for seamless technical demonstrations.

---

## 🧠 Technical Architecture

The core of the system is the `SiameseUNet` (implemented in PyTorch). It is specifically tailored for comparative analysis:

1. **Shared Encoder**: Both the "Before" and "After" images pass through identical convolutional blocks with shared weights, ensuring feature extraction is perfectly consistent across both images.
2. **Concatenation**: Deep feature representations are concatenated at the bottleneck. 
3. **Decoder**: A standard U-Net decoder with skip connections reconstructs the spatial resolution, outputting a binary mask highlighting changed pixels.
4. **Loss Function**: Trained using a combined **Dice-BCE loss function** to optimize for accurate semantic segmentation.

---

## 💻 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/Mahith-Reddy-Gangu/geospatial-change-detection.git
cd geospatial-change-detection
```

### 2. Install Dependencies
*(We recommend using a Python virtual environment)*
```bash
pip install -r requirements.txt
```

### 3. Model Weights
Place your trained PyTorch weights (`siamese_unet_weights.pth`) in the root directory. *(If omitted, the server runs in a demo mode with untrained weights).*

### 4. Start the Application
**On Windows:**
```bash
start_geochange.bat
```
**Manual Startup:**
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000` to access the interactive dashboard. You can upload custom images or enter latitude/longitude coordinates to fetch live Esri satellite data for comparison.

---

## 📁 Project Structure

```text
├── api/
│   ├── main.py          # FastAPI application & endpoints
│   └── inference.py     # Siamese predictor & basic OpenCV outline mapping
├── frontend/
│   ├── index.html       # Interactive Dashboard UI
│   ├── style.css        # Dashboard styling
│   └── test/            # High-resolution sample images for testing
├── kaggle_training/
│   ├── model.py         # PyTorch Siamese U-Net definition
│   ├── dataset.py       # Custom PyTorch Dataset loader
│   └── train.py         # Training loop & evaluation metrics (Dice-BCE Lo└── requirements.txt     # Python dependencies
```
