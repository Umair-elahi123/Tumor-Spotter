# 🔬 Breast Cancer Detection using YOLOv8

A sophisticated AI-powered breast cancer detection system that analyzes mammogram and ultrasound images to identify suspicious regions. Built with cutting-edge deep learning technology and a modern web interface.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Training Details](#-training-details)
- [Testing with Real Images](#-testing-with-real-images)
- [Environment Variables](#-environment-variables)
- [Troubleshooting](#-troubleshooting)
- [Disclaimer](#-disclaimer)
- [License](#-license)
- [Author](#-author)

---

## ✨ Features

### 🎯 Core Functionality
- **Real-time Detection**: Instant analysis of uploaded mammogram/ultrasound images
- **High Accuracy**: 98.7% mAP@50 and 76% mAP@50-95 on validation set
- **Bounding Box Visualization**: Clear marking of suspicious regions with confidence scores
- **Detailed Analysis**: Location coordinates, size, and classification for each detection
- **One-Click Detection**: Simple upload and detect workflow

### 🎨 User Experience
- **Modern Web Interface**: Beautiful Gradio-powered UI with pink theme
- **Drag & Drop Upload**: Intuitive image upload system
- **Real-time Results**: Instant visual feedback with annotated images
- **Styled Result Cards**: Professional result display with color-coded alerts
- **Responsive Design**: Works on desktop and mobile devices

### 🔧 Technical Features
- **Anti-Overfitting Architecture**: 7 advanced techniques for model generalization
- **GPU Accelerated**: NVIDIA CUDA support for fast training and inference
- **Lightweight Model**: YOLOv8s for optimal speed-accuracy balance
- **Efficient Processing**: Sub-second detection on modern hardware
- **Automated Training**: Complete training pipeline with validation

---

## 📊 Model Performance

| Metric | Score | Description |
|--------|-------|-------------|
| **mAP@50** | **98.7%** | Primary accuracy metric |
| **mAP@50-95** | **76.0%** | Strict accuracy across IoU thresholds |
| **Precision** | **High** | Low false positive rate |
| **Recall** | **High** | Low false negative rate |
| **Training Images** | **2,271** | Diverse dataset |
| **Validation Images** | **160** | Independent validation |
| **Test Images** | **80** | Final testing set |

### Dataset Distribution
```
Total: 2,511 annotated medical images
├── Training Set   : 2,271 images (90.4%)
├── Validation Set :   160 images  (6.4%)
└── Test Set       :    80 images  (3.2%)
```

**Image Types**: Mammograms, Ultrasounds, Various angles and breast densities

---

## 🏗️ Architecture

```
┌─────────────────────┐
│   Gradio Web UI     │
│   (User Interface)  │
└──────────┬──────────┘
           │
           │ Upload Image
           ▼
┌──────────────────────┐      ┌──────────────────┐
│   YOLOv8s Model      │◄────►│  Best Weights    │
│   (Detection)        │      │  (epoch35.pt)    │
└──────────┬───────────┘      └──────────────────┘
           │
           │ Inference
           ▼
┌──────────────────────┐
│  Result Processing   │
│  (Bounding Boxes)    │
└──────────┬───────────┘
           │
           │ Styled Output
           ▼
┌──────────────────────┐
│   Visual Results     │
│   (Annotated Image)  │
└──────────────────────┘
```

---

## 🛠️ Technologies Used

### Deep Learning
- **YOLOv8s** - Ultralytics object detection model
- **PyTorch** - Deep learning framework
- **CUDA** - GPU acceleration

### Computer Vision
- **OpenCV** - Image processing
- **PIL/Pillow** - Image manipulation
- **NumPy** - Numerical operations

### Web Interface
- **Gradio 4.0+** - Modern ML web interface
- **HTML/CSS** - Custom styling
- **JavaScript** - Interactive elements

### Development Tools
- **Python 3.8+** - Core programming language
- **Roboflow** - Dataset management
- **Git** - Version control

### Training Infrastructure
- **NVIDIA GTX 1660 Ti** - GPU training
- **6GB VRAM** - Training memory
- **Anti-overfitting suite** - Dropout, augmentation, regularization

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn
- Python 3.9+
- OpenRouter API Key (FREE at https://openrouter.ai/)
- Python 3.8+ and pip
- NVIDIA GPU with CUDA support (optional, but recommended for training)
- 6GB+ VRAM for training
- Windows/Linux/macOS

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-detection-yolov8.git
cd breast-cancer-detection-yolov8
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. GPU Setup (Optional)

For CUDA-enabled GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 5. Configure Environment

Create `.env` file:

```bash
cp .env.example .env
# Edit .env and add your Roboflow API key
```

Get your free Roboflow API key at: https://app.roboflow.com/settings/api

### Running the Application

#### 1. Download Dataset

```bash
python download_dataset.py
```

This downloads 2,511 annotated images from Roboflow.

#### 2. Train the Model (Optional)

```bash
python train_breast_cancer.py
```

**Training time**: ~2-3 hours on GTX 1660 Ti

#### 3. Run the Detection App

```bash
python detect_cancer_app.py
```

The web interface will open automatically at `http://localhost:7860`

---

## 📱 Usage

### Web Interface

1. **Upload Image**
   - Drag and drop or click to select a mammogram/ultrasound image
   - Supported formats: JPG, PNG, JPEG

2. **Detect Cancer**
   - Click the "🔍 Detect Cancer" button
   - AI processes the image automatically
   - Detection uses 10% confidence threshold by default

3. **View Results**
   - **Green Card**: No cancer detected - clean results
   - **Yellow Card**: Cancer detected - detailed findings with:
     - Number of suspicious regions
     - Confidence scores
     - Location coordinates
     - Bounding box sizes

4. **Interpret Results**
   - Review annotated image with highlighted regions
   - Check confidence levels (higher = more certain)
   - **Always consult a medical professional** for diagnosis

### Key Features

- **Instant Detection**: Results in under 1 second
- **Visual Markers**: Clear bounding boxes on suspicious areas
- **Detailed Metrics**: Coordinates, sizes, and confidence scores
- **Color-Coded Results**: Easy-to-understand visual feedback

---

## 📂 Project Structure

```
breast-cancer-detection/
│
├── detect_cancer_app.py           # Gradio web application
├── train_breast_cancer.py         # Training script with anti-overfitting
├── download_dataset.py            # Dataset downloader (Roboflow)
│
├── runs/                          # Training outputs (excluded from git)
│   └── detect/
│       └── breast_cancer_v2_fixed/
│           └── weights/
│               ├── best.pt        # Best model weights
│               └── epoch35.pt     # Checkpoint weights
│
├── Cancer-Detecion-1/             # Dataset (excluded from git)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── README.md                      # This file
├── SETUP_GUIDE.md                 # GitHub deployment guide
├── GITHUB_COMMANDS.txt            # Quick reference commands
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .env.example                   # Environment variables template
└── .venv/                         # Virtual environment (excluded from git)
```

---

## 🎓 Training Details

### Hardware Used

- **GPU**: NVIDIA GeForce GTX 1660 Ti
- **VRAM**: 6GB GDDR6
- **Training Time**: ~2-3 hours for 40 epochs

### Anti-Overfitting Techniques

| Technique | Value | Purpose |
|-----------|-------|---------|
| **Dropout** | 0.3 (30%) | Randomly deactivate neurons during training |
| **Weight Decay** | 0.001 | L2 regularization to prevent large weights |
| **Early Stopping** | Patience = 10 | Stop if validation loss doesn't improve |
| **Learning Rate** | 0.0005 | Lower rate prevents overfitting |
| **Mosaic Augmentation** | 1.0 | Combines 4 images for diversity |
| **MixUp** | 0.2 | Blends two images |
| **Copy-Paste** | 0.1 | Copies objects between images |

### Data Augmentation

```python
degrees       = 15.0        # Rotation (±15°)
translate     = 0.1         # Translation (10%)
scale         = 0.5         # Scaling (50%)
shear         = 5.0         # Shear (5°)
flipud        = 0.5         # Vertical flip (50%)
fliplr        = 0.5         # Horizontal flip (50%)
hsv_h         = 0.015       # Hue augment
hsv_s         = 0.7         # Saturation augment
hsv_v         = 0.4         # Brightness augment
```

### Training Configuration

```python
epochs        = 40          # Training epochs
batch_size    = 16          # Images per batch
image_size    = 640         # Input dimensions
optimizer     = "Adam"      # Optimization algorithm
dropout       = 0.3         # Dropout rate
weight_decay  = 0.001       # L2 regularization
patience      = 10          # Early stopping
```

---

## 🧪 Testing with Real Images

You can test the model with official diagnostic mammogram images from trusted medical sources:

### Recommended Test Images

**Healthline Medical Gallery**  
🔗 [Mammogram Images - Breast Cancer Diagnostics](https://www.healthline.com/health/breast-cancer/mammogram-images-breast-cancer#implants)

This resource provides:
- Real mammogram examples
- Various breast cancer types
- Different stages of detection
- Images with breast implants
- Educational diagnostic criteria

### How to Test

1. Download sample images from the link above
2. Upload to the Gradio interface
3. Click "Detect Cancer" button
4. Review the AI's detection results
5. Compare with actual diagnostic labels

> **Note**: Always verify AI results with qualified medical professionals. This tool is for research and educational purposes only.

---

## 🔒 Environment Variables

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### Backend (.env)
```
ROBOFLOW_API_KEY=your_api_key
MODEL_PATH=runs/detect/breast_cancer_v2_fixed/weights/epoch35.pt
CONFIDENCE_THRESHOLD=0.1
GRADIO_SERVER_PORT=7860
```

---

## 🐛 Troubleshooting

### Backend won't start
- Check Python version: `python --version` (must be 3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Verify Roboflow API key in `.env`

### Model not found
- Ensure you've trained the model or downloaded pre-trained weights
- Check path: `runs/detect/breast_cancer_v2_fixed/weights/epoch35.pt`
- Run training: `python train_breast_cancer.py`

### GPU not detected
- Verify CUDA installation: `nvidia-smi`
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Check GPU drivers are up to date

### Upload fails
- Check file format (JPG, PNG, JPEG only)
- Ensure file size is reasonable (< 50MB)
- Verify file is not corrupted

---

## ⚕️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This breast cancer detection system is designed for **research and educational purposes ONLY**. It is **NOT** a substitute for professional medical diagnosis.

### Critical Points:

- ❌ **Not FDA Approved**: This tool has not been evaluated by any medical regulatory authority
- ❌ **Not Diagnostic**: Results should never be used for clinical diagnosis without professional review
- ✅ **Always Consult Doctors**: Any concerns about breast health must be evaluated by qualified medical professionals
- ✅ **Screening Tool Only**: May be used as a preliminary screening aid, never as a final diagnostic tool
- ⚠️ **False Positives/Negatives**: AI systems can make errors; professional interpretation is essential

### Recommended Use:

- Research projects
- Educational demonstrations
- Algorithm development
- Performance benchmarking
- Academic studies

**When in doubt, always seek professional medical advice.**

---

## 📜 License

MIT License - feel free to use this project for any purpose!

```
MIT License

Copyright (c) 2024 Breast Cancer Detection Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

See [LICENSE](LICENSE) file for full text.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

Built with ❤️ using YOLOv8, Gradio, and Roboflow

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⭐ Show your support

Give a ⭐️ if this project helped you!

---

## 📊 About

### Resources
- 📖 [Documentation](SETUP_GUIDE.md)
- 🐛 [Report Bug](https://github.com/yourusername/breast-cancer-detection-yolov8/issues)
- 💡 [Request Feature](https://github.com/yourusername/breast-cancer-detection-yolov8/issues)

### Stats
- ⭐ 0 stars
- 👁️ 0 watching
- 🍴 0 forks

### Languages
- Python 85%
- HTML/CSS 10%
- Other 5%

---

**Note**: This project uses free datasets and open-source models. Perfect for learning, research, or educational use!

<div align="center">

**Made with ❤️ for early breast cancer detection**

</div>