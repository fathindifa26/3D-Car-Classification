# 🚗 SmartM2M Car Component Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Real-time AI-powered car component detection system using VGG-Lite architecture for multi-label classification of doors and hood status.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Information](#model-information)
- [Web Predictor](#web-predictor)
- [Development](#development)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

SmartM2M adalah sistem deteksi komponen mobil berbasis AI yang menggunakan deep learning untuk mendeteksi status pintu dan kap mesin secara real-time. Sistem ini menggunakan arsitektur VGG-Lite yang dioptimalkan untuk klasifikasi multi-label pada 5 komponen mobil.

### 🎯 Detected Components
- **front_left**: Pintu depan kiri
- **front_right**: Pintu depan kanan  
- **rear_left**: Pintu belakang kiri
- **rear_right**: Pintu belakang kanan
- **hood**: Kap mesin

## ✨ Features

- 🚀 **Real-time Detection**: 2 FPS inference dengan ONNX Runtime
- 🎯 **Multi-label Classification**: Deteksi simultan 5 komponen
- 🌐 **Web-based**: Inject langsung ke browser console
- 📱 **Responsive UI**: Status panel real-time dengan animasi
- 🔧 **Adjustable Threshold**: Threshold detection dapat disesuaikan
- 📊 **Debug Mode**: Comprehensive logging dan sample frame export
- 🎭 **Mock Mode**: Demo mode ketika model tidak tersedia

## 🏗️ Architecture

### Model Architecture (VGG-Lite)
```
Input: 224×224×3 RGB Image
│
├── Conv(32, 3×3) → ReLU → BatchNorm → MaxPool(2×2)  [224→112]
├── Conv(64, 3×3) → ReLU → BatchNorm → MaxPool(2×2)  [112→56]  
├── Conv(128, 3×3) → ReLU → BatchNorm → MaxPool(2×2) [56→28]
├── Conv(256, 3×3) → ReLU → BatchNorm               [28×28]
├── GlobalAveragePool                                [1×1]
├── FC(256→128) → ReLU → Dropout(0.3)
├── FC(128→64) → ReLU → Dropout(0.3)
└── FC(64→5) → Sigmoid
```

### Key Specifications
| Metric | Value |
|--------|-------|
| **Parameters** | ~140K |
| **Input Size** | 224×224×3 |
| **Output** | 5 sigmoid probabilities |
| **FLOPs** | ~50M |
| **Model Size** | ~560KB (ONNX) |

## 📁 Project Structure

```
SMARTM2M/
├── model/
│   └── vgg.py                          # VGG-Lite PyTorch implementation
├── predict.js                          # Real-time web predictor
├── zoom_latest_checkpoint.onnx         # Trained ONNX model
├── README.md                           # This file
├── train.py                            # Training script
├── main.py                             # Main training pipeline
├── config/
│   └── config.py                       # Configuration settings
├── data/
│   └── dataloader.py                   # Data loading utilities
├── dataset/                            # Training images
└── experiments/                        # Training logs and checkpoints
```

## 🚀 Quick Start

### Method 1: Browser Console Injection (Recommended)

1. **Buka halaman web dengan video/camera stream**
2. **Buka Developer Console** (`F12` atau `Ctrl+Shift+I`)
3. **Copy dan paste kode berikut:**

```javascript
// Load dan inject predictor script
fetch('https://raw.githubusercontent.com/[username]/SMARTM2M/main/predict.js')
  .then(response => response.text())
  .then(code => {
    eval(code);
    console.log('🚗 SmartM2M Car Predictor loaded!');
  })
  .catch(error => {
    console.error('Failed to load predictor:', error);
  });
```

### Method 2: Direct Script Injection

```javascript
// Create dan inject script element
const script = document.createElement('script');
script.src = 'https://your-domain.com/path/to/predict.js';
script.onload = () => console.log('🚗 Predictor loaded!');
document.head.appendChild(script);
```

### Method 3: Local File

```javascript
// Jika file local
const script = document.createElement('script');
script.src = 'file:///c:/Users/Finshot/DIFA/SMARTM2M/predict.js';
document.head.appendChild(script);
```

## 🎮 Usage

### Console Commands

Setelah inject berhasil, gunakan command berikut di browser console:

```javascript
// Get current predictions
carPredictor.getCurrentPredictions();
// Output: {front_left: 0, front_right: 1, rear_left: 0, rear_right: 0, hood: 0}

// Set detection threshold (0.0 - 1.0)
carPredictor.setThreshold(0.7);  // Default: 0.5

// Stop predictor
carPredictor.stop();

// Check if running
console.log('Status:', carPredictor.isRunning ? 'Running' : 'Stopped');

// Get model info
console.log('Model loaded:', carPredictor.modelLoaded);
```

### UI Controls

Status panel akan muncul di kanan atas dengan kontrol:

- **PAUSE/START**: Toggle prediction
- **CLOSE**: Tutup predictor
- **Real-time status**: Component status dengan confidence scores

## 📊 Model Information

### Generate Model Summary

```bash
# Generate model summary dan parameter count
python -c "
import sys
sys.path.append('c:/Users/Finshot/DIFA/SMARTM2M')
from model.vgg import create_model, model_summary
model = create_model(num_classes=5, dropout_rate=0.3)
model_summary(model, input_size=(3, 224, 224))
print(f'Total parameters: {model.count_parameters():,}')
"
```

### Expected Output
```
======================================================================
Layer (type)                     Output Shape              Param #
======================================================================
Conv2d-1                        [1, 32, 224, 224]         864
ReLU-2                          [1, 32, 224, 224]         0
BatchNorm2d-3                   [1, 32, 224, 224]         64
MaxPool2d-4                     [1, 32, 112, 112]         0
...
======================================================================
Total params: 140,229
Trainable params: 140,229
Non-trainable params: 0
======================================================================
```

### Training Pipeline Matching

```python
# Preprocessing pipeline yang sama dengan training
def preprocess_image(image):
    # 1. Resize to 1.1x (246×246)
    resized = transforms.Resize((246, 246))(image)
    
    # 2. Center crop to 224×224  
    cropped = transforms.CenterCrop(224)(resized)
    
    # 3. Convert to tensor dan normalize
    tensor = transforms.ToTensor()(cropped)
    normalized = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )(tensor)
    
    return normalized
```

## 🌐 Web Predictor

### Real-time Status Panel

```
🚗 Car Component Status
Real-time AI Prediction

🚪 Front Left:    CLOSE  85%
🚪 Front Right:   OPEN   92%  ⚠️
🚪 Rear Left:     CLOSE  76%
🚪 Rear Right:    CLOSE  88%
🚗 Hood:          CLOSE  94%

Status: Running | FPS: 2 | Model: ✅ Loaded
[PAUSE] [CLOSE]
```

### Features

- **Real-time Updates**: 500ms interval (2 FPS)
- **Color Coding**: 
  - 🟢 Green: CLOSE (normal)
  - 🔴 Red: OPEN (detected)
- **Confidence Scores**: Percentage confidence untuk setiap prediksi
- **Status Indicators**: Model loaded, FPS counter, running status

### Technical Details

| Aspect | Implementation |
|--------|----------------|
| **Framework** | Vanilla JavaScript + ONNX Runtime Web |
| **UI** | Pure CSS dengan fixed positioning |
| **Input** | Canvas stream capture |
| **Preprocessing** | Exact match dengan training pipeline |
| **Output** | Sigmoid probabilities → binary threshold |

## 🔧 Development

### Local Development

```bash
# Clone repository
git clone https://github.com/[username]/SMARTM2M.git
cd SMARTM2M

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy matplotlib
pip install tensorboard scikit-learn

# Install ONNX tools
pip install onnx onnxruntime
```

### Training New Model

```bash
# Train model dengan default settings
python train.py

# Atau gunakan main.py untuk full pipeline
python main.py
```

### Export to ONNX

```python
import torch
from model.vgg import create_model

# Load trained model
model = create_model(num_classes=5)
model.load_state_dict(torch.load('BEST/best_model.pth'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, 
    dummy_input, 
    "zoom_latest_checkpoint.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}},
    opset_version=11
)

print("✅ Model exported to ONNX format")
```

### Model Hosting Options

#### GitHub Raw (Recommended)
```
https://raw.githubusercontent.com/[username]/SMARTM2M/main/zoom_latest_checkpoint.onnx
```

#### Google Drive
```
https://drive.google.com/uc?export=download&id=[file_id]
```

#### Custom Server
```javascript
// Update model URLs in predict.js
const modelUrls = [
    "https://your-server.com/zoom_latest_checkpoint.onnx",
    "https://backup-server.com/zoom_latest_checkpoint.onnx"
];
```

## 📖 API Reference

### RealTimeCarPredictor Class

```javascript
class RealTimeCarPredictor {
    constructor()                           // Initialize predictor
    async init()                           // Load ONNX runtime dan model
    getCurrentPredictions()                // Get current binary predictions
    setThreshold(threshold)                // Set detection threshold (0-1)
    startPrediction()                      // Start prediction loop
    stopPrediction()                       // Stop prediction loop
    togglePrediction()                     // Toggle start/stop
    stop()                                 // Cleanup dan remove UI
}
```

### Public Methods

```javascript
// Get predictions object
const predictions = carPredictor.getCurrentPredictions();
// Returns: {front_left: 0, front_right: 1, rear_left: 0, rear_right: 0, hood: 0}

// Set custom threshold
carPredictor.setThreshold(0.3);  // Lower threshold = more sensitive

// Check running status
console.log(carPredictor.isRunning);  // true/false

// Check model status
console.log(carPredictor.modelLoaded); // true/false
```

## 🔍 Debug Features

### Console Logging

```javascript
// Debug information automatically logged:
console.log("🔍 DEBUG INFO:");
console.log("Raw outputs:", rawOutputs);        // Model raw logits
console.log("Sigmoid outputs:", sigmoidOutputs); // After sigmoid activation  
console.log("Tensor stats:", tensorStats);       // Input preprocessing stats
console.log("Max confidence:", Math.max(...predictions));
```

### Sample Frame Export

```javascript
// 10% chance untuk save sample frames untuk debugging
// Saves as: sample_frame_[timestamp].png
// Uncomment di captureCurrentFrame() untuk enable
```

### Error Handling

```javascript
// Automatic fallback ke mock predictions
if (!modelLoaded) {
    console.log("🔄 Using mock predictions for demo...");
    predictions = generateMockPredictions();
}
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "Canvas not found"
```javascript
// Check if canvas element exists
const canvas = document.querySelector('canvas');
if (!canvas) {
    console.error("No canvas element found on page");
}

// Solution: Navigate to page with video/canvas element
```

#### 2. Model loading fails
```javascript
// Check network dan CORS issues
// Model akan fallback ke mock predictions

// Solution: Host model di GitHub atau server dengan CORS enabled
```

#### 3. Low accuracy/false positives
```javascript
// Adjust threshold
carPredictor.setThreshold(0.7); // Higher threshold = less sensitive
carPredictor.setThreshold(0.3); // Lower threshold = more sensitive
```

#### 4. Performance issues
```javascript
// Reduce prediction frequency
carPredictor.predictionInterval = 1000; // 1 second intervals instead of 500ms
```

#### 5. UI conflicts
```javascript
// Remove existing panel manually
document.getElementById('car-predictor-panel')?.remove();

// Restart predictor
window.carPredictor = new RealTimeCarPredictor();
```

### Browser Compatibility

| Browser | Status | Notes |
|---------|--------|-------|
| Chrome 90+ | ✅ Fully supported | Recommended |
| Firefox 88+ | ✅ Fully supported | - |
| Safari 14+ | ⚠️ Limited | ONNX Runtime issues |
| Edge 90+ | ✅ Fully supported | - |

## 🎓 Educational Purpose

### Learning Objectives

Proyek ini dibuat untuk tujuan edukasi dengan fokus pada:

- **Deep Learning**: Implementation CNN untuk computer vision
- **Model Optimization**: Lightweight architecture design  
- **Web Deployment**: Real-time inference di browser
- **Multi-label Classification**: Simultaneous prediction multiple outputs
- **JavaScript Integration**: AI model integration dengan web technology

### Academic Applications

- Computer Vision course projects
- Machine Learning deployment tutorials  
- Real-time inference demonstrations
- Web-based AI application development

## 📄 License

```
MIT License

Copyright (c) 2025 SmartM2M Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🤝 Contributing

### How to Contribute

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

### Development Guidelines

- Follow PEP 8 untuk Python code
- Use meaningful commit messages
- Add tests untuk new features
- Update documentation
- Ensure cross-browser compatibility

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/[username]/SMARTM2M/issues)
- **Discussions**: [GitHub Discussions](https://github.com/[username]/SMARTM2M/discussions)
- **Email**: [your-email@domain.com](mailto:your-email@domain.com)

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📁 **Model Code** | [`model/vgg.py`](model/vgg.py) |
| 🌐 **Web Predictor** | [`predict.js`](predict.js) |
| 🤖 **ONNX Model** | `zoom_latest_checkpoint.onnx` |
| 🐛 **Report Issues** | [GitHub Issues](https://github.com/[username]/SMARTM2M/issues) |
| 📖 **Documentation** | [Wiki](https://github.com/[username]/SMARTM2M/wiki) |

## 💡 Pro Tips

### For Developers
```javascript
// Advanced threshold tuning
carPredictor.setThreshold(0.3);  // Sensitive detection  
carPredictor.setThreshold(0.7);  // Conservative detection

// Performance monitoring
console.time('prediction');
await carPredictor.predictFrame();
console.timeEnd('prediction');
```

### For Researchers
```python
# Feature visualization
features = model.get_feature_maps(input_tensor)
for layer_name, feature_map in features.items():
    print(f"{layer_name}: {feature_map.shape}")
```

### For Production
```javascript
// Error handling
try {
    const predictions = carPredictor.getCurrentPredictions();
    // Handle predictions
} catch (error) {
    console.error('Prediction failed:', error);
    // Fallback logic
}
```

---

<div align="center">

### 🌟 Star this repository if you find it helpful!

**Made with ❤️ for the AI community**

</div>
