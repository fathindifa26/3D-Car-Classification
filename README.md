# 🚗 Car Component Detection System

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

AI-based car component detection system that uses deep learning to detect the status of doors and hood in real-time. The system uses an optimized VGG-Lite architecture for multi-label classification of 5 car components.

### 🎯 Detected Components
- **front_left**: Front left door
- **front_right**: Front right door
- **rear_left**: Rear left door
- **rear_right**: Rear right door
- **hood**: Hood

## ✨ Features

- 🚀 **Real-time Detection**: 2 FPS inference with ONNX Runtime
- 🎯 **Multi-label Classification**: Simultaneous detection of 5 components
- 🌐 **Web-based**: Direct injection into browser console
- 📱 **Responsive UI**: Real-time status panel with animation
- 🔧 **Adjustable Threshold**: Customizable detection threshold
- 📊 **Debug Mode**: Comprehensive logging and sample frame export
- 🎭 **Mock Mode**: Demo mode when model is unavailable

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
Root/
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

1. **Open a web page with video/camera stream**
2. **Open Developer Console** (`F12` or `Ctrl+Shift+I`)
3. **Copy and paste the following code:**

```javascript
// Load and inject predictor script
fetch('https://raw.githubusercontent.com/fathindifa26/3D-Car-Classification/main/predict.js')
    .then(response => response.text())
    .then(code => {
        eval(code);
        console.log('🚗 Car Predictor loaded!');
    })
    .catch(error => {
        console.error('Failed to load predictor:', error);
    });
```

### Method 2: Direct Script Injection

```javascript
// Create and inject script element
const script = document.createElement('script');
script.src = 'https://your-domain.com/path/to/predict.js';
script.onload = () => console.log('🚗 Predictor loaded!');
document.head.appendChild(script);
```

### Method 3: Local File

```javascript
// If using local file
const script = document.createElement('script');
script.src = 'file:///3D-Car-Classification/predict.js';
document.head.appendChild(script);
```

## 🎮 Usage

### Console Commands

After successful injection, use the following commands in browser console:

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

Status panel will appear at the top right with controls:

- **PAUSE/START**: Toggle prediction
- **CLOSE**: Close predictor
- **Real-time status**: Component status with confidence scores

## 📊 Model Information

### Generate Model Summary

```bash
# Generate model summary and parameter count
python -c "
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
# Preprocessing pipeline matching training
def preprocess_image(image):
        # 1. Resize to 1.1x (246×246)
        resized = transforms.Resize((246, 246))(image)
        
        # 2. Center crop to 224×224  
        cropped = transforms.CenterCrop(224)(resized)
        
        # 3. Convert to tensor and normalize
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
- **Confidence Scores**: Percentage confidence for each prediction
- **Status Indicators**: Model loaded, FPS counter, running status

### Technical Details

| Aspect | Implementation |
|--------|----------------|
| **Framework** | Vanilla JavaScript + ONNX Runtime Web |
| **UI** | Pure CSS with fixed positioning |
| **Input** | Canvas stream capture |
| **Preprocessing** | Exact match with training pipeline |
| **Output** | Sigmoid probabilities → binary threshold |

## 🔧 Development

### Local Development

```bash
# Clone repository
git clone https://github.com/fathindifa26/3D-Car-Classification.git
cd 3D-Car-Classification

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python pillow numpy matplotlib
pip install tensorboard scikit-learn

# Install ONNX tools
pip install onnx onnxruntime
```

### Training New Model

```bash
# Train model with default settings
python train.py

# Or use main.py for full pipeline
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
https://raw.githubusercontent.com/fathindifa/3D-Car-Classification/main/zoom_latest_checkpoint.onnx
```

#### Google Drive
```
https://drive.google.com/uc?export=download&id=1_-kO_rgBgRJTJ1U2eDvfqs1jeMe0YWWg
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
        async init()                           // Load ONNX runtime and model
        getCurrentPredictions()                // Get current binary predictions
        setThreshold(threshold)                // Set detection threshold (0-1)
        startPrediction()                      // Start prediction loop
        stopPrediction()                       // Stop prediction loop
        togglePrediction()                     // Toggle start/stop
        stop()                                 // Cleanup and remove UI
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
// 10% chance to save sample frames for debugging
// Saves as: sample_frame_[timestamp].png
// Uncomment in captureCurrentFrame() to enable
```

### Error Handling

```javascript
// Automatic fallback to mock predictions
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
// Check network and CORS issues
// Model will fallback to mock predictions

// Solution: Host model on GitHub or server with CORS enabled
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

This project is created for educational purposes focusing on:

- **Deep Learning**: CNN implementation for computer vision
- **Model Optimization**: Lightweight architecture design  
- **Web Deployment**: Real-time inference in browser
- **Multi-label Classification**: Simultaneous prediction of multiple outputs
- **JavaScript Integration**: AI model integration with web technology

### Academic Applications

- Computer Vision course projects
- Machine Learning deployment tutorials  
- Real-time inference demonstrations
- Web-based AI application development

## 📄 License

```
MIT License

Copyright (c) 2025 Project

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

- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add tests for new features
- Update documentation
- Ensure cross-browser compatibility

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📁 **Model Code** | [`model/vgg.py`](model/vgg.py) |
| 🌐 **Web Predictor** | [`predict.js`](predict.js) |
| 🤖 **ONNX Model** | `zoom_latest_checkpoint.onnx` |
| 📖 **Documentation** | [Wiki][def] |

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


