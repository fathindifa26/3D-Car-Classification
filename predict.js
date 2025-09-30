class RealTimeCarPredictor {
  constructor() {
    this.modelLoaded = false;
    this.session = null;
    this.isRunning = false;
    this.currentPredictions = {
      front_left: 0,
      front_right: 0,
      rear_left: 0,
      rear_right: 0,
      hood: 0
    };
    
    this.labelNames = ['front_left', 'front_right', 'rear_left', 'rear_right', 'hood'];
    
    // UI elements
    this.statusPanel = null;
    
    // Prediction settings
    this.threshold = 0.5;
    this.predictionInterval = 500; // 500ms interval (2 FPS)
    this.intervalId = null;
    
    console.log("🚀 Real-time Car Predictor initialized!");
    this.init();
  }
  
  async init() {
    try {
      // Load ONNX Runtime
      await this.loadONNXRuntime();
      
      // Create UI
      this.createUI();
      
      // Start prediction loop
      this.startPrediction();
      
      console.log("✅ Real-time predictor ready!");
      
    } catch (error) {
      console.error("❌ Initialization failed:", error);
      this.showError("Failed to initialize predictor: " + error.message);
    }
  }
  
  async loadONNXRuntime() {
    return new Promise((resolve, reject) => {
      // Check if ONNX Runtime already loaded
      if (window.ort) {
        console.log("✅ ONNX Runtime already available");
        this.loadModel();
        resolve();
        return;
      }
      
      console.log("📦 Loading ONNX Runtime...");
      
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.16.3/dist/ort.min.js';
      script.onload = async () => {
        console.log("✅ ONNX Runtime loaded");
        await this.loadModel();
        resolve();
      };
      script.onerror = () => {
        reject(new Error("Failed to load ONNX Runtime"));
      };
      
      document.head.appendChild(script);
    });
  }
  
  async loadModel() {
    try {
      console.log("🔄 Loading ONNX model...");
      
      // Try multiple model URLs
      const modelUrls = [
        "https://raw.githubusercontent.com/fathindifa26/smartm2m/main/zoom_latest_checkpoint.onnx",
        "https://drive.google.com/uc?export=download&id=1_-kO_rgBgRJTJ1U2eDvfqs1jeMe0YWWg"
      ];
      
      let modelLoaded = false;
      
      for (const url of modelUrls) {
        try {
          console.log(`🔄 Trying to load from: ${url}`);
          this.session = await ort.InferenceSession.create(url);
          this.modelLoaded = true;
          modelLoaded = true;
          console.log("✅ Model loaded successfully!");
          console.log(`Input shape: ${JSON.stringify(this.session.inputNames)}`);
          console.log(`Output shape: ${JSON.stringify(this.session.outputNames)}`);
          break;
        } catch (error) {
          console.warn(`❌ Failed to load from ${url}: ${error.message}`);
        }
      }
      
      if (!modelLoaded) {
        throw new Error("All model URLs failed to load");
      }
      
    } catch (error) {
      console.error("❌ Model loading failed:", error);
      
      // Fallback: use mock predictions for demo
      console.log("🔄 Using mock predictions for demo...");
      this.modelLoaded = false;
      this.showWarning("Model not loaded. Using mock predictions for demo.");
    }
  }
  
  createUI() {
    // Remove existing panel if any
    const existing = document.getElementById('car-predictor-panel');
    if (existing) {
      existing.remove();
    }
    
    // Create status panel
    this.statusPanel = document.createElement('div');
    this.statusPanel.id = 'car-predictor-panel';
    this.statusPanel.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      width: 300px;
      background: rgba(0, 0, 0, 0.9);
      color: white;
      padding: 15px;
      border-radius: 10px;
      font-family: 'Courier New', monospace;
      font-size: 12px;
      z-index: 9999;
      border: 2px solid #00ff00;
      box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    `;
    
    // Create HTML content
    this.statusPanel.innerHTML = `
      <div style="text-align: center; margin-bottom: 10px;">
        <h3 style="margin: 0; color: #00ff00;">🚗 Car Component Status</h3>
        <small style="color: #888;">Real-time AI Prediction</small>
      </div>
      
      <div id="prediction-status">
        <div class="component-status" data-component="front_left">
          <span class="label">🚪 Front Left:</span>
          <span class="status" id="status-front_left">CLOSE</span>
          <span class="confidence" id="conf-front_left">0%</span>
        </div>
        <div class="component-status" data-component="front_right">
          <span class="label">🚪 Front Right:</span>
          <span class="status" id="status-front_right">CLOSE</span>
          <span class="confidence" id="conf-front_right">0%</span>
        </div>
        <div class="component-status" data-component="rear_left">
          <span class="label">🚪 Rear Left:</span>
          <span class="status" id="status-rear_left">CLOSE</span>
          <span class="confidence" id="conf-rear_left">0%</span>
        </div>
        <div class="component-status" data-component="rear_right">
          <span class="label">🚪 Rear Right:</span>
          <span class="status" id="status-rear_right">CLOSE</span>
          <span class="confidence" id="conf-rear_right">0%</span>
        </div>
        <div class="component-status" data-component="hood">
          <span class="label">🚗 Hood:</span>
          <span class="status" id="status-hood">CLOSE</span>
          <span class="confidence" id="conf-hood">0%</span>
        </div>
      </div>
      
      <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #333;">
        <div>
          <small>Status: <span id="predictor-status">Starting...</span></small><br>
          <small>FPS: <span id="prediction-fps">0</span></small><br>
          <small>Model: <span id="model-status">${this.modelLoaded ? '✅ Loaded' : '⚠️ Mock'}</span></small>
        </div>
        
        <div style="margin-top: 8px;">
          <button id="toggle-prediction" style="
            background: #00ff00;
            color: black;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 10px;
            margin-right: 5px;
          ">PAUSE</button>
          
          <button id="close-predictor" style="
            background: #ff0000;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 10px;
          ">CLOSE</button>
        </div>
      </div>
    `;
    
    // Add CSS for component status
    const style = document.createElement('style');
    style.textContent = `
      .component-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 5px 0;
        padding: 3px 0;
      }
      
      .component-status .label {
        flex: 1;
        font-weight: bold;
      }
      
      .component-status .status {
        flex: 0 0 60px;
        text-align: center;
        font-weight: bold;
      }
      
      .component-status .confidence {
        flex: 0 0 40px;
        text-align: right;
        font-size: 10px;
        color: #888;
      }
      
      .status.open { color: #ff4444; }
      .status.close { color: #44ff44; }
    `;
    
    document.head.appendChild(style);
    document.body.appendChild(this.statusPanel);
    
    // Add event listeners
    document.getElementById('toggle-prediction').addEventListener('click', () => {
      this.togglePrediction();
    });
    
    document.getElementById('close-predictor').addEventListener('click', () => {
      this.stop();
    });
  }
  
captureCurrentFrame() {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.querySelector('canvas');
      if (!canvas) {
        reject(new Error("Canvas not found"));
        return;
      }
      
      const stream = canvas.captureStream();
      const video = document.createElement('video');
      video.srcObject = stream;
      video.muted = true;
      
      video.onloadedmetadata = () => {
        video.play();
        
        setTimeout(() => {
          // === MATCH TRAINING PIPELINE ===
          const tempCanvas = document.createElement('canvas');
          
          // Step 1: Resize to 1.1x like training (246x246)
          const resizeSize = Math.floor(224 * 1.1); // 246
          tempCanvas.width = resizeSize;
          tempCanvas.height = resizeSize;
          const ctx = tempCanvas.getContext('2d');
          
          // Resize to 246x246 first
          ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, resizeSize, resizeSize);
          
          // Step 2: Center crop to 224x224 like training
          const cropSize = 224;
          const cropOffset = (resizeSize - cropSize) / 2; // (246-224)/2 = 11
          
          const finalCanvas = document.createElement('canvas');
          finalCanvas.width = cropSize;
          finalCanvas.height = cropSize;
          const finalCtx = finalCanvas.getContext('2d');
          
          // Center crop from 246x246 to 224x224
          finalCtx.drawImage(
            tempCanvas, 
            cropOffset, cropOffset, cropSize, cropSize,  // Source: crop from center
            0, 0, cropSize, cropSize                     // Dest: full canvas
          );
          
          // === SAVE SAMPLE FRAME FOR DEBUGGING ===
        //   if (Math.random() < 0.1) { // 10% chance to save sample
        //     const link = document.createElement('a');
        //     link.download = `sample_frame_${Date.now()}.png`;
        //     link.href = finalCanvas.toDataURL();
        //     document.body.appendChild(link);
        //     link.click();
        //     document.body.removeChild(link);
        //     console.log("📸 Sample frame saved!");
        //   }
          
          // Get image data from final cropped canvas
          const imageData = finalCtx.getImageData(0, 0, cropSize, cropSize);
          const pixels = imageData.data;
          
          // ImageNet normalization
          const imagenetMean = [0.485, 0.456, 0.406];
          const imagenetStd = [0.229, 0.224, 0.225];
          
          const tensor = new Float32Array(1 * 3 * cropSize * cropSize);
          let idx = 0;
          
          for (let c = 0; c < 3; c++) {
            for (let h = 0; h < cropSize; h++) {
              for (let w = 0; w < cropSize; w++) {
                const pixelIdx = (h * cropSize + w) * 4;
                const normalizedPixel = pixels[pixelIdx + c] / 255.0;
                const imagenetNormalized = (normalizedPixel - imagenetMean[c]) / imagenetStd[c];
                tensor[idx++] = imagenetNormalized;
              }
            }
          }
          
          // Debug stats
          const tensorMean = tensor.reduce((a, b) => a + b) / tensor.length;
          let tensorMin = tensor[0], tensorMax = tensor[0];
          for (let i = 1; i < tensor.length; i++) {
            if (tensor[i] < tensorMin) tensorMin = tensor[i];
            if (tensor[i] > tensorMax) tensorMax = tensor[i];
          }
          
          console.log("📊 TRAINING-MATCHED PIPELINE:");
          console.log("Resize:", resizeSize, "→ Crop:", cropSize);
          console.log("Tensor mean:", tensorMean.toFixed(6));
          console.log("Tensor range:", tensorMin.toFixed(3), "to", tensorMax.toFixed(3));
          
          video.srcObject = null;
          stream.getTracks().forEach(track => track.stop());
          
          resolve(tensor);
        }, 100);
      };
      
    } catch (error) {
      reject(error);
    }
  });
}
  
async predictFrame() {
  try {
    // AWAIT the Promise from captureCurrentFrame()
    const inputTensor = await this.captureCurrentFrame();
    let predictions;
    
    if (this.modelLoaded && this.session) {
      const feeds = {};
      feeds[this.session.inputNames[0]] = new ort.Tensor('float32', inputTensor, [1, 3, 224, 224]);
      
      const results = await this.session.run(feeds);
      const output = results[this.session.outputNames[0]];
      const rawOutputs = Array.from(output.data);
      
      // Try both raw and sigmoid outputs
      const sigmoidOutputs = rawOutputs.map(x => 1 / (1 + Math.exp(-x)));
      
      console.log("🔍 DEBUG INFO:");
      console.log("Raw outputs:", rawOutputs);
      console.log("Sigmoid outputs:", sigmoidOutputs);
      console.log("Max raw:", Math.max(...rawOutputs));
      console.log("Min raw:", Math.min(...rawOutputs));
      
      // Use sigmoid if raw outputs are large (likely logits)
      if (Math.max(...rawOutputs) > 5 || Math.min(...rawOutputs) < -5) {
        predictions = sigmoidOutputs;
        console.log("✅ Using sigmoid outputs (detected logits)");
      } else {
        predictions = rawOutputs;
        console.log("✅ Using raw outputs (already probabilities)");
      }
      
    } else {
      predictions = this.generateMockPredictions();
    }
    
    this.updatePredictions(predictions);
    
  } catch (error) {
    console.error("Prediction error:", error);
    document.getElementById('predictor-status').textContent = 'Error: ' + error.message;
  }
}
  
  generateMockPredictions() {
    // Generate semi-realistic mock predictions
    // Most of the time components are closed
    const mockProbs = [];
    
    for (let i = 0; i < 5; i++) {
      // 80% chance of being closed (probability < 0.5)
      if (Math.random() < 0.8) {
        mockProbs.push(Math.random() * 0.4); // 0.0 - 0.4 (closed)
      } else {
        mockProbs.push(0.5 + Math.random() * 0.5); // 0.5 - 1.0 (open)
      }
    }
    
    return mockProbs;
  }
  
  updatePredictions(probabilities) {
    for (let i = 0; i < this.labelNames.length; i++) {
      const component = this.labelNames[i];
      const prob = probabilities[i];
      const isOpen = prob > this.threshold;
      
      // Update current predictions
      this.currentPredictions[component] = isOpen ? 1 : 0;
      
      // Update UI
      const statusElement = document.getElementById(`status-${component}`);
      const confElement = document.getElementById(`conf-${component}`);
      
      if (statusElement && confElement) {
        statusElement.textContent = isOpen ? 'OPEN' : 'CLOSE';
        statusElement.className = `status ${isOpen ? 'open' : 'close'}`;
        confElement.textContent = `${(prob * 100).toFixed(0)}%`;
      }
    }
    
    // Update status
    document.getElementById('predictor-status').textContent = 'Running';
  }
  
  startPrediction() {
    if (this.isRunning) return;
    
    this.isRunning = true;
    let frameCount = 0;
    let lastFPSUpdate = Date.now();
    
    this.intervalId = setInterval(async () => {
      await this.predictFrame();
      
      // Update FPS counter
      frameCount++;
      const now = Date.now();
      if (now - lastFPSUpdate >= 1000) {
        const fps = frameCount;
        document.getElementById('prediction-fps').textContent = fps;
        frameCount = 0;
        lastFPSUpdate = now;
      }
    }, this.predictionInterval);
    
    console.log("🔄 Prediction started");
  }
  
  stopPrediction() {
    if (!this.isRunning) return;
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    
    this.isRunning = false;
    document.getElementById('predictor-status').textContent = 'Stopped';
    console.log("⏸️ Prediction stopped");
  }
  
  togglePrediction() {
    const button = document.getElementById('toggle-prediction');
    
    if (this.isRunning) {
      this.stopPrediction();
      button.textContent = 'START';
      button.style.background = '#ff4444';
    } else {
      this.startPrediction();
      button.textContent = 'PAUSE';
      button.style.background = '#00ff00';
    }
  }
  
  stop() {
    this.stopPrediction();
    
    if (this.statusPanel) {
      this.statusPanel.remove();
    }
    
    console.log("🛑 Real-time predictor stopped");
  }
  
  showError(message) {
    alert("Error: " + message);
  }
  
  showWarning(message) {
    console.warn("⚠️ " + message);
  }
  
  // Public API
  getCurrentPredictions() {
    return { ...this.currentPredictions };
  }
  
  setThreshold(newThreshold) {
    this.threshold = Math.max(0, Math.min(1, newThreshold));
    console.log(`🎯 Threshold set to: ${this.threshold}`);
  }
}

// Initialize and expose to global
window.carPredictor = new RealTimeCarPredictor();

console.log(`
🚗 Real-time Car Component Predictor Started!

📋 Available Commands:
  carPredictor.getCurrentPredictions() - Get current predictions
  carPredictor.setThreshold(0.7) - Set prediction threshold  
  carPredictor.stop() - Stop pcredictor
  
📝 Instructions:
1. Upload your .onnx model to a hosting service (GitHub, Google Drive, etc.)
2. Reload this script and provide the model URL when prompted
3. The predictor will automatically detect component changes!

⚠️  Currently using mock predictions for demo. Load real model for actual AI prediction.
`);