const car = temp1;

if (!car) {
  console.warn("Objek mobil tidak ditemukan!");
} else {
  // === Konfigurasi Dataset ===
  const components = ["Front Left Door", "Front Right Door", "Rear Left Door", "Rear Right Door", "Hood"];
  
  // === Konfigurasi Kamera 3D - LEBIH HALUS ===
  const yawStep = 15; // Step rotasi horizontal lebih halus (15°)
  const pitchStep = 10; // Step rotasi vertikal lebih halus (10°)
  const pitchRange = [-30, 30]; // Range pitch praktis
  const yawRange = [0, 360]; // Range yaw penuh 360°
  
  // === PITCH OFFSET YANG BENAR ===
  const pitchOffset = 90; // Offset +90° untuk membuat 0° = horizontal
  
  // === Recording Config untuk Single Frame ===
  const recordDuration = 1; // Durasi recording singkat (1 detik)
  
  // === Dataset Storage ===
  let datasetLabels = [];
  let imageCounter = 0;
  
  // === Track current state ===
  let currentState = [0, 0, 0, 0, 0]; // Mulai dengan semua tertutup
  
  // Generate semua kombinasi binary (2^5 = 32 kombinasi)
  function generateAllCombinations() {
    const combinations = [];
    for (let i = 0; i < 32; i++) {
      const binary = i.toString(2).padStart(5, '0');
      combinations.push(binary.split('').map(bit => parseInt(bit)));
    }
    return combinations;
  }
  
  // Generate semua kombinasi sudut kamera
  function generateCameraAngles() {
    const angles = [];
    for (let pitch = pitchRange[0]; pitch <= pitchRange[1]; pitch += pitchStep) {
      for (let yaw = yawRange[0]; yaw < yawRange[1]; yaw += yawStep) {
        angles.push({ pitch, yaw });
      }
    }
    return angles;
  }
  
  const allStates = generateAllCombinations();
  const allCameraAngles = generateCameraAngles();
  
  // === Setup CSV Header ===
  console.log("📋 CSV Header:");
  console.log("filename,front_left,front_right,rear_left,rear_right,hood,pitch,yaw");
  
  console.log(`📊 Total kombinasi state: ${allStates.length}`);
  console.log(`📊 Total sudut kamera: ${allCameraAngles.length}`);
  console.log(`📊 Estimasi total gambar: ${allStates.length * allCameraAngles.length} (1 gambar per posisi)`);
  
  // Hitung detail sudut
  const pitchCount = Math.floor((pitchRange[1] - pitchRange[0]) / pitchStep) + 1;
  const yawCount = Math.floor((yawRange[1] - yawRange[0]) / yawStep);
  console.log(`📐 Detail: ${pitchCount} pitch levels × ${yawCount} yaw positions = ${allCameraAngles.length} angles`);
  
  // === Fungsi untuk set state komponen dengan state tracking ===
  async function setCarState(targetStateVector) {
    const buttons = Array.from(document.querySelectorAll("button"));
    
    console.log(`🎯 Target state: [${targetStateVector.join(',')}]`);
    console.log(`🔄 Current state: [${currentState.join(',')}]`);
    
    for (let i = 0; i < components.length; i++) {
      const componentName = components[i];
      const targetState = targetStateVector[i];
      const currentComponentState = currentState[i];
      
      // Hanya klik jika state berubah
      if (targetState !== currentComponentState) {
        const btn = buttons.find(b => b.textContent.includes(componentName));
        if (btn) {
          btn.click();
          console.log(`🔧 ${componentName}: ${currentComponentState === 0 ? 'CLOSE → OPEN' : 'OPEN → CLOSE'}`);
          
          // Update current state
          currentState[i] = targetState;
          
          // Delay untuk animasi
          await new Promise(resolve => setTimeout(resolve, 300));
        } else {
          console.warn(`⚠️ Button untuk ${componentName} tidak ditemukan!`);
        }
      } else {
        console.log(`✓ ${componentName}: sudah dalam state ${targetState === 1 ? 'OPEN' : 'CLOSE'}`);
      }
    }
    
    // Delay tambahan untuk memastikan semua animasi selesai
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log(`✅ Final state: [${currentState.join(',')}]`);
  }
  
  // === Fungsi untuk set posisi kamera 3D ===
  function setCameraPosition(pitch, yaw) {
    // Konversi ke radian dengan offset koreksi +90°
    const correctedPitch = pitch + pitchOffset;
    const pitchRad = (correctedPitch * Math.PI) / 180;
    const yawRad = (yaw * Math.PI) / 180;
    
    // Set rotasi mobil untuk simulasi perubahan sudut pandang kamera
    if (car.rotation) {
      car.rotation.y = yawRad; // Rotasi horizontal
      car.rotation.x = pitchRad; // Rotasi vertikal dengan koreksi +90°
    }
    
    console.log(`📷 Camera Position: Pitch ${pitch}° (corrected: ${correctedPitch}°), Yaw ${yaw}°`);
  }
  
  // === Fungsi untuk record singkat dan extract 1 frame saja ===
  function recordAndExtractSingleFrame(stateVector, pitch, yaw, stateIndex, angleIndex) {
    return new Promise((resolve) => {
      const canvas = document.querySelector("canvas");
      if (!canvas) {
        console.error("❌ Canvas tidak ditemukan!");
        resolve();
        return;
      }
      
      const stream = canvas.captureStream();
      const recorder = new MediaRecorder(stream, { mimeType: "video/webm" });
      let chunks = [];
      
      recorder.ondataavailable = e => chunks.push(e.data);
      
      recorder.onstop = () => {
        console.log("🎥 Record selesai, ekstrak 1 frame...");
        
        const blob = new Blob(chunks, { type: "video/webm" });
        const video = document.createElement("video");
        video.src = URL.createObjectURL(blob);
        
        video.onloadedmetadata = () => {
          const off = document.createElement("canvas");
          off.width = video.videoWidth;
          off.height = video.videoHeight;
          const ctx = off.getContext("2d");
          
          // Set ke tengah video untuk frame terbaik
          video.currentTime = recordDuration / 2;
          
          video.onseeked = () => {
            ctx.clearRect(0, 0, off.width, off.height);
            ctx.drawImage(video, 0, 0);
            
            // Generate filename dengan info sudut
            imageCounter++;
            const filename = `img_${imageCounter.toString().padStart(4, '0')}_p${pitch}_y${yaw}.jpg`;
            
            // Download frame
            const link = document.createElement("a");
            link.href = off.toDataURL("image/jpeg", 0.9);
            link.download = filename;
            link.click();
            
            // Simpan label dengan info sudut kamera
            const labelRow = {
              filename: filename,
              front_left: stateVector[0],
              front_right: stateVector[1],
              rear_left: stateVector[2],
              rear_right: stateVector[3],
              hood: stateVector[4],
              pitch: pitch,
              yaw: yaw
            };
            
            datasetLabels.push(labelRow);
            
            console.log(`📸 ${filename},${stateVector.join(',')},${pitch},${yaw}`);
            
            // Cleanup
            URL.revokeObjectURL(video.src);
            resolve();
          };
        };
      };
      
      // Mulai recording singkat
      recorder.start();
      console.log(`⏺️ Recording 1 detik untuk state [${stateVector.join(',')}] angle [${pitch}°,${yaw}°]...`);
      
      // Stop recording setelah 1 detik
      setTimeout(() => {
        recorder.stop();
      }, recordDuration * 1000);
    });
  }
  
  // === Fungsi untuk reset ke state awal ===
  async function resetToInitialState() {
    console.log("🔄 Reset ke state awal...");
    await setCarState([0, 0, 0, 0, 0]);
  }
  
  // === Fungsi utama untuk generate dataset ===
  async function generateDataset() {
    console.log("🚀 Mulai generate dataset 3D (Record & Extract Single Frame)...");
    
    const startTime = Date.now();
    
    // Reset ke state awal
    await resetToInitialState();
    
    // Loop semua kombinasi state
    for (let stateIndex = 0; stateIndex < allStates.length; stateIndex++) {
      const stateVector = allStates[stateIndex];
      console.log(`\n🔄 State ${stateIndex + 1}/${allStates.length}: [${stateVector.join(',')}]`);
      
      // Set state komponen mobil
      await setCarState(stateVector);
      
      // Loop semua sudut kamera untuk state ini
      for (let angleIndex = 0; angleIndex < allCameraAngles.length; angleIndex++) {
        const { pitch, yaw } = allCameraAngles[angleIndex];
        
        console.log(`  📐 Angle ${angleIndex + 1}/${allCameraAngles.length}: Pitch ${pitch}°, Yaw ${yaw}°`);
        
        // Set posisi kamera
        setCameraPosition(pitch, yaw);
        
        // Delay untuk memastikan posisi kamera stabil
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Record singkat dan extract 1 frame
        await recordAndExtractSingleFrame(stateVector, pitch, yaw, stateIndex, angleIndex);
        
        // Progress indicator
        const totalCombinations = allStates.length * allCameraAngles.length;
        const currentCombination = (stateIndex * allCameraAngles.length) + angleIndex + 1;
        const percentage = ((currentCombination / totalCombinations) * 100).toFixed(1);
        
        // Estimasi waktu tersisa
        const elapsed = (Date.now() - startTime) / 1000; // detik
        const avgTimePerImage = elapsed / currentCombination;
        const remainingImages = totalCombinations - currentCombination;
        const estimatedRemaining = (avgTimePerImage * remainingImages / 60).toFixed(1); // menit
        
        console.log(`    ⏳ Progress: ${percentage}% (${currentCombination}/${totalCombinations}) | ETA: ${estimatedRemaining} min`);
        
        // Delay minimal antar capture
        await new Promise(resolve => setTimeout(resolve, 300));
      }
      
      // Delay antar state
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // Generate final CSV content
    console.log("\n✅ Dataset generation selesai!");
    
    const totalTime = ((Date.now() - startTime) / 1000 / 60).toFixed(1);
    console.log(`⏱️ Total waktu: ${totalTime} menit`);
    console.log(`📊 Total gambar: ${imageCounter}`);
    
    console.log("\n📋 Complete CSV Data:");
    console.log("filename,front_left,front_right,rear_left,rear_right,hood,pitch,yaw");
    
    datasetLabels.forEach(row => {
      console.log(`${row.filename},${row.front_left},${row.front_right},${row.rear_left},${row.rear_right},${row.hood},${row.pitch},${row.yaw}`);
    });
    
    // Download CSV file
    const csvContent = "data:text/csv;charset=utf-8," + 
      "filename,front_left,front_right,rear_left,rear_right,hood,pitch,yaw\n" +
      datasetLabels.map(row => 
        `${row.filename},${row.front_left},${row.front_right},${row.rear_left},${row.rear_right},${row.hood},${row.pitch},${row.yaw}`
      ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "labels_3d.csv");
    link.click();
    
    console.log("💾 File labels_3d.csv berhasil didownload!");
  }
  
  // === Test Functions - Expose to global ===
  window.testPitchOffset = function(offset) {
    console.log(`🔧 Testing pitch offset: ${offset}°`);
    
    const testPitch = 0;
    const correctedPitch = testPitch + offset;
    const pitchRad = (correctedPitch * Math.PI) / 180;
    
    if (car.rotation) {
      car.rotation.x = pitchRad;
      car.rotation.y = 0;
    }
    
    console.log(`📐 Pitch ${testPitch}° + offset ${offset}° = ${correctedPitch}°`);
    console.log(`   Mobil sekarang ${offset === 90 ? '✅ HORIZONTAL' : '⚠️ belum horizontal'}`);
  };
  
  // === Manual Control ===
  window.setPitch = function(angle) {
    const correctedPitch = angle + pitchOffset;
    const rad = (correctedPitch * Math.PI) / 180;
    car.rotation.x = rad;
    console.log(`📐 Manual pitch set to ${angle}° (corrected: ${correctedPitch}°)`);
  };
  
  window.resetCar = function() {
    car.rotation.x = (0 + pitchOffset) * Math.PI / 180;
    car.rotation.y = 0;
    console.log("🔄 Car position reset to horizontal");
  };
  
  // === EXPOSE GENERATE DATASET TO GLOBAL ===
  window.generateDataset = generateDataset;
  window.startDatasetGeneration = function() {
    console.log("🚀 Starting dataset generation...");
    generateDataset();
  };
  
  // === Quick Test Function ===
  window.quickTest = function() {
    console.log("🧪 Quick test - capturing 3 angles...");
    setCameraPosition(0, 0);   // Horizontal, front
    setTimeout(() => setCameraPosition(15, 90), 2000);  // Slightly up, right side
    setTimeout(() => setCameraPosition(-15, 180), 4000); // Slightly down, back
  };
  
  // === Test dulu sebelum generate dataset ===
  console.log("🔧 Testing pitch offset +90°...");
  setTimeout(() => {
    window.testPitchOffset(90);
    console.log("✅ Jika mobil sudah horizontal, dataset generation siap dijalankan");
    console.log("🎮 Available commands:");
    console.log("   generateDataset() - Mulai generate dataset (Record & Extract)");
    console.log("   quickTest() - Test 3 angle cepat");
    console.log("   setPitch(15) - Test pitch manual");
    console.log("   resetCar() - Reset ke posisi horizontal");
  }, 1000);
}