# Real-Time Road Anomaly Detection (Raspberry Pi Edge AI)

This project implements an edge AI application that processes dashcam footage in real-time to detect and log road anomalies such as potholes, cracks, and obstacles. It is optimized for high-performance inference on Raspberry Pi 4/5 hardware.

---

## 🛠 Project Components

### 1. Core Application
- **`src/pi_inference.py`**: The main deployment script.
  - Implements a **threaded video pipeline** for smooth frame capture.
  - Uses **TFLite Runtime** for low-latency inference on ARM CPUs.
  - Features **Automated Logging**: Saves detected anomalies to `anomaly_log.csv`.
  - Features **Evidence Capture**: Automatically saves Bounding Box snapshots to the `detections/` folder.
  - Custom Class Support: Configured for 7 anomaly classes (Cracks, Potholes, Repairs, etc.).

### 2. Optimization Utilities
- **`scripts/export_tflite.py`**: Converts training weights (`.pt`) to optimized edge formats.
  - Exports to **TFLite Float32** (Standard) and includes logic for **INT8 Quantization**.
  - Scales input resolution to **320x320** for a ~4x speed boost on edge hardware.
  - Automatically manages file organization in the `models/` folder.

### 3. Documentation
- **`DEPLOYMENT.md`**: A detailed guide on setting up the Raspberry Pi environment, installing light-weight runtimes (`tflite-runtime`), and fine-tuning Performance/Accuracy trade-offs.

---

## 📂 Project Structure

```bash
anomaly-detection/
├── data/               # Training datasets and configuration
├── detections/         # (Auto-generated) Snapshots of detected anomalies
├── models/             # Optimized model files
│   ├── best_road_anomaly.pt         # Original training weights
│   └── best_road_anomaly_float32.tflite # DEPLOY THIS to Pi
├── scripts/            # Optimization and data scripts
│   ├── export_tflite.py     # PT -> TFLite conversion script
│   ├── sync_labels.py       # Data preparation utility
│   └── ...                  # Other helper scripts
├── src/                # Source code for inference
│   ├── pi_inference.py      # MAIN INFERENCE SCRIPT
│   └── inference.ipynb      # Development/Testing notebook
├── DEPLOYMENT.md       # Pi setup and optimization guide
├── README.md           # Project overview (this file)
└── anomaly_log.csv     # (Auto-generated) Detection history
```

---

## 🚀 Quick Start (on Raspberry Pi)

1. **Install Dependencies**:
   ```bash
   pip install tflite-runtime opencv-python numpy
   ```

2. **Run Inference**:
   ```bash
   python src/pi_inference.py
   ```

3. **View Results**:
   - Watch the live feed with detection boxes.
   - Check `anomaly_log.csv` for timestamped entries.
   - Review `detections/` for captured JPEG evidence of road damage.

---

## 📈 Optimization Stats
- **Target Resolution**: 320x320
- **Target FPS**: ≥ 5-10 FPS (on Pi 4)
- **Model Size**: ~10MB (Float32) / ~2.5MB (INT8)
- **Classes**: Alligator Crack, Box Crack, Longitudinal Crack, Other Corruption, Pothole, Repair, Transverse Crack.
