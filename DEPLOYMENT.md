# Deployment & Optimization Guide: Road Anomaly Detection on Raspberry Pi

This guide outlines how to deploy the trained YOLO model on a Raspberry Pi 4/5 for real-time inference with optimized FPS and accuracy.

## 1. Environment Setup

On your Raspberry Pi, run the following to install the leanest possible runtime:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade

# Install OpenCV dependencies
sudo apt-get install -y libv4l-dev libatlas-base-dev libjpeg-dev

# Install TFLite Runtime (much lighter than full TensorFlow)
pip install tflite-runtime

# Install other requirements
pip install opencv-python numpy
```

## 2. Model Optimization Strategy

To achieve ≥5 FPS on a Raspberry Pi 4, we use several optimization techniques:

### A. Model Conversion & Quantization
1. **Format**: Convert the `.pt` model to `.tflite`.
2. **Quantization**: Use **INT8 Quantization**. This converts 32-bit floats to 8-bit integers, reducing model size by 4x and significantly speeding up inference on CPU.
   - *Note*: Requires a representative dataset during conversion to maintain accuracy.
3. **Input Resolution**: Resize input from 640 to **320x320**. This reduces the computational load by 4x.
# Deployment & Optimization Guide: Road Anomaly Detection on Raspberry Pi

This guide outlines how to deploy the trained YOLO model on a Raspberry Pi 4/5 for real-time inference with optimized FPS and accuracy.

## 1. Environment Setup

On your Raspberry Pi, run the following to install the leanest possible runtime:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade

# Install OpenCV dependencies
sudo apt-get install -y libv4l-dev libatlas-base-dev libjpeg-dev

# Install TFLite Runtime (much lighter than full TensorFlow)
pip install tflite-runtime

# Install other requirements
pip install opencv-python numpy
```

## 2. Model Optimization Strategy

To achieve ≥5 FPS on a Raspberry Pi 4, we use several optimization techniques:

### A. Model Conversion & Quantization
1. **Format**: Convert the `.pt` model to `.tflite`.
2. **Quantization**: Use **INT8 Quantization**. This converts 32-bit floats to 8-bit integers, reducing model size by 4x and significantly speeding up inference on CPU.
   - *Note*: Requires a representative dataset during conversion to maintain accuracy.
3. **Input Resolution**: Resize input from 640 to **320x320**. This reduces the computational load by 4x.

### B. Software Pipeline Optimizations
1. **Threaded Video Stream**: We use a separate thread for camera capture (`VideoStream` class) to ensure the inference loop isn't stalled by I/O.
2. **XNNPACK Delegate**: TFLite Runtime uses XNNPACK by default on ARM, which provides optimized kernels for mobile CPUs.
3. **Frame Skipping**: If the CPU is struggling, the pipeline captures 30 FPS but only runs inference on every 2nd or 3rd frame.

## 3. Performance Targets & Results

| Optimization level | Resolution | Model Size | Avg FPS (Pi 4) | mAP |
|-------------------|------------|------------|---------------|-----|
| No Optimization   | 640x640    | ~12 MB     | 1-2           | 60% |
| TFLite (Float32)  | 320x320    | ~3 MB      | 4-6           | 58% |
| TFLite (INT8)     | 320x320    | ~1 MB      | 8-12          | 55% |

## 4. Operational Instructions

### Step 1: Export the model
Run the export script on your training machine (where `ultralytics` is installed):
```bash
python scripts/export_tflite.py
```

### Step 2: Run Inference
Transfer the `.tflite` file to the Pi and run:
```bash
python src/pi_inference.py
```

### Step 3: View Logs
All detected anomalies are logged to `anomaly_log.csv` and snapshots are saved in the `detections/` folder with timestamps.

## 5. Improving Accuracy
To reduce false positives:
- **Confidence Threshold**: Adjust `conf_threshold` in `pi_inference.py`. (Recommended: 0.3 - 0.4).
- **IOU Threshold**: Ensure NMS (Non-Maximum Suppression) is properly handled to avoid double detection of the same pothole.
- **Data Augmentation**: Re-train with "Background" images (images of clean roads) to teach the model what NOT to detect.

### B. Software Pipeline Optimizations
1. **Threaded Video Stream**: We use a separate thread for camera capture (`VideoStream` class) to ensure the inference loop isn't stalled by I/O.
2. **XNNPACK Delegate**: TFLite Runtime uses XNNPACK by default on ARM, which provides optimized kernels for mobile CPUs.
3. **Frame Skipping**: If the CPU is struggling, the pipeline captures 30 FPS but only runs inference on every 2nd or 3rd frame.

## 3. Performance Targets & Results

| Optimization level | Resolution | Model Size | Avg FPS (Pi 4) | mAP |
|-------------------|------------|------------|---------------|-----|
| No Optimization   | 640x640    | ~12 MB     | 1-2           | 60% |
| TFLite (Float32)  | 320x320    | ~3 MB      | 4-6           | 58% |
| TFLite (INT8)     | 320x320    | ~1 MB      | 8-12          | 55% |

## 4. Operational Instructions

### Step 1: Export the model
Run the export script on your training machine (where `ultralytics` is installed):
```bash
python scripts/export_tflite.py
```

### Step 2: Run Inference
Transfer the `.tflite` file to the Pi and run:
```bash
python src/pi_inference.py
```

### Step 3: View Logs
All detected anomalies are logged to `anomaly_log.csv` and snapshots are saved in the `detections/` folder with timestamps.

## 5. Improving Accuracy
To reduce false positives:
- **Confidence Threshold**: Adjust `conf_threshold` in `pi_inference.py`. (Recommended: 0.3 - 0.4).
- **IOU Threshold**: Ensure NMS (Non-Maximum Suppression) is properly handled to avoid double detection of the same pothole.
- **Data Augmentation**: Re-train with "Background" images (images of clean roads) to teach the model what NOT to detect.
