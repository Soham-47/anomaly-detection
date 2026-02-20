# Final Project Report: Real-Time Road Anomaly Detection

## 1. Executive Summary
This project aims to develop a real-time road anomaly detection system capable of identifying potholes and various types of cracks from dashcam footage. The system is designed for edge deployment on Raspberry Pi hardware, utilizing optimized deep learning models to achieve high performance with limited computational resources.

## 2. Methodology
### 2.1 Dataset
The project utilizes the **RDD2022 (Road Damage Dataset 2022)**, which contains thousands of images with annotated road damages across multiple countries. The dataset includes 7 primary classes:
- Alligator Crack
- Block Crack
- Longitudinal Crack
- Transverse Crack
- Pothole
- Repair
- Other Corruption

### 2.2 Model Architecture
We selected **YOLOv11n**, the latest and most lightweight version of the YOLO (You Only Look Once) family. This architecture provides an excellent balance between inference speed and detection accuracy, making it ideal for real-time applications on edge devices.

### 2.3 Training Process
- **Framework**: Ultralytics YOLOv11
- **Epochs**: 50
- **Input Resolution**: 640x640 (Training), 320x320 (Inference)
- **Batch Size**: 16
- **Optimizer**: MuSGD
- **Augmentations**: Mosaic, Mixup, and standard spatial augmentations.

## 3. Results
The model was evaluated on a validation set, achieving the following metrics:
- **Precision (B)**: 0.657
- **Recall (B)**: 0.556
- **mAP50 (B)**: 0.604
- **mAP50-95 (B)**: 0.333

### Class-specific Performance:
| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| Alligator Crack | 0.592 | 0.512 | 0.543 |
| Block Crack | 0.603 | 0.502 | 0.543 |
| Longitudinal Crack | 0.658 | 0.602 | 0.660 |
| Other Corruption | 0.697 | 0.745 | 0.769 |
| Pothole | 0.634 | 0.404 | 0.476 |

## 4. Hardware Utilization and Optimization
### 4.1 Target Hardware
The deployment target is **Raspberry Pi 4/5** (ARMv8 processor).

### 4.2 Optimization Techniques
To ensure real-time performance (aiming for ≥10 FPS), several optimization techniques were applied:
1. **Model Format Conversion**: The model was converted from PyTorch (`.pt`) to **TensorFlow Lite (TFLite)** format.
2. **Quantization**: **INT8 Post-Training Quantization** was used to reduce the model size from ~10MB to ~2.5MB and speed up CPU inference.
3. **Resolution Scaling**: Downscaling input resolution to **320x320** resulted in a 4x reduction in computational load.
4. **Threaded Pipeline**: A multi-threaded video stream handling system was implemented to prevent I/O bottlenecks during frame capture.
5. **XNNPACK Delegate**: Leveraging the XNNPACK engine for optimized floating-point and fixed-point kernels on ARM CPUs.

## 5. Conclusion
The developed system demonstrates the feasibility of low-latency road anomaly detection on inexpensive edge hardware. With a mAP50 of 60.4% and optimized inference rates, it provides a robust foundation for automated road maintenance and safety monitoring systems.
