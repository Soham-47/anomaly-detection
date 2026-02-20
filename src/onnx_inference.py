import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
import onnxruntime as ort
import argparse

class ONNXAnomalyDetector:
    def __init__(self, model_path, labels_path=None, conf_threshold=0.25, iou_threshold=0.45):
        # Initialize ONNX Runtime session
        # Use CPU provider by default, but can be extended for CUDA/OpenVINO
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Get input dimensions
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Default RDD2022 labels if not provided
        self.labels = ["alligator crack", "block crack", "longitudinal crack", "other corruption", "pothole", "repair", "transverse crack"]
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

        self.log_file = "onnx_anomaly_log.csv"
        self._init_logger()

    def _init_logger(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "AnomalyType", "Confidence"])

    def log_anomaly(self, anomaly_type, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, anomaly_type, confidence])

    def preprocess(self, frame):
        # Resize and pad if necessary (here simple resize)
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to [C, H, W]
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def nms(self, boxes, scores, iou_threshold):
        """Simple Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep

    def run_inference(self, frame):
        input_data = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # YOLOv8 ONNX output shape: [1, 4 + num_classes, 8400]
        output = outputs[0][0]
        output = output.T # Shape: [8400, 4 + num_classes]
        
        boxes = []
        scores = []
        class_ids = []
        
        h, w, _ = frame.shape
        
        for pred in output:
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > self.conf_threshold:
                # Center X, Center Y, Width, Height
                cx, cy, bw, bh = pred[:4]
                
                # Rescale to original frame size
                x1 = (cx - bw/2) * w / self.input_width
                y1 = (cy - bh/2) * h / self.input_height
                x2 = (cx + bw/2) * w / self.input_width
                y2 = (cy + bh/2) * h / self.input_height
                
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
                class_ids.append(class_id)
        
        if not boxes:
            return []
            
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        indices = self.nms(boxes, scores, self.iou_threshold)
        
        detections = []
        for i in indices:
            label = self.labels[class_ids[i]] if class_ids[i] < len(self.labels) else f"ID_{class_ids[i]}"
            detections.append({
                "label": label,
                "confidence": float(scores[i]),
                "box": (int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]))
            })
            
            # Log anomaly
            self.log_anomaly(label, scores[i])
            self.save_anomaly_snapshot(frame, label, detections[-1]["box"])
            
        return detections

    def save_anomaly_snapshot(self, frame, label, box):
        """Saves a snapshot of the detected anomaly"""
        if not os.path.exists("onnx_detections"):
            os.makedirs("onnx_detections")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"onnx_detections/{label}_{timestamp}.jpg"
        
        snapshot = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(snapshot, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(filename, snapshot)

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference for Road Anomaly Detection")
    parser.add_argument("--video", type=str, required=True, help="Path to dashcam video file")
    parser.add_argument("--model", type=str, default="models/best_road_anomaly.onnx", help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument("--show", action="store_true", default=True, help="Display video output")
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    detector = ONNXAnomalyDetector(args.model, conf_threshold=args.conf, iou_threshold=args.iou)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    print(f"Processing video: {args.video}")
    print("Press 'q' to exit.")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = detector.run_inference(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS Calculation
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if args.show:
            cv2.imshow("ONNX Road Anomaly Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")

if __name__ == "__main__":
    main()
