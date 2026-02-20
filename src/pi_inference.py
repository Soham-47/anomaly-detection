import cv2
import numpy as np
import time
import os
import csv
from datetime import datetime
from threading import Thread

# Try to use tflite_runtime for the Pi (lower overhead)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

class VideoStream:
    """Camera stream handling in a separate thread for better Performance"""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

class AnomalyDetector:
    def __init__(self, model_path, labels_path=None, conf_threshold=0.25):
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.conf_threshold = conf_threshold
        
        # Default RDD2022 labels if not provided
        self.labels = ["alligator crack", "block crack", "longitudinal crack", "other corruption", "pothole", "repair", "transverse crack"] # Update this based on your specific classes
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

        self.log_file = "anomaly_log.csv"
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
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def run_inference(self, frame):
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # YOLOv8/v11 TFLite output shape is usually [1, classes+4, 8400]
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        output = output.T # Transpose to [8400, classes+4]
        
        detections = []
        h, w, _ = frame.shape
        
        for pred in output:
            scores = pred[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > self.conf_threshold:
                # Center X, Center Y, Width, Height (normalized to input size)
                cx, cy, bw, bh = pred[:4]
                
                # Rescale to original frame size
                # Note: This assumes simple resize mapping, adjust if letterboxing was used
                x1 = int((cx - bw/2) * w / self.input_width)
                y1 = int((cy - bh/2) * h / self.input_height)
                x2 = int((cx + bw/2) * w / self.input_width)
                y2 = int((cy + bh/2) * h / self.input_height)
                
                label = self.labels[class_id] if class_id < len(self.labels) else f"ID_{class_id}"
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                })
                
                # Log and save image
                self.log_anomaly(label, confidence)
                self.save_anomaly_snapshot(frame, label, (x1, y1, x2, y2))
                
        return detections

    def save_anomaly_snapshot(self, frame, label, box):
        """Saves a snapshot of the detected anomaly"""
        if not os.path.exists("detections"):
            os.makedirs("detections")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detections/{label}_{timestamp}.jpg"
        
        # Draw box on a copy to save
        snapshot = frame.copy()
        x1, y1, x2, y2 = box
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(snapshot, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imwrite(filename, snapshot)

def main():
    # CONFIGURATION
    MODEL_PATH = "models/best_road_anomaly_int8.tflite" # Path to your exported model
    W, H = 640, 480
    
    detector = AnomalyDetector(MODEL_PATH)
    stream = VideoStream(src=0, width=W, height=H).start()
    
    print("Starting Road Anomaly Detection... Press 'q' to exit.")
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue
            
            # Run detection and measure model speed
            infer_start = time.time()
            detections = detector.run_inference(frame)
            infer_time_ms = (time.time() - infer_start) * 1000
            model_fps = 1000 / infer_time_ms if infer_time_ms > 0 else 0
            
            # Draw detections for display
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["label"]
                conf = det["confidence"]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # FPS Calculation (display/capture speed)
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            cv2.putText(frame, f"Video FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Model FPS: {model_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Road Anomaly Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
