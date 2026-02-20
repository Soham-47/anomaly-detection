from ultralytics import YOLO

def export_model(model_path, format="tflite", imgsz=320, int8=False):
    """
    Exports a YOLO model to the specified format.
    imgsz=320 is recommended for Raspberry Pi for better FPS.
    int8=True significantly increases FPS on Pi but requires a representative dataset.
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting to {format} with imgsz={imgsz}, int8={int8}...")
    
    import shutil
    import os
    path = model.export(
        format=format, 
        imgsz=imgsz, 
        int8=int8,
        data="data/yolo_format/data.yaml"
    ) 
    
    # Move to a cleaner location for the Pi
    suffix = "_int8" if int8 else "_float32"
    target_path = f"models/best_road_anomaly{suffix}.tflite"
    
    # Ultralytics typically saves to [model_name]_saved_model/folder
    if os.path.exists(path):
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".tflite"):
                    shutil.move(os.path.join(path, f), target_path)
                    print(f"Moved exported model to: {target_path}")
                    break
        else:
            shutil.move(path, target_path)
            print(f"Moved exported model to: {target_path}")
    else:
        print(f"Export path not found: {path}")

if __name__ == "__main__":
    model_path = "models/best_road_anomaly.pt" 
    
    # 1. Export standard Float32 model at 320px (good accuracy, faster on Pi)
    # export_model(model_path, int8=False)
    
    # 2. Export INT8 quantized model (Highest FPS on Pi, uses dataset for calibration)
    export_model(model_path, int8=True) 
