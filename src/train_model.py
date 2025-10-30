# src/train_model.py

import os
from ultralytics import YOLO

# Define constants
# This path 'src/pcb_config.yaml' is correct when running from the ROOT directory.
CONFIG_FILE = 'src/pcb_config.yaml' 
MODEL_NAME = 'yolov8n.pt' # 'n' for nano (lightweight), can be 's', 'm', or 'l'

def train():
    """
    Initializes and trains the YOLOv8 object detection model.
    """
    
    # **FIXED PATH LOGIC:** We use the defined CONFIG_FILE path directly.
    config_path = CONFIG_FILE 
    
    print(f"Loading YOLO model: {MODEL_NAME}")
    
    # 1. Load a pre-trained YOLOv8 model (transfer learning)
    model = YOLO(MODEL_NAME) 

    # 2. Train the model
    print("Starting training...")
    results = model.train(
        data=config_path,
        epochs=10,         # Reduced to 10 epochs for faster testing with dummy data
        imgsz=640,
        batch=16,           # Adjust this number based on your GPU VRAM
        device='cpu',           # 0 for GPU, 'cpu' for CPU 
        name='pcb_detection_run'
    )

    print("\nTraining complete. Check the 'runs/detect/pcb_detection_run' folder for results.")

if __name__ == '__main__':
    # **FINAL PATH CHECK:** We check for the existence of the file using the full relative path.
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file {CONFIG_FILE} not found. Ensure the path is correct.")
    else:
        train()