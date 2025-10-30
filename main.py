#!/usr/bin/env python3
"""main.py

Single entry-point to train or run predictions for the PCB defect detection project.

Usage examples:
  python main.py train --epochs 20
  python main.py predict --model runs/detect/pcb_detection_run/weights/best.pt --source data/images/test

This file wraps the functionality from src/train_model.py and src/predict_defects.py
and exposes a small CLI using argparse.
"""

import argparse
import os
import sys
from typing import Optional


DEFAULT_CONFIG = 'src/pcb_config.yaml'
DEFAULT_MODEL_NAME = 'yolov8n.pt'
DEFAULT_RUN_NAME = 'pcb_detection_run'
DEFAULT_PREDICTION_NAME = 'pcb_predictions'
DEFAULT_MODEL_PATH = os.path.join('runs', 'detect', DEFAULT_RUN_NAME, 'weights', 'best.pt')


def train(config: str = DEFAULT_CONFIG,
          model_name: str = DEFAULT_MODEL_NAME,
          epochs: int = 10,
          imgsz: int = 640,
          batch: int = 16,
          device: str = 'cpu',
          run_name: str = DEFAULT_RUN_NAME):
    """Train a model with Ultralytics YOLOv8.

    This function imports ultralytics at runtime so running the file without the
    package installed won't fail until training is actually invoked.
    """

    # Basic path checks
    if not os.path.exists(config):
        print(f"Config file not found: {config}")
        return

    print(f"Training using config: {config}")
    print(f"Model (base): {model_name}, epochs: {epochs}, imgsz: {imgsz}, batch: {batch}, device: {device}")

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Failed to import ultralytics. Install it with 'pip install ultralytics'.")
        print("Error:", e)
        return

    # Initialize model and train
    model = YOLO(model_name)
    model.train(
        data=config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=run_name
    )

    print(f"Training finished. Check runs/detect/{run_name} for outputs.")


def predict(model_path: str = DEFAULT_MODEL_PATH,
            source: str = os.path.join('data', 'images', 'test'),
            conf: float = 0.5,
            save_name: str = DEFAULT_PREDICTION_NAME,
            device: Optional[str] = None):
    """Run prediction using a trained model (Ultralytics YOLOv8).

    The function will print a short summary and save visual prediction outputs
    to the runs/detect/<save_name> folder.
    """

    if not os.path.exists(model_path):
        print(f"Trained model not found at: {model_path}")
        return

    if not os.path.exists(source):
        print(f"Source path for prediction not found: {source}")
        return

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Failed to import ultralytics. Install it with 'pip install ultralytics'.")
        print("Error:", e)
        return

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    predict_kwargs = dict(source=source, save=True, conf=conf, name=save_name)
    if device is not None:
        predict_kwargs['device'] = device

    print(f"Running prediction on: {source} (conf={conf})")
    results = model.predict(**predict_kwargs)

    print("\n--- Prediction Summary ---")
    try:
        names = model.names
    except Exception:
        names = {i: f'Class_{i}' for i in range(20)}

    for res in results:
        base = os.path.basename(getattr(res, 'path', 'unknown'))
        boxes = getattr(res, 'boxes', None)
        num = len(boxes) if boxes is not None else 0
        print(f"Image: {base} -> {num} boxes")
        if num > 0:
            for box in boxes:
                cls_idx = int(box.cls[0]) if hasattr(box, 'cls') else None
                conf_score = float(box.conf[0]) if hasattr(box, 'conf') else None
                cls_name = names.get(cls_idx, 'Unknown') if cls_idx is not None else 'Unknown'
                print(f"  - {cls_name} ({conf_score:.2f})")

    print(f"Visual outputs saved to runs/detect/{save_name}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='PCB defect detection - train or predict')
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='Train a model')
    t.add_argument('--config', default=DEFAULT_CONFIG)
    t.add_argument('--model-name', default=DEFAULT_MODEL_NAME)
    t.add_argument('--epochs', type=int, default=10)
    t.add_argument('--imgsz', type=int, default=640)
    t.add_argument('--batch', type=int, default=16)
    t.add_argument('--device', default='cpu')
    t.add_argument('--run-name', default=DEFAULT_RUN_NAME)

    pr = sub.add_parser('predict', help='Run prediction using a trained model')
    pr.add_argument('--model', default=DEFAULT_MODEL_PATH)
    pr.add_argument('--source', default=os.path.join('data', 'images', 'test'))
    pr.add_argument('--conf', type=float, default=0.5)
    pr.add_argument('--save-name', default=DEFAULT_PREDICTION_NAME)
    pr.add_argument('--device', default=None)

    return p


def main(argv=None):
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == 'train':
        train(config=args.config,
              model_name=args.model_name,
              epochs=args.epochs,
              imgsz=args.imgsz,
              batch=args.batch,
              device=args.device,
              run_name=args.run_name)
    elif args.cmd == 'predict':
        predict(model_path=args.model,
                source=args.source,
                conf=args.conf,
                save_name=args.save_name,
                device=args.device)


if __name__ == '__main__':
    main()
