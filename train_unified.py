from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR\dataset_unified\dataset.yaml",
        epochs=30, # 30 epochs is usually enough for a small dataset to see decent results
        imgsz=640,
        project=r"c:\Users\carol\testee\Detector_objetos_AI\DETECTOR\runs",
        name="unified_model",
        device="cpu" # Forcing CPU to avoid CUDA OOM if there's no GPU
    )

if __name__ == '__main__':
    main()
