from ultralytics import YOLO

def main():
    # Load the base YOLOv8 model (you can use yolov8n.pt, yolov8s.pt, etc.)
    model = YOLO("yolov8n.pt")  # lightweight; change to 'yolov8s.pt' for more accuracy

    # Train the model on the merged dataset
    model.train(
        data="merged_dataset/data.yaml",  # Your merged dataset config
        epochs=30,                        # Number of training epochs
        imgsz=640,                        # Image size (default: 640x640)
        batch=16,                         # Adjust based on your GPU's capability
        name="civilian_soldier_yolov8"    # Optional experiment name
    )

if __name__ == "__main__":
    main()