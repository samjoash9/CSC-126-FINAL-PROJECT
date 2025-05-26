from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # lightweight model

    model.train(
        data="merged_dataset/data.yaml",  # merged dataset config
        epochs=30,                        # training epochs
        imgsz=640,                       # image size
        batch=16,                       # batch size
        name="final_mc",  # experiment name

        # # # Augmentation parameters:
        # mosaic=1.0,    # apply mosaic augmentation with 100% probability
        # mixup=0.2,     # apply mixup augmentation with 20% probability
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # HSV color augmentation
        # degrees=0.2, translate=0.1, scale=0.5, shear=0.0,  # geometric augmentations
        # flipud=0.0, fliplr=0.5  # vertical flip 0%, horizontal flip 50%
    )

if __name__ == "__main__":
    main()
