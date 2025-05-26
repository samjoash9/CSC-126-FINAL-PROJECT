from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # lightweight model

    model.train(
        data="merged_dataset/data.yaml", 
        epochs=30,                       
        imgsz=640,                     
        batch=16,                      
        name="final_mc",  # experiment name
    )

if __name__ == "__main__":
    main()
