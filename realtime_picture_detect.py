import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Soldier vs Civilian Detector")

        self.model = YOLO("runs/detect/final_mc2/weights/best.pt")  # path to your trained weights

        # Canvas for image display
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Button to load image
        self.btn_load = tk.Button(root, text="Load Image", command=self.load_image)
        self.btn_load.pack()

        self.image = None
        self.img_for_display = None

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        # Read image with OpenCV
        img = cv2.imread(file_path)
        if img is None:
            print("Failed to load image.")
            return

        # Run detection
        results = self.model.predict(img)

        # Draw results on image
        annotated_img = img.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bbox coordinates and confidence
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())

                # Map class index to name
                label = "civilian" if cls == 0 else "soldier"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # green for civilian, red for soldier

                # Draw box
                cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                # Put label text
                text = f"{label} {conf:.2f}"
                cv2.putText(annotated_img, text, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert BGR OpenCV image to RGB PIL image for Tkinter
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize image to fit canvas, keeping aspect ratio
        pil_img.thumbnail((640, 480))

        self.img_for_display = ImageTk.PhotoImage(pil_img)

        # Clear previous image on canvas and show new
        self.canvas.delete("all")
        self.canvas.create_image(320, 240, image=self.img_for_display)

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectorApp(root)
    root.mainloop()
