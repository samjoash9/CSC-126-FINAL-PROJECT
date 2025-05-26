import tkinter as tk
from tkinter import filedialog
from threading import Thread
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO

class SoldierCivilianDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Soldier & Civilian Detection")
        self.root.geometry("800x600")

        # Load YOLOv8 model (Replace with your trained model path)
        self.model = YOLO("runs/detect/lookdown_yolov82/weights/best.pt")  # adjust path as needed

        # Video panel
        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)

        # Control buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack()

        self.load_btn = tk.Button(self.btn_frame, text="Load Video", command=self.load_video)
        self.load_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop", command=self.stop_video)
        self.stop_btn.pack(side="left", padx=10)

        self.running = False
        self.video_path = None
        self.cap = None

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.running = True
            self.cap = cv2.VideoCapture(self.video_path)
            self.process_frame()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Run YOLOv8 inference
                results = self.model(frame, verbose=False)[0]

                # Draw boxes and labels on frame
                annotated_frame = frame.copy()
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = self.model.names[cls]

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(img_pil)

                self.panel.imgtk = img_tk
                self.panel.configure(image=img_tk)

                self.root.after(16, self.process_frame)
            else:
                self.stop_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = SoldierCivilianDetector(root)
    root.mainloop()
