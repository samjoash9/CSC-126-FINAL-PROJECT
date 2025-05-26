import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import cv2

MODEL_PATH = "runs/detect/final_mc2/weights/best.pt"

class UnifiedDetectorApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.loading_overlay = None

        # Main container
        self.container = tk.Frame(self.root, bg="white")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # Pages
        self.frames = {}
        for Page in (MenuPage, VideoPage, PicturePage):
            page = Page(self.container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.frames[Page] = page

        self.show_frame(MenuPage)

    def show_frame(self, page):
        self.frames[page].tkraise()

    def get_model(self, callback=None):
        """Load YOLO in a background thread, then call callback."""
        def _load():
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            if callback:
                self.root.after(0, callback)
            self.hide_loading()
        
        self.show_loading()
        threading.Thread(target=_load, daemon=True).start()

    def show_loading(self):
        if self.loading_overlay: return
        self.loading_overlay = tk.Toplevel(self.root)
        self.loading_overlay.overrideredirect(True)
        self.loading_overlay.attributes("-topmost", True)
        w, h = 200, 100
        x = self.root.winfo_screenwidth() // 2 - w//2
        y = self.root.winfo_screenheight() // 2 - h//2
        self.loading_overlay.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(self.loading_overlay, text="Loading Model...", font=("Helvetica",14)).pack(expand=True)

    def hide_loading(self):
        if self.loading_overlay:
            self.loading_overlay.destroy()
            self.loading_overlay = None


class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        container = tk.Frame(self, bg="white")
        container.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(container, text="Soldier vs Civilian Detector",
                 font=("Helvetica", 24), bg="white").pack(pady=40)
        tk.Button(container, text="Detect from Video", font=("Helvetica", 16),
                  width=25, height=2,
                  command=lambda: controller.show_frame(VideoPage)
                 ).pack(pady=10)
        tk.Button(container, text="Detect from Picture", font=("Helvetica", 16),
                  width=25, height=2,
                  command=lambda: controller.show_frame(PicturePage)
                 ).pack(pady=10)
        tk.Button(container, text="Exit", font=("Helvetica", 16),
                  width=25, height=2,
                  command=controller.root.quit
                 ).pack(pady=20)


class VideoPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller

        self.canvas = tk.Canvas(self, width=800, height=600, bg="white",
                                highlightthickness=2, highlightbackground="#ccc")
        self.canvas.pack(pady=20)
        self.image_id = None

        btns = tk.Frame(self, bg="white")
        btns.pack()
        tk.Button(btns, text="Load Video", command=self.load_video).pack(side="left", padx=10)
        tk.Button(btns, text="Stop", command=self.stop_video).pack(side="left", padx=10)
        tk.Button(self, text="Back to Menu",
                  command=lambda: controller.show_frame(MenuPage)
                 ).pack(pady=20)

        self.cap = None
        self.running = False

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            return

        def start_processing():
            self.cap = cv2.VideoCapture(path)
            self.running = True
            self.process_frame()

        # Ensure model is loaded first
        if self.controller.model is None:
            self.controller.get_model(callback=start_processing)
        else:
            start_processing()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self):
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return self.stop_video()

        results = self.controller.model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.controller.model.names[cls]
            color = (0,255,0) if label=="civilian" else (0,0,255)
            cv2.rectangle(frame, (x1,y1),(x2,y2), color,2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

        disp = cv2.resize(frame, (800,600))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        imgTK = ImageTk.PhotoImage(Image.fromarray(rgb))

        if self.image_id is None:
            self.image_id = self.canvas.create_image(0,0,anchor="nw",image=imgTK)
        else:
            self.canvas.itemconfig(self.image_id, image=imgTK)
        self.canvas.image = imgTK

        self.after(16, self.process_frame)


class PicturePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller

        self.canvas = tk.Canvas(self, width=800, height=600, bg="white",
                                highlightthickness=2, highlightbackground="#ccc")
        self.canvas.pack(pady=20)

        tk.Button(self, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(self, text="Back to Menu",
                  command=lambda: controller.show_frame(MenuPage)
                 ).pack(pady=10)

        self.img_disp = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg *.jpeg *.png")])
        if not path:
            return

        def do_predict():
            img = cv2.imread(path)
            results = self.controller.model.predict(img)
            for r in results:
                for box in r.boxes:
                    xy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = "civilian" if cls==0 else "soldier"
                    color = (0,255,0) if cls==0 else (0,0,255)
                    cv2.rectangle(img, (xy[0],xy[1]),(xy[2],xy[3]), color,2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (xy[0],xy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            disp = cv2.resize(img, (800,600))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self.img_disp = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(0,0,anchor="nw",image=self.img_disp)

        if self.controller.model is None:
            self.controller.get_model(callback=do_predict)
        else:
            do_predict()


if __name__ == "__main__":
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.overrideredirect(True)
    root.bind("<Escape>", lambda e: root.destroy())

    style = ttk.Style()
    style.theme_use("clam")

    app = UnifiedDetectorApp(root)
    root.mainloop()