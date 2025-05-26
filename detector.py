import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import cv2
from datetime import datetime

MODEL_PATH = "runs/detect/final_mc2/weights/best.pt"

DEFAULT_FONT = ("Helvetica Neue", 14)
TITLE_FONT = ("Helvetica Neue", 24, "bold")

WINDOW_BG = "#41436A"
BUTTON_BG = "#984063"
BUTTON_HOVER_BG = "#F64668"
BUTTON_ACTIVE_BG = "#F64668"
BUTTON_FG = "#FE9677"
BUTTON_RADIUS = 15

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, width=200, height=50,
                 radius=BUTTON_RADIUS, bg=BUTTON_BG, fg=BUTTON_FG, font=DEFAULT_FONT):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent["bg"])
        self.command = command
        self.radius = radius
        self.bg = bg
        self.fg = fg
        self.font = font
        self.width = width
        self.height = height
        self.is_pressed = False

        self.text = text

        self.create_rounded_rect(0, 0, width, height, radius, fill=bg, outline="#ccc")
        self.text_id = self.create_text(width//2, height//2, text=text, fill=fg, font=font)

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.tag_bind(self.text_id, "<Enter>", self.on_enter)
        self.tag_bind(self.text_id, "<Leave>", self.on_leave)
        self.tag_bind(self.text_id, "<ButtonPress-1>", self.on_press)
        self.tag_bind(self.text_id, "<ButtonRelease-1>", self.on_release)

        self.hover = False

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [
            x1+r, y1,
            x2-r, y1,
            x2, y1,
            x2, y1+r,
            x2, y2-r,
            x2, y2,
            x2-r, y2,
            x1+r, y2,
            x1, y2,
            x1, y2-r,
            x1, y1+r,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def on_enter(self, event=None):
        self.hover = True
        if not self.is_pressed:
            self.itemconfig(1, fill=BUTTON_HOVER_BG)

    def on_leave(self, event=None):
        self.hover = False
        if not self.is_pressed:
            self.itemconfig(1, fill=BUTTON_BG)

    def on_press(self, event=None):
        self.is_pressed = True
        self.itemconfig(1, fill=BUTTON_ACTIVE_BG)

    def on_release(self, event=None):
        if self.is_pressed:
            self.is_pressed = False
            self.itemconfig(1, fill=BUTTON_HOVER_BG if self.hover else BUTTON_BG)
            if self.command:
                self.command()

    def config(self, **kwargs):
        if "text" in kwargs:
            self.text = kwargs["text"]
            self.itemconfig(self.text_id, text=self.text)
        if "command" in kwargs:
            self.command = kwargs["command"]
        if "width" in kwargs or "height" in kwargs:
            w = kwargs.get("width", self.width)
            h = kwargs.get("height", self.height)
            self.config(width=w, height=h)
            self.width = w
            self.height = h
        if "bg" in kwargs:
            self.bg = kwargs["bg"]
            self.itemconfig(1, fill=self.bg)
        if "fg" in kwargs:
            self.fg = kwargs["fg"]
            self.itemconfig(self.text_id, fill=self.fg)

class UnifiedDetectorApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.loading_overlay = None

        self.container = tk.Frame(self.root, bg=WINDOW_BG)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for Page in (MenuPage, VideoPage, PicturePage):
            page = Page(self.container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.frames[Page] = page

        self.show_frame(MenuPage)

    def show_frame(self, page):
        self.frames[page].tkraise()

    def get_model(self, callback=None):
        def _load():
            from ultralytics import YOLO
            self.model = YOLO(MODEL_PATH)
            if callback:
                self.root.after(0, callback)
            self.hide_loading()

        self.show_loading()
        threading.Thread(target=_load, daemon=True).start()

    def show_loading(self):
        if self.loading_overlay:
            return
        self.loading_overlay = tk.Toplevel(self.root)
        self.loading_overlay.overrideredirect(True)
        self.loading_overlay.attributes("-topmost", True)
        w, h = 200, 100
        x = self.root.winfo_screenwidth() // 2 - w // 2
        y = self.root.winfo_screenheight() // 2 - h // 2
        self.loading_overlay.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(self.loading_overlay, text="Loading Model...", font=DEFAULT_FONT).pack(expand=True)

    def hide_loading(self):
        if self.loading_overlay:
            self.loading_overlay.destroy()
            self.loading_overlay = None

class MenuPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=WINDOW_BG)
        container = tk.Frame(self, bg=WINDOW_BG)
        container.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(container, text="Soldier vs Civilian Detector", font=TITLE_FONT, bg=WINDOW_BG, fg="white").pack(pady=40)

        buttons = [
            ("Detect from Video", lambda: controller.show_frame(VideoPage)),
            ("Detect from Picture", lambda: controller.show_frame(PicturePage)),
            ("Exit", controller.root.quit)
        ]

        for text, cmd in buttons:
            btn = RoundedButton(container, text=text, width=250, height=50, command=cmd)
            btn.pack(pady=10)

class VideoPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=WINDOW_BG)
        self.controller = controller
        self.paused = False

        self.canvas = tk.Canvas(self, width=800, height=600, bg=WINDOW_BG,
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(pady=20)
        self.image_id = None

        btns = tk.Frame(self, bg=WINDOW_BG)
        btns.pack()

        self.load_btn = RoundedButton(btns, text="Load Video", command=self.load_video, width=130, height=40)
        self.pause_btn = RoundedButton(btns, text="Pause", command=self.toggle_pause, width=130, height=40)
        self.snap_btn = RoundedButton(btns, text="Snapshot", command=self.snapshot, width=130, height=40)
        self.stop_btn = RoundedButton(btns, text="Stop", command=self.stop_video, width=130, height=40)
        self.back_btn = RoundedButton(self, text="Back to Menu", command=self.back_to_menu, width=150, height=45)

        for btn in [self.load_btn, self.pause_btn, self.snap_btn, self.stop_btn]:
            btn.pack(side="left", padx=10, pady=10)
        self.back_btn.pack(pady=10)

        self.cap = None
        self.running = False
        self.current_frame = None
        self.after_id = None  # To keep track of the scheduled callback

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume" if self.paused else "Pause")

    def snapshot(self):
        if self.current_frame is not None:
            filename = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Snapshot", f"Frame saved as {filename}")

    def back_to_menu(self):
        self.stop_video()
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None
        self.canvas.image = None
        self.controller.show_frame(MenuPage)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not path:
            return

        # Stop previous video and cancel scheduled callbacks before loading new one
        self.stop_video()
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None
        self.canvas.image = None
        self.current_frame = None

        def start_processing():
            self.cap = cv2.VideoCapture(path)
            self.running = True
            self.paused = False
            self.process_frame()

        if self.controller.model is None:
            self.controller.get_model(callback=start_processing)
        else:
            start_processing()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None

    def process_frame(self):
        if not self.running or not self.cap or not self.cap.isOpened():
            return

        if self.paused:
            self.after_id = self.after(100, self.process_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        results = self.controller.model(frame, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.controller.model.names[cls]
            color = (0, 255, 0) if label == "civilian" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self.current_frame = frame.copy()
        disp = cv2.resize(frame, (800, 600))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        imgTK = ImageTk.PhotoImage(Image.fromarray(rgb))

        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=imgTK)
        else:
            self.canvas.itemconfig(self.image_id, image=imgTK)
        self.canvas.image = imgTK

        # Schedule next frame and store the after_id
        self.after_id = self.after(16, self.process_frame)

class PicturePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=WINDOW_BG)
        self.controller = controller

        self.canvas = tk.Canvas(self, width=800, height=600, bg=WINDOW_BG,
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack(pady=20)

        btns = tk.Frame(self, bg=WINDOW_BG)
        btns.pack()

        btn_load = RoundedButton(btns, text="Load Image", command=self.load_image, width=130, height=40)
        btn_clear = RoundedButton(btns, text="Clear Canvas", command=self.clear_canvas, width=130, height=40)
        btn_back = RoundedButton(btns, text="Back to Menu", command=lambda: controller.show_frame(MenuPage), width=150, height=45)

        for btn in [btn_load, btn_clear, btn_back]:
            btn.pack(side="left", padx=10, pady=10)

        self.img_disp = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.img_disp = None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return

        def do_predict():
            img = cv2.imread(path)
            results = self.controller.model.predict(img)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < 0.4:
                        continue
                    xy = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0].cpu().numpy())
                    label = "civilian" if cls == 0 else "soldier"
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    cv2.rectangle(img, (xy[0], xy[1]), (xy[2], xy[3]), color, 2)
                    cv2.putText(img, f"{label} {conf:.2f}",
                                (xy[0], xy[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            disp = cv2.resize(img, (800, 600))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            self.img_disp = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.img_disp)

        if self.controller.model is None:
            self.controller.get_model(callback=do_predict)
        else:
            do_predict()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Soldier vs Civilian Detector")
    root.geometry("1024x768")
    root.minsize(800, 600)
    root.configure(bg=WINDOW_BG)
    root.bind("<Escape>", lambda e: root.destroy())

    style = ttk.Style()
    style.theme_use("clam")

    app = UnifiedDetectorApp(root)
    root.mainloop()
