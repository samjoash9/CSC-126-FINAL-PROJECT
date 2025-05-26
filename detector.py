import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from threading import Thread

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Military Personnel Detector")
        self.root.geometry("900x700")
        
        # Load YOLO model
        self.model = YOLO("runs/detect/final_mc8/weights/best.pt")
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create menu buttons
        self.create_menu()
        
        # Detection frame (initially empty)
        self.detection_frame = ttk.Frame(self.main_frame)
        self.detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status variables
        self.current_mode = None
        self.running = False
        self.cap = None
    
    def create_menu(self):
        """Create the main menu buttons"""
        menu_frame = ttk.Frame(self.main_frame)
        menu_frame.pack(pady=20)
        
        btn_style = ttk.Style()
        btn_style.configure('Menu.TButton', font=('Helvetica', 12), padding=10)
        
        self.image_btn = ttk.Button(
            menu_frame, 
            text="Image Detection", 
            style='Menu.TButton',
            command=self.setup_image_detection
        )
        self.image_btn.pack(side=tk.LEFT, padx=10)
        
        self.video_btn = ttk.Button(
            menu_frame, 
            text="Video Detection", 
            style='Menu.TButton',
            command=self.setup_video_detection
        )
        self.video_btn.pack(side=tk.LEFT, padx=10)
        
        self.exit_btn = ttk.Button(
            menu_frame, 
            text="Exit", 
            style='Menu.TButton',
            command=self.root.quit
        )
        self.exit_btn.pack(side=tk.LEFT, padx=10)
    
    def clear_detection_frame(self):
        """Clear the current detection interface"""
        for widget in self.detection_frame.winfo_children():
            widget.destroy()
    
    def setup_image_detection(self):
        """Set up the image detection interface"""
        self.current_mode = "image"
        self.clear_detection_frame()
        
        # Create image display canvas
        self.canvas = tk.Canvas(self.detection_frame, width=800, height=600)
        self.canvas.pack()
        
        # Create control buttons
        control_frame = ttk.Frame(self.detection_frame)
        control_frame.pack(pady=10)
        
        self.load_btn = ttk.Button(
            control_frame, 
            text="Load Image", 
            command=self.load_image
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.back_btn = ttk.Button(
            control_frame, 
            text="Back to Menu", 
            command=self.return_to_menu
        )
        self.back_btn.pack(side=tk.LEFT, padx=5)
        
        # Initialize image variables
        self.image = None
        self.img_for_display = None
    
    def setup_video_detection(self):
        """Set up the video detection interface"""
        self.current_mode = "video"
        self.clear_detection_frame()
        
        # Create video display panel
        self.video_panel = ttk.Label(self.detection_frame)
        self.video_panel.pack()
        
        # Create control buttons
        control_frame = ttk.Frame(self.detection_frame)
        control_frame.pack(pady=10)
        
        self.load_vid_btn = ttk.Button(
            control_frame, 
            text="Load Video", 
            command=self.load_video
        )
        self.load_vid_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            control_frame, 
            text="Stop", 
            command=self.stop_video,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.back_btn = ttk.Button(
            control_frame, 
            text="Back to Menu", 
            command=self.return_to_menu
        )
        self.back_btn.pack(side=tk.LEFT, padx=5)
        
        # Initialize video variables
        self.running = False
        self.video_path = None
        self.cap = None
    
    def return_to_menu(self):
        """Return to the main menu"""
        if self.current_mode == "video":
            self.stop_video()
        self.clear_detection_frame()
        self.create_menu()
        self.current_mode = None
    
    def load_image(self):
        """Load and process an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return
        
        # Read and process image
        img = cv2.imread(file_path)
        if img is None:
            print("Failed to load image.")
            return
        
        # Run detection
        results = self.model.predict(img)
        
        # Draw results
        annotated_img = img.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                label = "civilian" if cls == 0 else "soldier"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                
                cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(annotated_img, text, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert for display
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((800, 600))
        
        self.img_for_display = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.img_for_display)
    
    def load_video(self):
        """Load a video file"""
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if self.video_path:
            self.running = True
            self.cap = cv2.VideoCapture(self.video_path)
            self.load_vid_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            Thread(target=self.process_video, daemon=True).start()
    
    def stop_video(self):
        """Stop video processing"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.load_vid_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def process_video(self):
        """Process video frames"""
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run detection
            results = self.model(frame, verbose=False)[0]
            
            # Draw results
            annotated_frame = frame.copy()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = "civilian" if cls == 0 else "soldier"
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display frame
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.video_panel.imgtk = img_tk
            self.video_panel.configure(image=img_tk)
            
            # Control frame rate
            self.root.after(16, self.process_frame)
        
        self.stop_video()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()