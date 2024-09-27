import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import threading
import time
import winsound
from datetime import datetime
from PIL import Image, ImageTk

class RadarSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("TEAM SKY SENTINEL")
        self.root.geometry('1920x1080')
        self.root.configure(bg="#1c1c2d")

        self.is_streaming = False
        self.loaded_image = None

        # Create a gradient background
        self.create_gradient_background()

        # Header
        self.create_header()

        # Main container frame
        self.main_frame = tk.Frame(self.root, bg="#2c2c4a")
        self.main_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.75)

        # Left side: Image Visualization & Radar Animation
        self.create_visualization_frame()

        # Right side: Controls
        self.create_control_frame()

        # Load the pre-trained model
        self.load_model()

        # Apply custom styles
        self.apply_styles()

        # Start updating the time
        self.update_time()

    def create_gradient_background(self):
        self.bg_canvas = tk.Canvas(self.root, width=1920, height=1080)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True)
        self.bg_canvas.create_rectangle(0, 0, 1920, 540, fill="#1c1c2d", outline="")
        self.bg_canvas.create_rectangle(0, 540, 1920, 1080, fill="#2c2c4a", outline="")

    def create_header(self):
        self.header_frame = tk.Frame(self.bg_canvas, bg="#29293e")
        self.header_frame.place(relx=0.001, rely=0.05, relwidth=1, relheight=0.1)

        # Logo
        try:
            logo = Image.open("hawkeye_logo.png")  # Ensure you have this image file
            logo = logo.resize((80, 80), Image.Resampling.LANCZOS)
            logo = ImageTk.PhotoImage(logo)
            logo_label = tk.Label(self.header_frame, image=logo, bg="#29293e")
            logo_label.image = logo
            logo_label.pack(side=tk.LEFT, padx=20)
        except FileNotFoundError:
            print("Logo file not found. Skipping logo display.")
        except Exception as e:
            print(f"Error loading logo: {e}")

        # Add text shadow effect using a label with offset
        shadow_label = tk.Label(self.header_frame, text="HAWKEYE: DRONE VS BIRD CLASSIFICATION", 
                                font=("Comic Sans MS", 36, "bold"), fg="#8B8B00", bg="#29293e")
        shadow_label.place(relx=0.5, rely=0.5, anchor='center', x=2, y=2)
        
        # Header label with artistic style
        self.header_label = tk.Label(self.header_frame, text="HAWKEYE: DRONE VS BIRD CLASSIFICATION", 
                                     font=("Comic Sans MS", 36, "bold"), fg="#FFDD00", bg="#29293e")
        self.header_label.place(relx=0.5, rely=0.5, anchor='center')

        # Time frame
        self.time_frame = tk.Frame(self.header_frame, bg="#29293e")
        self.time_frame.pack(side=tk.RIGHT, padx=20, pady=10)

        self.time_label = tk.Label(self.time_frame, text="", 
                                   font=("Helvetica", 18), fg="#ffffff", bg="#29293e")
        self.time_label.pack()

        self.date_label = tk.Label(self.time_frame, text="", 
                                   font=("Helvetica", 18), fg="#ffffff", bg="#29293e")
        self.date_label.pack()

    def create_visualization_frame(self):
        self.image_frame = tk.LabelFrame(self.main_frame, text="Image Visualization", 
                                         bg="#2e2e4a", fg="white", font=("Helvetica", 14, "bold"))
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create a blank image canvas
        self.fig_image, self.ax_image = plt.subplots(figsize=(8, 6))
        self.ax_image.set_facecolor("#3e3e5a")
        self.fig_image.patch.set_facecolor('#3e3e5a')
        self.ax_image.tick_params(colors='white')
        for spine in self.ax_image.spines.values():
            spine.set_color('white')
        self.image_canvas = FigureCanvasTkAgg(self.fig_image, master=self.image_frame)
        self.image_canvas.draw()
        self.image_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_control_frame(self):
        self.control_frame = tk.Frame(self.main_frame, bg="#2e2e4a")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        buttons = [
            ("Load Image File", self.load_image_file),
            ("Classify", self.classify),
            ("Start Streaming", self.toggle_streaming),
            ("Exit", self.root.quit)
        ]

        for text, command in buttons:
            btn = ttk.Button(self.control_frame, text=text, command=command, style="Custom.TButton")
            btn.pack(pady=10, fill=tk.X)

        self.classify_button = self.control_frame.winfo_children()[1]
        self.classify_button.config(state=tk.DISABLED)
        self.stream_button = self.control_frame.winfo_children()[2]

        # Frame for displaying the result
        self.result_frame = tk.LabelFrame(self.control_frame, text="Classification Result", 
                                          bg="#3e3e5a", fg="white", font=("Helvetica", 14, "bold"))
        self.result_frame.pack(pady=20, fill=tk.X)

        self.result_label = tk.Label(self.result_frame, text="No result yet", 
                                     font=("Helvetica", 20, "bold"), bg="#3e3e5a", fg="#00ff00")
        self.result_label.pack(expand=True, pady=10)

        # Logs Frame
        self.log_frame = tk.LabelFrame(self.control_frame, text="Logs", 
                                       bg="#3e3e5a", fg="white", font=("Helvetica", 14, "bold"))
        self.log_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(self.log_frame, height=12, bg="#2e2e4a", fg="white", 
                                font=("Helvetica", 12, "bold"))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def apply_styles(self):
        style = ttk.Style()
        style.configure("Custom.TButton", 
                        font=("Helvetica", 14, "bold"), 
                        padding=15, 
                        background="#4e4e69", 
                        foreground="black", 
                        borderwidth=0)
        style.map("Custom.TButton", 
                  background=[('active', '#6c6c94'), ('pressed', '#3e3e5a')])

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model('trainedCNNModel.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.root.quit()

    def load_image_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            try:
                image = tf.io.read_file(file_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, (525, 700))
                image = image / 255.0
                self.loaded_image = image

                self.visualize_image()
                self.result_label.config(text="Ready to classify", fg="yellow")
                self.classify_button.config(state=tk.NORMAL)
                self.log_message(f"Loaded image: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image file:\n{e}")
                self.loaded_image = None
                self.classify_button.config(state=tk.DISABLED)
                self.result_label.config(text="No result yet", fg="#00ff00")

    def visualize_image(self):
        if self.loaded_image is not None:
            self.ax_image.clear()
            self.ax_image.imshow(self.loaded_image.numpy())
            self.ax_image.set_title("Image", color="white", fontsize=16)
            self.image_canvas.draw()

    def play_beep(self):
        winsound.Beep(1000, 1000)

    def classify(self):
        if self.loaded_image is not None:
            try:
                prediction = self.model.predict(self.loaded_image[tf.newaxis, ...])
                result = "Drone" if prediction >= 0.5 else "Bird"
                color = "#ff5733" if result == "Drone" else "#00ff00"

                self.result_label.config(text=result, fg=color)
                self.log_message(f"Classification result: {result}")

                if result == "Drone":
                    self.play_beep()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to classify image:\n{e}")

    def toggle_streaming(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_button.config(text="Start Streaming")
        else:
            self.is_streaming = True
            self.stream_button.config(text="Stop Streaming")
            threading.Thread(target=self.stream_data, daemon=True).start()

    def stream_data(self):
        def stream():
            while self.is_streaming:
                time.sleep(1)
                random_signal = np.random.random((525, 700, 3))
                self.ax_image.clear()
                self.ax_image.imshow(random_signal)
                self.ax_image.set_title("Streaming Radar Signal", color="white", fontsize=16)
                self.image_canvas.draw()
                self.log_message("Streaming radar signal...")
                if not self.is_streaming:
                    break
        thread = threading.Thread(target=stream)
        thread.start()

    def update_time(self):
        now = datetime.now()
        self.time_label.config(text=now.strftime("%H:%M:%S"))
        self.date_label.config(text=now.strftime("%Y-%m-%d"))
        self.root.after(1000, self.update_time)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarSimulator(root)
    root.mainloop()