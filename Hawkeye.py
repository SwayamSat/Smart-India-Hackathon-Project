import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import scipy.io.wavfile as wav
from librosa.util import normalize

class RadarSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface")

        # Set the window to full-screen mode
        self.root.state('zoomed')

        # Styling the interface
        self.root.configure(bg="#1e1e2f")
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TButton",
                        font=("Helvetica", 12),
                        padding=10,
                        background="#4e4e69",
                        foreground="white",
                        borderwidth=0)
        style.map("TButton",
                  background=[('active', '#6c6c94')])

        # Main container frame
        self.main_frame = tk.Frame(self.root, bg="#1e1e2f")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Frame for the waveform visualization
        self.waveform_frame = tk.LabelFrame(self.main_frame, text="Waveform Visualization", bg="#2e2e3e", fg="white", font=("Helvetica", 14, "bold"))
        self.waveform_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame for controls and result
        self.control_frame = tk.Frame(self.main_frame, bg="#1e1e2f")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Load File Button
        self.load_button = ttk.Button(self.control_frame, text="Load WAV File", command=self.load_wav_file)
        self.load_button.pack(pady=20, fill=tk.X)

        # Classify Button
        self.classify_button = ttk.Button(self.control_frame, text="Classify", command=self.classify, state=tk.DISABLED)
        self.classify_button.pack(pady=10, fill=tk.X)

        # Exit Button
        self.exit_button = ttk.Button(self.control_frame, text="Exit", command=self.root.quit)
        self.exit_button.pack(pady=10, fill=tk.X)

        # Frame for displaying the result
        self.result_frame = tk.LabelFrame(self.control_frame, text="Classification Result", bg="#2e2e3e", fg="white", font=("Helvetica", 14, "bold"))
        self.result_frame.pack(pady=30, fill=tk.BOTH, expand=True)

        # Label for displaying the result
        self.result_label = tk.Label(self.result_frame, text="No result yet", font=("Helvetica", 20, "bold"), bg="#2e2e3e", fg="#00ff00")
        self.result_label.pack(expand=True)

        # Initialize placeholders for the loaded wav file and plot objects
        self.loaded_wav = None
        self.fs = None

        # Create a blank waveform canvas
        self.fig_waveform, self.ax_waveform = plt.subplots(figsize=(6, 4))
        self.ax_waveform.set_facecolor("#2e2e3e")
        self.fig_waveform.patch.set_facecolor('#2e2e3e')
        self.ax_waveform.tick_params(colors='white')
        self.ax_waveform.spines['bottom'].set_color('white')
        self.ax_waveform.spines['top'].set_color('white')
        self.ax_waveform.spines['right'].set_color('white')
        self.ax_waveform.spines['left'].set_color('white')
        self.waveform_canvas = FigureCanvasTkAgg(self.fig_waveform, master=self.waveform_frame)
        self.waveform_canvas.draw()
        self.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Load the pre-trained model
        self.model = tf.keras.models.load_model('trainedCNNModel.h5')

    def load_wav_file(self):
        # Open file dialog to select WAV file
        file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if file_path:
            try:
                self.fs, self.loaded_wav = wav.read(file_path)
                self.loaded_wav = self.loaded_wav.astype(np.float32)

                # Normalize the waveform
                self.loaded_wav = self.loaded_wav / np.max(np.abs(self.loaded_wav))

                self.visualize_waveform()
                self.result_label.config(text="Ready to classify", fg="yellow")
                self.classify_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load WAV file:\n{e}")
                self.loaded_wav = None
                self.fs = None
                self.classify_button.config(state=tk.DISABLED)
                self.result_label.config(text="No result yet", fg="#00ff00")

    def visualize_waveform(self):
        if self.loaded_wav is not None:
            self.ax_waveform.clear()
            time_axis = np.linspace(0, len(self.loaded_wav) / self.fs, num=len(self.loaded_wav))
            self.ax_waveform.plot(time_axis, self.loaded_wav, color="#00aced")
            self.ax_waveform.set_title("Waveform", color="white", fontsize=16)
            self.ax_waveform.set_xlabel("Time [s]", color="white", fontsize=12)
            self.ax_waveform.set_ylabel("Amplitude", color="white", fontsize=12)
            self.waveform_canvas.draw()

    def classify(self):
        if self.loaded_wav is not None:
            try:
                # Convert the waveform to a spectrogram
                S = librosa.feature.melspectrogram(y=self.loaded_wav, sr=self.fs, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)

                # Normalize the spectrogram
                S_db_normalized = normalize(S_db)

                # Resize or pad the spectrogram to fit the desired dimensions (128, 128)
                target_width = 128
                if S_db_normalized.shape[1] > target_width:
                    resized_spectrogram = S_db_normalized[:, :target_width]
                else:
                    padding = target_width - S_db_normalized.shape[1]
                    resized_spectrogram = np.pad(S_db_normalized, ((0, 0), (0, padding)), mode='constant')

                # Add channel dimension and batch size dimension
                input_data = resized_spectrogram[np.newaxis, ..., np.newaxis]

                # Make a prediction
                prediction = self.model.predict(input_data)

                # Interpret the result
                if np.argmax(prediction) == 1:
                    result = "Drone"
                    color = "#00ff00"
                else:
                    result = "Bird"
                    color = "#ff5733"

                self.result_label.config(text=result, fg=color)

            except Exception as e:
                messagebox.showerror("Error", f"Classification failed:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarSimulator(root)
    root.mainloop()
