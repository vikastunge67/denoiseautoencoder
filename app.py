from flask import Flask, render_template, request, send_from_directory
import os
import torch
import torchaudio
import torch.nn as nn
import h5py

# Flask app setup
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
MODEL_PTH_PATH = "model_weights.pth"  # Ensure correct path
MODEL_H5_PATH = "model_weights.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Define model architecture
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)  # Remove channel dimension

# Load model
device = torch.device("cpu")
model = DenoisingAutoencoder().to(device)

def load_model():
    try:
        if os.path.exists(MODEL_PTH_PATH):
            checkpoint = torch.load(MODEL_PTH_PATH, map_location=device)
            model.load_state_dict(checkpoint)
            model.eval()
            print("✅ Model loaded from PTH file!")
        elif os.path.exists(MODEL_H5_PATH):
            with h5py.File(MODEL_H5_PATH, "r") as f:
                for name, param in model.state_dict().items():
                    param.copy_(torch.tensor(f[name][()]))
            model.eval()
            print("✅ Model loaded from H5 file!")
        else:
            print("❌ Model file not found!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

# Normalize audio
def normalize_audio(audio):
    max_val = torch.max(torch.abs(audio))
    return audio / max_val if max_val > 0 else audio  # Prevent division by zero

# Function to denoise audio
def denoise_audio(input_path, output_path):
    waveform, sample_rate = torchaudio.load(input_path)  

    print(f"📌 Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")

    # Convert stereo to mono
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Keep it 2D

    # Normalize waveform (-1 to 1)
    waveform = waveform / waveform.abs().max()

    print(f"📌 Normalized waveform shape: {waveform.shape}")

    # Ensure correct shape (1, samples)
    waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

    # Store original length
    original_length = waveform.shape[1]

    # Target length for model
    target_length = 16000  
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]  # Trim
    else:
        pad_length = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))  # Pad

    print(f"📌 After padding/trimming: {waveform.shape}")

    # Pass through model
    with torch.no_grad():
        denoised_waveform = model(waveform.to(device))

    # Ensure output is 2D
    denoised_waveform = denoised_waveform.squeeze(0)

    # Trim back to original length
    denoised_waveform = denoised_waveform[:original_length]

    print(f"📌 Final waveform shape before saving: {denoised_waveform.shape}")

    # Normalize again to prevent distortion
    denoised_waveform = denoised_waveform / denoised_waveform.abs().max()

    # Save with original sample rate
    torchaudio.save(output_path, denoised_waveform.unsqueeze(0).cpu(), sample_rate)
    print(f"✅ Denoised audio saved: {output_path}")





# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        processed_filename = "denoised_" + file.filename
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        denoise_audio(file_path, processed_path)

        return render_template("index.html", filename=processed_filename)

    return render_template("index.html", filename=None)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    load_model()  # Load the model before running the app
    app.run(debug=True)
