import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio
import torch.optim as optim
import soundfile as sf
from torch.utils.data import DataLoader, TensorDataset
import h5py

# 🔹 Normalization Function
def normalize_audio(audio):
    max_val = torch.max(torch.abs(audio))
    return audio / max_val if max_val > 0 else audio  # Prevent division by zero

# 🔹 Autoencoder Model
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

# 🔹 Spectrogram Loss Function
class SpectrogramLoss(nn.Module):
    def __init__(self, n_fft=512, power=2):
        super(SpectrogramLoss, self).__init__()
        self.stft = T.Spectrogram(n_fft=n_fft, power=power)
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output_spec = self.stft(output)
        target_spec = self.stft(target)
        # Avoid division by zero
        output_spec = (output_spec - torch.mean(output_spec)) / (torch.std(output_spec) + 1e-6)
        target_spec = (target_spec - torch.mean(target_spec)) / (torch.std(target_spec) + 1e-6)
        return self.mse(output_spec, target_spec)

# 🔹 Save Model Weights (H5 & PyTorch)
def save_model(model, h5_filename="model_weights.h5", torch_filename="model_weights.pth"):
    torch.save(model.state_dict(), torch_filename)
    with h5py.File(h5_filename, "w") as f:
        for name, param in model.state_dict().items():
            f.create_dataset(name, data=param.cpu().numpy())
    print(f"✅ Model saved as '{h5_filename}' and '{torch_filename}'")

# 🔹 Load Model Weights from H5
def load_model_from_h5(model, h5_filename="model_weights.h5"):
    with h5py.File(h5_filename, "r") as f:
        for name, param in model.state_dict().items():
            param.copy_(torch.tensor(f[name][()]))
    print(f"✅ Model loaded from '{h5_filename}'")
    return model

# 🔹 Train Autoencoder
def train_autoencoder(model, dataloader, num_epochs=50, learning_rate=0.001):
    model.train()  # Set model to training mode
    criterion = SpectrogramLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for noisy_audio, clean_audio in dataloader:
            optimizer.zero_grad()
            output = model(noisy_audio)
            loss = criterion(output, clean_audio)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    save_model(model)  # Save after training
    return model

# 🔹 Process Audio (Train & Denoise)
def process_audio(input_file, chunk_size=16384, batch_size=32, num_epochs=100, learning_rate=0.001, noise_level=1.9):
    print("🔹 Loading audio...")
    waveform, sample_rate = torchaudio.load(input_file)
    waveform = normalize_audio(waveform[0])
    print(f"✅ Audio Loaded! Shape: {waveform.shape}")

    print("🔹 Adding noise...")
    noise = noise_level * torch.randn_like(waveform)
    noisy_waveform = normalize_audio(waveform + noise)
    sf.write('noisy_audio.wav', noisy_waveform.numpy(), sample_rate)
    print("✅ Noise Added!")

    # Chunking the waveform
    noisy_chunks = [noisy_waveform[i:i + chunk_size] for i in range(0, len(noisy_waveform) - chunk_size, chunk_size)]
    clean_chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform) - chunk_size, chunk_size)]

    # Filter out incomplete chunks
    noisy_chunks = [chunk for chunk in noisy_chunks if len(chunk) == chunk_size]
    clean_chunks = [chunk for chunk in clean_chunks if len(chunk) == chunk_size]

    # Convert to tensors
    dataset = TensorDataset(torch.stack(noisy_chunks), torch.stack(clean_chunks))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("🔹 Training autoencoder...")
    model = DenoisingAutoencoder()
    model = train_autoencoder(model, dataloader, num_epochs, learning_rate)
    print("✅ Training Complete!")

    print("🔹 Denoising audio...")
    model.eval()  # Set model to evaluation mode
    denoised_audio_chunks = []
    with torch.no_grad():
        for noisy_chunk in torch.stack(noisy_chunks):
            denoised_chunk = model(noisy_chunk.unsqueeze(0)).squeeze(0)
            denoised_audio_chunks.append(denoised_chunk)

    denoised_audio = torch.cat(denoised_audio_chunks)
    sf.write('denoised_audio.wav', denoised_audio.numpy(), sample_rate)
    print("✅ Denoised audio saved as 'denoised_audio.wav'")

    return denoised_audio.numpy(), noisy_waveform.numpy(), sample_rate

# 🔹 Hyperparameter Tuning
def simple_hyperparameter_tuning(input_file, chunk_size=16384, batch_size=32, noise_level=0.9):
    learning_rates = [0.001, 0.0005, 0.0001]
    num_epochs_values = [50, 100]
    best_loss, best_model, best_lr, best_epochs = float('inf'), None, None, None

    print("🔹 Loading audio for hyperparameter tuning...")
    waveform, _ = torchaudio.load(input_file)
    waveform = normalize_audio(waveform[0])

    print("🔹 Adding noise...")
    noisy_waveform = normalize_audio(waveform + noise_level * torch.randn_like(waveform))

    # Prepare dataset
    dataset = TensorDataset(torch.stack([noisy_waveform[i:i + chunk_size] for i in range(0, len(noisy_waveform) - chunk_size, chunk_size)]),
                            torch.stack([waveform[i:i + chunk_size] for i in range(0, len(waveform) - chunk_size, chunk_size)]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for lr in learning_rates:
        for epochs in num_epochs_values:
            model = train_autoencoder(DenoisingAutoencoder(), dataloader, num_epochs=epochs, learning_rate=lr)
            avg_loss = sum(SpectrogramLoss()(model(noisy_audio), clean_audio).item() for noisy_audio, clean_audio in dataloader) / len(dataloader)
            if avg_loss < best_loss:
                best_loss, best_model, best_lr, best_epochs = avg_loss, model, lr, epochs

    print(f"✅ Best Model: LR = {best_lr}, Epochs = {best_epochs}, Loss = {best_loss:.4f}")
    return best_model

if __name__ == "__main__":
    process_audio("female-vocal-321-countdown-240912.wav")
