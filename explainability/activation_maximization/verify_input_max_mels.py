import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

SR = 16000
N_FFT = 512       # Matched to the 10ms hop/25ms window config
HOP_LENGTH = 160   # 10ms frame shift
WIN_LENGTH = 400   # 25ms window length
N_MELS = 80
TARGET_FRAMES = 300 

OUTDIR = Path("actual_audio_spectrograms")
OUTDIR.mkdir(exist_ok=True)

def normalize(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def plot_and_save_spectrogram(feats, out_path, title):
    feats = normalize(feats)
    plt.figure(figsize=(8, 4))
    plt.imshow(feats, aspect="auto", origin="lower", cmap="magma")
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Mel bins")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def generate_actual_spectrogram(audio_path, intent_name):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"Error: Audio file not found at {audio_path}")
        return

    y, sr = librosa.load(audio_path, sr=SR)

    # Extract Power Spectrogram
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS
    )

    # Convert to Log-Mel scale (matching Zipformer input features)
    mel_log = np.log(mel_power + 1e-5) # shape: (80, Frames)

    # Enforce the exact same 300-frame width (3 seconds) for direct comparison
    current_frames = mel_log.shape[1]
    if current_frames < TARGET_FRAMES:
        # Pad with silence if the audio clip is too short
        pad_width = TARGET_FRAMES - current_frames
        mel_log = np.pad(mel_log, ((0, 0), (0, pad_width)), mode='edge')
    elif current_frames > TARGET_FRAMES:
        # Truncate if the audio clip is too long
        mel_log = mel_log[:, :TARGET_FRAMES]

    audio_name = audio_path.stem
    safe_intent = intent_name.replace("/", "_").replace(" ", "_")
    
    out_img_path = OUTDIR / f"{safe_intent}_{audio_name}.png"
    out_npy_path = OUTDIR / f"{safe_intent}_{audio_name}.npy"

    # Save image plot
    plot_and_save_spectrogram(mel_log, out_img_path, title=f"Actual: {intent_name} ({audio_name})")
    np.save(out_npy_path, mel_log.T)
    
    print(f"Generated actual spectrogram:")
    print(f"  └ Image: {out_img_path}")
    print(f"  └ Array: {out_npy_path}\n")

if __name__ == "__main__":

    AUDIO_PATH = "/Users/nde/Downloads/audio_2/train/audio-1488990184.flac"
    INTENT_NAME = "currency"

    generate_actual_spectrogram(AUDIO_PATH, INTENT_NAME)