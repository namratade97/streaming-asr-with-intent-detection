import sys
import os
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from transformers import SpeechT5HifiGan

# Directories
indir = Path("actual_audio_spectrograms")
outdir = Path("reconstructed_audio_neural_actual_audio_spectrograms")
outdir.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading production-grade HiFi-GAN Vocoder from Hugging Face...")
# Load Microsoft's standalone SpeechT5 HiFi-GAN vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
vocoder.to(DEVICE).eval()

for npy_file in indir.glob("*.npy"):
    # Load optimized (T, F) features
    feats = np.load(npy_file)  # shape: (300, 80)
    mel_tensor = torch.from_numpy(feats).unsqueeze(0).to(DEVICE).float()
    
    print(f"Synthesizing {npy_file.name}...")
    with torch.no_grad():
        # Pass features directly to the model
        audio = vocoder(mel_tensor)
        audio = audio.squeeze().cpu().numpy()

    # Normalize audio levels to avoid clipping
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # SpeechT5's HiFi-GAN outputs native 16000Hz audio
    out_file = outdir / (npy_file.stem + ".wav")
    sf.write(out_file, audio, 16000)
    print(f"Successfully saved clean audio: {out_file}\n")