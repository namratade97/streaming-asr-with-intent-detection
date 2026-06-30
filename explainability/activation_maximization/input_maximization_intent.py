import sys
import os
import pathlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
#use icefall files here
from icefall.utils import AttributeDict
from train import get_model
from model import some_mapping


NUM_INTENTS = 92
FEATURE_DIM = 80
FRAMES = 300          
LR = 0.05
STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTDIR = Path("input_maximization_results_3")
OUTDIR.mkdir(exist_ok=True)

def build_params():
    return AttributeDict(
        {
            "feature_dim": 80,
            "num_encoder_layers": "2,2,3,4,3,2",
            "downsampling_factor": "1,2,4,8,4,2",
            "encoder_dim": "192,256,384,512,384,256",
            "encoder_unmasked_dim": "192,192,256,256,256,192",
            "query_head_dim": "32",
            "value_head_dim": "12",
            "pos_head_dim": "4",
            "pos_dim": 48,
            "num_heads": "4,4,4,8,4,4",
            "feedforward_dim": "512,768,1024,1536,1024,768",
            "cnn_module_kernel": "31,31,15,15,15,31",
            "causal": True,
            "chunk_size": "16",
            "left_context_frames": "512",
            "subsampling_factor": 4,
            "decoder_dim": 512,
            "use_transducer": True,
            "use_ctc": False,
            "vocab_size": 500,
            "blank_id": 0,
            "context_size": 4,
            "joiner_dim": 512,
        }
    )

def normalize(x):
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x

def plot_input(feats, out_path, title=""):
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




def maximize_intent(model, class_idx, steps=STEPS, lr=0.02): 
    # Initialize random logmel input
    x = torch.randn(1, FRAMES, FEATURE_DIM, device=DEVICE, requires_grad=True)
    x_lens = torch.tensor([FRAMES], device=DEVICE)

    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        
       
        shift = np.random.randint(-4, 5)
        if shift != 0:
            x_perturbed = torch.roll(x, shifts=shift, dims=1)
        else:
            x_perturbed = x
            
        logits = model.infer_intent_logits(x_perturbed, x_lens)
        
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[0, class_idx]

        tv_time = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_freq = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
        tv_loss = (tv_time + tv_freq) * 5e-2  #1e-2
        
        energy_loss = torch.mean(x ** 2) * 1e-4
        
        loss = -target_log_prob + tv_loss + energy_loss  

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x.clamp_(-4, 4)

    return x.detach().cpu().squeeze(0).numpy()

def main():
    # Load model
    params = build_params()
    model = get_model(params)
    
    ckpt = torch.load("/Users/nde/Downloads/epoch-200.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(DEVICE).eval()
    print("Model loaded successfully. Starting generation loop...")

    id_to_intent = {v: k for k, v in some_mapping.items()}

    for class_idx in tqdm(range(NUM_INTENTS)):
        feats = maximize_intent(model, class_idx)
        
        # Apply Gaussian smoothing over both axes (Time and Mel-frequency bins) 
        feats_smooth = gaussian_filter1d(feats, sigma=1.5, axis=0)  # Time axis
        feats_smooth = gaussian_filter1d(feats_smooth, sigma=0.8, axis=1)  # Freq axis

        # Parse valid string labels for OS systems
        intent_name = id_to_intent.get(class_idx, f"intent_{class_idx}")
        safe_name = intent_name.replace("/", "_").replace(" ", "_")

        # Plot structural features (.T maps dimensions correctly to horizontal/vertical axes)
        out_path = OUTDIR / f"{safe_name}.png"
        plot_input(feats_smooth.T, out_path, title=f"{intent_name}")
        np.save(OUTDIR / f"{safe_name}.npy", feats_smooth)

if __name__ == "__main__":
    main()

