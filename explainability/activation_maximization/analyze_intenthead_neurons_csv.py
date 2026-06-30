import torch
import torchaudio
import numpy as np
from pathlib import Path
from icefall.utils import AttributeDict
from .train import get_model
import csv
from tqdm import tqdm

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
            "decoder_dim": 512,
            "joiner_dim": 512,
        }
    )

def wav_to_logmel(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        sample_frequency=target_sr,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
    )
    return fbank

def normalize(mat):
    """Z-score normalization along time axis (frames)"""
    mu = mat.mean(axis=0, keepdims=True)
    sigma = mat.std(axis=0, keepdims=True) + 1e-8
    z = (mat - mu) / sigma
    return np.clip(z, -3, 3)

def hook_fn(storage):
    def hook(module, input, output):
        storage["layer9"] = output.detach().cpu()
    return hook

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="intent_layer9_csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    params = build_params()
    model = get_model(params)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()
    print("Model loaded.")

    # Hook layer 9
    layer_outputs = {}
    layer9 = dict(model.intent_classifier.named_children()).get("9")
    if layer9 is None:
        raise ValueError("Layer 9 not found in intent_classifier")
    layer9.register_forward_hook(hook_fn(layer_outputs))
    print("Hooked intent_classifier layer 9")

    # CSV
    csv_file = out_dir / "intent_layer9_top10.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "top10_neuron_indices"])

        audio_folder = Path(args.audio)
        audio_files = sorted(audio_folder.glob("*.flac"))

        for audio_path in tqdm(audio_files):
            audio_name = audio_path.stem
            feats = wav_to_logmel(audio_path).unsqueeze(0).to(device)
            x_lens = torch.tensor([feats.shape[1]], device=device)

            layer_outputs.clear()

            # Forward pass
            enc_out, _ = model.encoder(feats, x_lens)
            _ = model.intent_classifier(enc_out)

            # Get top 10 neurons using the same normalization
            act_layer9 = layer_outputs["layer9"].squeeze(0).numpy()  # [T, D]
            act_layer9_norm = normalize(act_layer9)
            top10_idx = np.argsort(-np.abs(act_layer9_norm).mean(axis=0))[:10] 

            writer.writerow([audio_name, ",".join(map(str, top10_idx))])

    print(f"Done. CSV saved to {csv_file}")

if __name__ == "__main__":
    main()
