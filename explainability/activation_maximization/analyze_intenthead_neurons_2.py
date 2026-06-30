import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from icefall.utils import AttributeDict
from .train import get_model
import csv


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



def plot_topk_neurons(mat, title, out_path, top_k=10, return_topk=False):
    """
    mat: (T, D) time x neurons
    """
    mat = normalize(mat)
    T, D = mat.shape
    topk_idx = np.argsort(-np.abs(mat).mean(axis=0))[:top_k]

    times = np.arange(T)  # approximate frame indices
    plt.figure(figsize=(10, 4))
    for i in topk_idx:
        plt.plot(times, mat[:, i], label=f"Neuron {i}")
    plt.xlabel("Time frames (approx.)")
    plt.ylabel("Activation (z-score)")
    plt.title(title + f" | Top-{top_k} neurons")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    if return_topk:
        return topk_idx



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
    mu = mat.mean(axis=0, keepdims=True)
    sigma = mat.std(axis=0, keepdims=True) + 1e-8
    z = (mat - mu) / sigma
    return np.clip(z, -3, 3)


def hook_fn(name, storage):
    def hook(module, input, output):
        storage[name] = output.detach().cpu()
    return hook


def plot_heatmap(mat, title, out_path):
    mat = normalize(mat)
    plt.figure(figsize=(10, 5))
    plt.imshow(mat.T, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(label="Activation (z-score)")
    plt.xlabel("Time frames (approx.)")
    plt.ylabel("Neuron index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="intent_classifier_heatmaps")
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

    # Hook intent classifier layers
    layer_outputs = {}
    intent_layers = [1, 4, 7, 9]
    for i in intent_layers:
        layer = dict(model.intent_classifier.named_children()).get(str(i))
        if layer is not None:
            layer.register_forward_hook(hook_fn(f"intent_classifier.{i}", layer_outputs))
            print(f"Hooked intent_classifier.{i}")

    # Process each audio file
    audio_folder = Path(args.audio)
    audio_files = sorted(audio_folder.glob("*.flac"))

    # Before audio loop
    top_neurons_file = out_dir / "intent_layer9_top10.csv"
    with open(top_neurons_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_file", "top10_neuron_indices"])  # header

        # Process each audio file
        for audio_path in tqdm(audio_files):
            audio_name = audio_path.stem
            feats = wav_to_logmel(audio_path).unsqueeze(0).to(device)
            x_lens = torch.tensor([feats.shape[1]], device=device)

            # Clear storage
            layer_outputs.clear()

            # Forward pass: encoder → intent classifier
            enc_out, _ = model.encoder(feats, x_lens)
            _ = model.intent_classifier(enc_out)

            # Plot activations
            audio_dir = out_dir / audio_name
            audio_dir.mkdir(exist_ok=True)
            

            for lname, act in layer_outputs.items():
                arr = act.squeeze(0).numpy()
                if arr.ndim == 1:
                    arr = arr[None, :]  # make (T=1, D)

                # plot_heatmap(arr, title=f"{audio_name} | {lname}", out_path=audio_dir / f"{lname}.png")

                # Plot top10 neurons and get indices
                topk_idx = plot_topk_neurons(arr, title=f"{audio_name} | {lname}", out_path=audio_dir / f"{lname}_top{10}.png", return_topk=True)

                if lname == "intent_classifier.9":
                    writer.writerow([audio_name, ",".join(map(str, topk_idx))])


            # Save top 10 neurons for layer 9
            

            act_layer9 = layer_outputs["intent_classifier.9"].squeeze(0).cpu().numpy()  # shape [T, D]
            mean_act_abs = np.abs(act_layer9).mean(axis=0)  # mean of absolute activations across time
            top10_indices = mean_act_abs.argsort()[-10:][::-1]  # descending order
            writer.writerow([audio_name, ",".join(map(str, top10_indices))])




    print(f"Done. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
