import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
from icefall.utils import AttributeDict
from .train import get_model
from .zipformer import DownsampledZipformer2Encoder
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import pandas as pd

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
            "warm_step": 2000,
            "decoder_dim": 512,
            "use_transducer": True,
            "use_ctc": False,
            "vocab_size": 500,
            "blank_id": 0,
            "context_size": 4,
            "decoder_dim": 512,
            "joiner_dim": 512,
            "decoding_method": "greedy_search",
            "num_decode_streams": 1,
            "batch_size": 1,
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


def ensure_time_neuron_matrix(arr):
    """
    Convert numpy array `arr` of unknown shape into (T, D) where
    T = time frames and D = neurons/features.
    Returns arr2d or None if impossible.
    Rules:
      - If arr.ndim == 3 and shape like (1, T, D) or (T, 1, D) or (T, D, 1) try to squeeze appropriately.
      - If arr.ndim == 2 treat as (T, D).
      - If arr.ndim == 1 treat as (T=arr.shape[0], D=1) or (T=1, D=arr_len) depending on context.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        # prefer (batch, time, dim) => squeeze batch if 1
        if a.shape[0] == 1:
            a2 = a.squeeze(0)  # (T, D)
            if a2.ndim == 2:
                return a2
        # else maybe (T, batch, D)
        if a.shape[1] == 1:
            a2 = a.squeeze(1)  # (T, D)
            if a2.ndim == 2:
                return a2
        # fallback: collapse middle axis (take mean across batch)
        a_mean = a.mean(axis=1)
        if a_mean.ndim == 2:
            return a_mean
        return None
    elif a.ndim == 2:
    
        return a
    elif a.ndim == 1:
        #treat this as (T, D=1)
        return a.reshape(-1, 1)
    else:
        return None


def normalize_activations_timewise(mat, method="zscore", clip_std=3.0):
    """
    mat: (T, D)
    method:
      - "zscore": per-neuron (column) z-score across time
      - "minmax": per-neuron minmax to [0,1]
      - "global_z": zscore on whole matrix
    Returns normalized matrix (T, D) as float32.
    """
    mat = mat.astype(np.float32)
    if method == "zscore":
        # z-score per neuron (column)
        mu = mat.mean(axis=0, keepdims=True)
        sigma = mat.std(axis=0, keepdims=True) + 1e-8
        z = (mat - mu) / sigma
        z = np.clip(z, -clip_std, clip_std)  # clip extremes for visualization
        return z
    elif method == "minmax":
        mn = mat.min(axis=0, keepdims=True)
        mx = mat.max(axis=0, keepdims=True)
        denom = (mx - mn) + 1e-8
        return (mat - mn) / denom
    elif method == "global_z":
        mu = mat.mean()
        sigma = mat.std() + 1e-8
        z = (mat - mu) / sigma
        z = np.clip(z, -clip_std, clip_std)
        return z
    else:
        return mat



def plot_detailed_layerwise_heatmaps(all_activations, out_dir, frame_shift=0.01, top_k=10):
    """
    Visualizes activations in multiple ways:
      1. Heatmap per (block, layer) showing neuron activations over time.
      2. Top-K neuron time courses per layer.
      3. Blockwise summary grid (layer timeline).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for audio_name, layer_dict in all_activations.items():
        print(f"Plotting detailed activations for {audio_name}")
        audio_dir = out_dir / audio_name
        audio_dir.mkdir(exist_ok=True)

        #  Normalize across all layers to comparable scale 
        all_vals = np.concatenate([v.flatten() for v in layer_dict.values()])
        global_mean, global_std = all_vals.mean(), all_vals.std() + 1e-6

        #  Individual layer plots 
        for lname, act in layer_dict.items():
            act_np = act.squeeze(0).numpy()  # (T, D)
            act_np = (act_np - global_mean) / global_std
            act_np = gaussian_filter1d(act_np, sigma=1, axis=0)
            times = np.arange(act_np.shape[0]) * frame_shift

            # Parse block/layer for naming
            block_idx, layer_idx = 0, 0
            if "block" in lname:
                try:
                    block_idx = int(lname.split("block")[1].split("_")[0])
                except:
                    pass
            if "layer" in lname:
                try:
                    layer_idx = int(lname.split("layer")[1].split("_")[0])
                except:
                    pass

            # Heatmap
            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(
                act_np.T,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=[times[0], times[-1], 0, act_np.shape[1]],
            )
            ax.set_title(f"{audio_name} | Block {block_idx} Layer {layer_idx}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Neuron index")
            plt.colorbar(im, ax=ax, label="Normalized Activation")
            plt.tight_layout()
            fig.savefig(audio_dir / f"{lname}_heatmap.png", dpi=200)
            plt.close(fig)

            # Top-K neuron traces
            topk_idx = np.argsort(-np.abs(act_np).mean(axis=0))[:top_k]
            plt.figure(figsize=(10, 4))
            for i in topk_idx:
                plt.plot(times, act_np[:, i], label=f"Neuron {i}")
            plt.legend(loc="upper right", ncol=2)
            plt.xlabel("Time (s)")
            plt.ylabel("Activation (norm.)")
            plt.title(f"Top-{top_k} active neurons | {audio_name} | {lname}")
            plt.tight_layout()
            plt.savefig(audio_dir / f"{lname}_top{top_k}.png", dpi=200)
            plt.close()

        #  Blockwise summary (grid) 
        blocks = sorted(set([k.split("_")[0] for k in layer_dict.keys() if "block" in k]))
        for bname in blocks:
            layers_in_block = {k: v for k, v in layer_dict.items() if k.startswith(bname)}
            if not layers_in_block:
                continue
            n_layers = len(layers_in_block)
            fig, axes = plt.subplots(
                n_layers, 1, figsize=(12, 2.5 * n_layers), sharex=True
            )
            if n_layers == 1:
                axes = [axes]

            for ax, (lname, act) in zip(axes, sorted(layers_in_block.items())):
                act_np = act.squeeze(0).numpy()
                act_np = (act_np - global_mean) / global_std
                act_np = gaussian_filter1d(act_np, sigma=1, axis=0)
                times = np.arange(act_np.shape[0]) * frame_shift
                im = ax.imshow(
                    act_np.T,
                    aspect="auto",
                    origin="lower",
                    cmap="magma",
                    extent=[times[0], times[-1], 0, act_np.shape[1]],
                )
                ax.set_ylabel(lname.split("_")[1])  # layer number
            axes[-1].set_xlabel("Time (s)")
            fig.colorbar(im, ax=axes, location="right", shrink=0.6, label="Normalized Activation")
            fig.suptitle(f"Block {bname} summary - {audio_name}")
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(audio_dir / f"{bname}_summary.png", dpi=200)
            plt.close(fig)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="phoneme_layerwise_activations_2")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    frame_shift = 0.01 
    top_k = 10   

    device = torch.device(args.device if args.device in ("cpu", "cuda", "mps") else "cpu")
    print("Using device:", device)

    params = build_params()
    model = get_model(params)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        loaded = False
        for k in ckpt.keys():
            if "model" in k:
                model.load_state_dict(ckpt[k])
                loaded = True
                break
        if not loaded:
            model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # Collect layers
    
    layer_outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output.detach().cpu()
        return hook

    def hook_all_linears(module, prefix=""):
        for name, sub_module in module.named_children():
            sub_prefix = f"{prefix}_{name}" if prefix else name
            if isinstance(sub_module, torch.nn.Linear):
                sub_module.register_forward_hook(hook_fn(sub_prefix))
                print(f"Hooked: {sub_prefix}")
            else:
                hook_all_linears(sub_module, sub_prefix)

    for block_idx, block in enumerate(model.encoder.encoders):
        print("block_idx:", block_idx)
        # Handle DownsampledZipformer2Encoder wrapping
        if isinstance(block, DownsampledZipformer2Encoder):
            zip_encoder = block.encoder
        else:
            zip_encoder = block

        for layer_idx, layer in enumerate(zip_encoder.layers):
            hook_all_linears(layer, prefix=f"block{block_idx}_layer{layer_idx}")


    def plot_neuron_time_heatmap(activations_time: np.ndarray,
                             audio_name: str,
                             block_idx: int,
                             layer_idx: int,
                             out_dir: str,
                             vmin=None,
                             vmax=None):
        """
        Plot heatmap of activations over time for each neuron in one layer.
        activations_time shape: (T, D)  where T = time‑frames, D = neurons.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{audio_name}_block{block_idx}_layer{layer_idx}.png"

        plt.figure(figsize=(12, 6))
        # Here we just plot the raw activations_time
        plt.imshow(activations_time.T, aspect='auto', cmap='viridis',
                vmin=vmin, vmax=vmax)
        plt.colorbar(label="Activation")
        plt.xlabel("Time frame")
        plt.ylabel("Neuron index")
        D = activations_time.shape[1]
        yticks = np.linspace(0, D-1, min(10, D)).astype(int)
        plt.yticks(yticks, [f"n{idx}" for idx in yticks])
        plt.title(f"{audio_name} ‑ Block {block_idx} Layer {layer_idx}")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    def plot_summary_all_layers(activations_by_layer: dict,
                                audio_name: str,
                                block_idx: int,
                                out_dir: str,
                                vmin=None, vmax=None):
        """
        Concatenate all layers of a block vertically and plot one big heatmap.
        activations_by_layer: dict mapping layer_idx -> np.ndarray (T, D_layer).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{audio_name}_block{block_idx}_all_layers.png"

        
        mats = []
        for layer_idx in sorted(activations_by_layer.keys()):
            mats.append(activations_by_layer[layer_idx].T)  # neurons x time
        big_mat = np.vstack(mats)

        plt.figure(figsize=(12, 6))
        plt.imshow(big_mat, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar(label="Activation")
        plt.xlabel("Time frame")
        plt.ylabel("Neuron index (stacked layers)")
        plt.title(f"{audio_name} ‑ Block {block_idx} ‑ All Layers")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
    
    
    audio_folder = Path(args.audio)
    audio_files = sorted(audio_folder.glob("*.flac"))
    if not audio_files:
        raise ValueError(f"No .flac files in {audio_folder}")

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)

    all_activations = defaultdict(dict)

    for audio_path in tqdm(audio_files, desc="Processing audios"):
        audio_name = audio_path.stem
        audio_dir = out_dir / audio_name
        audio_dir.mkdir(exist_ok=True)

        feats = wav_to_logmel(audio_path).unsqueeze(0).to(device)
        x_lens = torch.tensor([feats.shape[1]], device=device)
        y_dummy = torch.tensor([[0]], device=device)

        # clear layer_outputs before each audio so old hooks don't leak
        layer_outputs.clear()

        # Run encoder only (hooks already capture layer_outputs)
        _ = model.encoder(feats, x_lens)

        # For each hooked layer, plot activations over time
        # Each act: shape (1, T, D)
        block_layer_dict = defaultdict(dict)

        for lname, act in list(layer_outputs.items()):
            # get raw numpy
            act_raw = act.detach().cpu().numpy()
            mat = ensure_time_neuron_matrix(act_raw)  # returns (T, D) or None
            if mat is None:
                print(f"Skipping {lname}: can't coerce shape {act_raw.shape} to (T,D)")
                continue

            # optional smoothing (time axis)
            mat = gaussian_filter1d(mat, sigma=1.0, axis=0)

            # normalize per neuron
            mat_norm = normalize_activations_timewise(mat, method="zscore", clip_std=3.0)
            


            # pick vmin/vmax from percentiles to avoid outliers dominating colors
            vmin = np.percentile(mat_norm, 1)
            vmax = np.percentile(mat_norm, 99)
            if vmin == vmax:
                vmin, vmax = mat_norm.min(), mat_norm.max()

            block_match = re.search(r"block(\d+)", lname)
            layer_match = re.search(r"layer(\d+)", lname)
            block_idx = int(block_match.group(1)) if block_match else -1
            layer_idx = int(layer_match.group(1)) if layer_match else -1

            # plot heatmap: (time x neurons) plotted as neurons on Y (we want neurons on Y)
            times = np.arange(mat_norm.shape[0]) * frame_shift
            if len(times) == 1:
                # expand by one frame for visualization
                times = np.array([0, frame_shift])
                mat_norm = np.vstack([mat_norm, mat_norm])  # duplicate row

            fig, ax = plt.subplots(figsize=(12, 5))
            im = ax.imshow(mat_norm.T, aspect="auto", origin="lower",
                        extent=[times[0], times[-1], 0, mat_norm.shape[1]],
                        vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Neuron index")
            #show only a few neuron labels to avoid clutter
            D = mat_norm.shape[1]
            ytick_locs = np.linspace(0, D, min(10, D), endpoint=False)
            ytick_labels = [f"n{int(x)}" for x in (np.linspace(0, D-1, min(10, D))).astype(int)]
            ax.set_yticks(ytick_locs)
            ax.set_yticklabels(ytick_labels)
            plt.title(f"{audio_name} | {lname}")
            fig.colorbar(im, ax=ax, label="zscore (per neuron)")
            fig.tight_layout()
            fig.savefig(audio_dir / f"{lname}_time_heatmap.png", dpi=200)
            plt.close(fig)

            # Top-K neuron traces 
            topk_idx = np.argsort(-np.abs(mat_norm).mean(axis=0))[:top_k]
            plt.figure(figsize=(10, 4))
            for i in topk_idx:
                plt.plot(times, mat_norm[:, i], label=f"Neuron {i}")
            plt.legend(loc="upper right", ncol=2, fontsize="small")
            plt.xlabel("Time (s)")
            plt.ylabel("Activation (zscore)")
            plt.title(f"Top-{top_k} neurons | {audio_name} | {lname}")
            plt.tight_layout()
            plt.savefig(audio_dir / f"{lname}_top{top_k}.png", dpi=200)
            plt.close()


            # Collect for summary plot per block
            all_activations[audio_name][lname] = torch.tensor(mat_norm)
            block_layer_dict[layer_idx] = mat_norm

        # Blockwise summary
        # for block_idx, layers in block_layer_dict.items():
        plot_summary_all_layers(
            activations_by_layer=block_layer_dict,
            audio_name=audio_name,
            block_idx=block_idx,
            out_dir=out_dir,
        )

    # Save mean activations to CSV for numeric analysis
    rows = []
    for audio, layers in all_activations.items():
        for layer, act in layers.items():
            mat = np.asarray(act)
            mat2 = ensure_time_neuron_matrix(mat)
            if mat2 is None:
                continue
            T, D = mat2.shape
            # mean per neuron
            mean_per_neuron = mat2.mean(axis=0).tolist()
            for i, val in enumerate(mean_per_neuron):
                rows.append({"audio": audio, "layer": layer, "neuron": i, "activation": float(val)})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "layerwise_mean_activations_2.csv", index=False)
    print(f"Saved mean activations CSV to {out_dir}/layerwise_mean_activations_2.csv")

    # # Visualize
    plot_detailed_layerwise_heatmaps(all_activations, out_dir)



if __name__ == "__main__":
    main()


# python -m icefall.egs.librispeech.ASR.zipformer.analyze_phoneme_neurons_2 --outdir /Volumes/DataN/streaming-asr-with-intent-detection-results/phoneme_layerwise_activation_slurp_xai --checkpoint /Users/nde/Downloads/epoch-200.pt --audio "/Users/nde/Downloads/audio/test_slurp_xai/"