import torch
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from lhotse import CutSet
from kaldifeat import Fbank, FbankOptions
import numpy as np

# using icefall's files here

from .decode_stream import DecodeStream
from .train import get_model
from .streaming_decode import get_init_states, decode_one_chunk
from icefall.utils import AttributeDict

CUTS_JSONL = "/Users/nde/Downloads/fbank_test_cuts/slurp_cuts_test_later.jsonl"
CHECKPOINT = "/Users/nde/Downloads/epoch-200.pt"
OUT_CSV = "slurp_test_mean_embeddings.csv"

def extract_embeddings():
    params = AttributeDict(
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
            "joiner_dim": 512,
            "decoding_method": "greedy_search",
            "num_decode_streams": 1,
            "batch_size": 1,
        }
    )

    #  Load model 
    model = get_model(params)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    #  Load cuts 
    cut_set = CutSet.from_jsonl(CUTS_JSONL)

    #  Fbank extractor via kaldifeat 
    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    fbank = Fbank(opts)

    with open(OUT_CSV, "w") as f:
        f.write("cut_id,mean_embedding\n")

        for cut in cut_set:
            initial_states = get_init_states(model=model, batch_size=1, device=device)
            decode_stream = DecodeStream(
                params=params,
                cut_id=cut.id,
                initial_states=initial_states,
                decoding_graph=None,
                device=device,
            )

            audio = cut.load_audio()
            assert len(audio.shape) == 2 and audio.shape[0] == 1
            
            samples = torch.from_numpy(audio).squeeze(0).to(device)

            feature = fbank(samples)  # Shape: (T, feature_dim)

            decode_stream.set_features(feature, tail_pad_len=30)

            finished_streams = []
            all_chunk_embeddings = []  

            while len(finished_streams) < 1:
                finished_streams, raw_encoder_out = decode_one_chunk(
                    params=params, model=model, decode_streams=[decode_stream]
                )
                
                if raw_encoder_out is not None and raw_encoder_out.numel() > 0:
                    raw_encoder_out = raw_encoder_out.squeeze(0)  # Shape: [T_chunk, encoder_dim]
                    all_chunk_embeddings.append(raw_encoder_out.detach().cpu())

            #  Compute Mean Pooling across the time dimension 
            if all_chunk_embeddings:
                # Stack all chunks together
                encoder_output = torch.cat(all_chunk_embeddings, dim=0)  # Shape: [T_total, 512]
                
                # Calculate average across time (dim=0)
                mean_embedding = torch.mean(encoder_output, dim=0)  # Shape: [512]
                mean_embedding_np = mean_embedding.numpy()

                embedding_str = ",".join(map(str, mean_embedding_np.tolist()))
                f.write(f"{cut.id},{embedding_str}\n")
            else:
                print(f"Warning: No embeddings extracted for audio sample {cut.id}")

            del decode_stream, finished_streams, feature, all_chunk_embeddings
            if 'encoder_output' in locals(): del encoder_output
            if 'mean_embedding' in locals(): del mean_embedding
            torch.cuda.empty_cache()

if __name__ == "__main__":
    extract_embeddings()