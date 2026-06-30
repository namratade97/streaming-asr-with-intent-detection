import torch
from torch.nn.utils.rnn import pad_sequence
import json
from pathlib import Path
from lhotse import CutSet
from kaldifeat import Fbank, FbankOptions
import numpy as np

from .decode_stream import DecodeStream
from .train import get_model
from .streaming_decode import get_init_states, decode_one_chunk
from icefall.utils import AttributeDict

CUTS_JSONL = "/Users/nde/Downloads/fbank/slurp_cuts_test.jsonl"
CHECKPOINT = "/Users/nde/Downloads/epoch-200.pt"
OUT_CSV_MEAN = "slurp_test_intent_mean_embeddings.csv"
OUT_CSV_FRAME = "slurp_test_intent_frame_embeddings.csv"

def extract_intent_embeddings():
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
            "decoder_dim": 512,
            "joiner_dim": 512,
            "decoding_method": "greedy_search",
            "num_decode_streams": 1,
            "batch_size": 1,
        }
    )

    model = get_model(params)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cut_set = CutSet.from_jsonl(CUTS_JSONL)

    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80
    fbank = Fbank(opts)

    f_mean = open(OUT_CSV_MEAN, "w")
    f_frame = open(OUT_CSV_FRAME, "w")
    f_mean.write("cut_id,intent_mean_embedding\n")
    f_frame.write("cut_id,frame_idx,intent_embedding\n")

    cut_intent = {}
    with open(CUTS_JSONL, "r") as f:
        for line in f:
            cut = json.loads(line)
            cid = str(cut["id"]).strip()
            text = cut["supervisions"][0]["text"]
            intent = text[text.find("<")+1:text.find(">")]
            cut_intent[cid] = intent

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
        samples = torch.from_numpy(audio).squeeze(0).to(device)

        feature = fbank(samples)
        decode_stream.set_features(feature, tail_pad_len=30)

        finished_streams = []
        all_chunk_embeddings = []

        while len(finished_streams) < 1:
            finished_streams, raw_encoder_out = decode_one_chunk(
                params=params, model=model, decode_streams=[decode_stream]
            )

            raw_encoder_out = raw_encoder_out.squeeze(0)  # [T_chunk, encoder_dim]
            all_chunk_embeddings.append(raw_encoder_out.cpu())

        encoder_output = torch.cat(all_chunk_embeddings, dim=0).to(device)  # [T_total, encoder_dim]

        # Get intent head embeddings per frame
        with torch.no_grad():
            intent_logits = model.intent_classifier(encoder_output.unsqueeze(0))  # [1, T, 92]
            intent_logits = intent_logits.squeeze(0)  # [T, 92]

            # Save frame-wise embeddings
            for t, vec in enumerate(intent_logits):
                f_frame.write(f"{cut.id},{t},{','.join(map(str, vec.cpu().numpy()))}\n")

            # Save mean-pooled embedding
            mean_intent = intent_logits.mean(dim=0)  # [92]
            f_mean.write(f"{cut.id},{','.join(map(str, mean_intent.cpu().numpy()))}\n")

        del decode_stream, finished_streams, feature, encoder_output, all_chunk_embeddings
        torch.cuda.empty_cache()

    f_mean.close()
    f_frame.close()


if __name__ == "__main__":
    extract_intent_embeddings()

# (kaldifeat) nde@Namratas-MacBook-Pro streaming-asr-with-intent-detection % python -m icefall.egs.librispeech.ASR.zipformer.extract_intent_head_embedding_wrapper