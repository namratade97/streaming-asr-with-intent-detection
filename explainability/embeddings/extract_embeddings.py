import csv
import torch
from lhotse import CutSet
from kaldifeat import Fbank, FbankOptions
import torch.nn as nn
from lhotse import Recording
import os
from typing import Dict, List, Optional, Tuple
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

# using icefall's files and definitions here

from model import AsrModel 
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))

class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


params = AttributeDict(
    {
        # Model architecture
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
        "chunk_size": "16,32,64,-1",
        # "chunk_size": "32",
        "left_context_frames": "512",
        "subsampling_factor": 4,
        "warm_step": 2000,
        "decoder_dim": 512,
        "use_transducer": True,
        "use_ctc": False,
        "vocab_size": 500,
    }
)





def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    encoder_embed = Conv2dSubsampling(
        in_channels=80,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed

def get_encoder_model(params: AttributeDict) -> nn.Module:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder

def get_model(params: AttributeDict) -> nn.Module:
    assert params.use_transducer or params.use_ctc, (
        f"At least one of them should be True, "
        f"but got params.use_transducer={params.use_transducer}, "
        f"params.use_ctc={params.use_ctc}"
    )

    encoder_embed = get_encoder_embed(params)
    encoder = get_encoder_model(params)
 
    decoder = None
    joiner = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AsrModel(
        encoder_embed=encoder_embed,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=max(_to_int_tuple(params.encoder_dim)),
        decoder_dim=params.decoder_dim,
        vocab_size=500,
        use_transducer=params.use_transducer,
        use_ctc=params.use_ctc,
    )


    return model


def get_init_states(
    model: nn.Module,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> List[torch.Tensor]:
    """
    Returns a list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
    is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
    states[-2] is the cached left padding for ConvNeXt module,
    of shape (batch_size, num_channels, left_pad, num_freqs)
    states[-1] is processed_lens of shape (batch,), which records the number
    of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.
    """
    states = model.encoder.get_init_states(batch_size, device)

    embed_states = model.encoder_embed.get_init_states(batch_size, device)
    states.append(embed_states)

    processed_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    states.append(processed_lens)

    return states


def stack_states(state_list: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """Stack list of zipformer states that correspond to separate utterances
    into a single emformer state, so that it can be used as an input for
    zipformer when those utterances are formed into a batch.

    Args:
      state_list:
        Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance. For element-n,
        state_list[n] is a list of cached tensors of all encoder layers. For layer-i,
        state_list[n][i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1,
        cached_val2, cached_conv1, cached_conv2).
        state_list[n][-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
        state_list[n][-1] is processed_lens of shape (batch,), which records the number
        of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Note:
      It is the inverse of :func:`unstack_states`.
    """
    batch_size = len(state_list)
    assert (len(state_list[0]) - 2) % 6 == 0, len(state_list[0])
    tot_num_layers = (len(state_list[0]) - 2) // 6

    batch_states = []
    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key = torch.cat(
            [state_list[i][layer_offset] for i in range(batch_size)], dim=1
        )
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn = torch.cat(
            [state_list[i][layer_offset + 1] for i in range(batch_size)], dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1 = torch.cat(
            [state_list[i][layer_offset + 2] for i in range(batch_size)], dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2 = torch.cat(
            [state_list[i][layer_offset + 3] for i in range(batch_size)], dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1 = torch.cat(
            [state_list[i][layer_offset + 4] for i in range(batch_size)], dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2 = torch.cat(
            [state_list[i][layer_offset + 5] for i in range(batch_size)], dim=0
        )
        batch_states += [
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        ]

    cached_embed_left_pad = torch.cat(
        [state_list[i][-2] for i in range(batch_size)], dim=0
    )
    batch_states.append(cached_embed_left_pad)

    processed_lens = torch.cat([state_list[i][-1] for i in range(batch_size)], dim=0)
    batch_states.append(processed_lens)

    return batch_states


def unstack_states(batch_states: List[Tensor]) -> List[List[Tensor]]:
    """Unstack the zipformer state corresponding to a batch of utterances
    into a list of states, where the i-th entry is the state from the i-th
    utterance in the batch.

    Note:
      It is the inverse of :func:`stack_states`.

    Args:
        batch_states: A list of cached tensors of all encoder layers. For layer-i,
          states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
          cached_conv1, cached_conv2).
          state_list[-2] is the cached left padding for ConvNeXt module,
          of shape (batch_size, num_channels, left_pad, num_freqs)
          states[-1] is processed_lens of shape (batch,), which records the number
          of processed frames (at 50hz frame rate, after encoder_embed) for each sample in batch.

    Returns:
        state_list: A list of list. Each element in state_list corresponding to the internal state
        of the zipformer model for a single utterance.
    """
    assert (len(batch_states) - 2) % 6 == 0, len(batch_states)
    tot_num_layers = (len(batch_states) - 2) // 6

    processed_lens = batch_states[-1]
    batch_size = processed_lens.shape[0]

    state_list = [[] for _ in range(batch_size)]

    for layer in range(tot_num_layers):
        layer_offset = layer * 6
        # cached_key: (left_context_len, batch_size, key_dim)
        cached_key_list = batch_states[layer_offset].chunk(chunks=batch_size, dim=1)
        # cached_nonlin_attn: (num_heads, batch_size, left_context_len, head_dim)
        cached_nonlin_attn_list = batch_states[layer_offset + 1].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val1: (left_context_len, batch_size, value_dim)
        cached_val1_list = batch_states[layer_offset + 2].chunk(
            chunks=batch_size, dim=1
        )
        # cached_val2: (left_context_len, batch_size, value_dim)
        cached_val2_list = batch_states[layer_offset + 3].chunk(
            chunks=batch_size, dim=1
        )
        # cached_conv1: (#batch, channels, left_pad)
        cached_conv1_list = batch_states[layer_offset + 4].chunk(
            chunks=batch_size, dim=0
        )
        # cached_conv2: (#batch, channels, left_pad)
        cached_conv2_list = batch_states[layer_offset + 5].chunk(
            chunks=batch_size, dim=0
        )
        for i in range(batch_size):
            state_list[i] += [
                cached_key_list[i],
                cached_nonlin_attn_list[i],
                cached_val1_list[i],
                cached_val2_list[i],
                cached_conv1_list[i],
                cached_conv2_list[i],
            ]

    cached_embed_left_pad_list = batch_states[-2].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(cached_embed_left_pad_list[i])

    processed_lens_list = batch_states[-1].chunk(chunks=batch_size, dim=0)
    for i in range(batch_size):
        state_list[i].append(processed_lens_list[i])

    return state_list


def streaming_forward(
    features: Tensor,
    feature_lens: Tensor,
    model: nn.Module,
    states: List[Tensor],
    chunk_size: int,
    left_context_len: int,
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """
    Returns encoder outputs, output lengths, and updated states.
    """
    cached_embed_left_pad = states[-2]
    (x, x_lens, new_cached_embed_left_pad,) = model.encoder_embed.streaming_forward(
        x=features,
        x_lens=feature_lens,
        cached_left_pad=cached_embed_left_pad,
    )
    # assert x.size(1) == chunk_size, (x.size(1), chunk_size)

    src_key_padding_mask = make_pad_mask(x_lens)

    # processed_mask is used to mask out initial states
    processed_mask = torch.arange(left_context_len, device=x.device).expand(
        x.size(0), left_context_len
    )
    processed_lens = states[-1]  # (batch,)
    # (batch, left_context_size)
    processed_mask = (processed_lens.unsqueeze(1) <= processed_mask).flip(1)
    # Update processed lengths
    new_processed_lens = processed_lens + x_lens

    # (batch, left_context_size + chunk_size)
    src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

    x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
    encoder_states = states[:-2]
    (
        encoder_out,
        encoder_out_lens,
        new_encoder_states,
    ) = model.encoder.streaming_forward(
        x=x,
        x_lens=x_lens,
        states=encoder_states,
        src_key_padding_mask=src_key_padding_mask,
    )
    encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

    new_states = new_encoder_states + [
        new_cached_embed_left_pad,
        new_processed_lens,
    ]
    return encoder_out, encoder_out_lens, new_states



# extract_embeddings_from_cuts.py
import os
import csv
import torch
import numpy as np
from pathlib import Path
from lhotse import CutSet
from kaldifeat import Fbank, FbankOptions



DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

CUTS_JSONL = "/Users/nde/Downloads/fbank/slurp_cuts_test.jsonl"
CHECKPOINT = "/Users/nde/Downloads/epoch-200.pt"   
OUT_CSV = "slurp_test_embeddings.csv"

AUDIO_PREFIX = None #"/Users/nde/Downloads"  # set to None to use cut paths as-is


model = get_model(params)
# load checkpoint
ckpt = torch.load(CHECKPOINT, map_location="cpu")
state_dict = ckpt.get("model", ckpt)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
model.device = DEVICE

fbank_opts = FbankOptions()
fbank_opts.device = DEVICE
fbank_opts.frame_opts.dither = 0
fbank_opts.frame_opts.snip_edges = False
fbank_opts.frame_opts.samp_freq = 16000
fbank_opts.mel_opts.num_bins = 80
fbank = Fbank(fbank_opts)

# Load cuts (jsonl)
cuts = CutSet.from_jsonl(CUTS_JSONL)

rows = []  # collect (audio_id, embedding np.array)
first_dim = None

for cut in cuts:
    audio_id = cut.id
    orig_source = cut.recording.sources[0].source

    if AUDIO_PREFIX:
        if orig_source.startswith("download/"):
            rel = orig_source[len("download/") :]
            candidate = os.path.join(AUDIO_PREFIX, "audio", rel)  # e.g. /Users/nde/Downloads/audio/test/...
        else:
            candidate = os.path.join(AUDIO_PREFIX, orig_source.lstrip("/"))
        audio_path_to_load = candidate
    else:
        audio_path_to_load = orig_source

    
    try:
        audio = cut.load_audio(path=audio_path_to_load)
    except Exception as e:
        print(f"[WARN] cut {audio_id}: failed to load from override path {audio_path_to_load}: {e}")
        audio = cut.load_audio()

    assert audio.ndim == 2 and audio.shape[0] == 1, f"Unexpected audio shape for {audio_id}: {audio.shape}"
    samples = torch.from_numpy(audio).squeeze(0).to(DEVICE)  # (num_samples,)

    # Fbank -> feature tensor (T, 80)
    features = fbank(samples)  # tensor on DEVICE

    # features -> (1, T, F)
    # Make features 4D: (B, C, T, F) for Conv2d
    features_batch = features.unsqueeze(0).unsqueeze(1)      # (1, 1, T, F)

    # Pad time dimension to at least first conv kernel size
    min_T = 7  # first Conv2d kernel size in time
    if features_batch.size(2) < min_T:
        pad_len = min_T - features_batch.size(2)
        features_batch = torch.nn.functional.pad(features_batch, (0, 0, 0, pad_len))

    feature_lens = torch.tensor([features_batch.size(2)], device=DEVICE)

    # Init streaming states
    streaming_states = get_init_states(model, batch_size=1, device=DEVICE)

    # Run encoder_embed + streaming encoder
    raw_encoder_out, encoder_out_lens, streaming_states = streaming_forward(
        features=features_batch,
        feature_lens=feature_lens,
        model=model,
        states=streaming_states,
        chunk_size=features_batch.size(2),   # feed all frames at once
        left_context_len=0,
    )

    # raw_encoder_out: (N, T, C)  -> squeeze batch
    raw_encoder_out = raw_encoder_out.squeeze(0)

    # Take mean over time to get utterance-level embedding
    avg_embedding = raw_encoder_out.mean(dim=0).cpu().numpy()

    print("Embedding shape:", avg_embedding.shape)




