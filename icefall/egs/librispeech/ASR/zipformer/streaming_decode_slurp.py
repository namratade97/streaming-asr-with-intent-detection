#!/usr/bin/env python3
# Copyright 2022-2023 Xiaomi Corporation (Authors: Wei Kang,
#                                                  Fangjun Kuang,
#                                                  Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:
./zipformer/streaming_decode.py \
  --epoch 28 \
  --avg 15 \
  --causal 1 \
  --chunk-size 32 \
  --left-context-frames 256 \
  --exp-dir ./zipformer/exp \
  --decoding-method greedy_search \
  --num-decode-streams 2000
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import numpy as np
import sentencepiece as spm
import torch
# from asr_datamodule import LibriSpeechAsrDataModule
from asr_datamodule_slurp import SlurpAsrDataModule #####
from decode_stream import DecodeStream
from kaldifeat import Fbank, FbankOptions
from lhotse import CutSet
from streaming_beam_search import (
    fast_beam_search_one_best,
    greedy_search,
    modified_beam_search,
)
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from train import add_model_arguments, get_model, get_params

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.utils import (
    AttributeDict,
    make_pad_mask,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)
import os
print("Current Working Directory:", os.getcwd())

from model import some_mapping
intent_id2label = { v: k for k, v in some_mapping.items() }

from collections import defaultdict
import csv

# ---------- relevance helper functions--------------------------------
def token_relevance(frame_scores, hyp_tokens):
    """
    frame_scores : 1D tensor of length F  (relevance per encoder frame)
    hyp_tokens   : list[str]               (decoded word pieces)

    We simply give each token an equal slice of frames:
    """
    F = frame_scores.numel() # total number of encoder frames in the utterance
    n = len(hyp_tokens) # number of tokens tokens that appeared in the final ASR hypothesis
    if n == 0:
        return []

    frames_per_tok = F / n  # **evenly** divide the F frames into n buckets
    tok_scores = []
    for j, tok in enumerate(hyp_tokens):
        s = int(round(j * frames_per_tok)) # start frame index (inclusive)
        e = int(round((j + 1) * frames_per_tok)) # end   frame index (exclusive)
        tok_scores.append(frame_scores[s:e].sum().item()) #   sum of relevance over the slice that we assigned to this token

    tot = sum(tok_scores) + 1e-8 # avoid dividing by zero
    tok_scores = [x / tot for x in tok_scores]
    # pick top‑3
    top3 = sorted(
        list(zip(hyp_tokens, tok_scores)),
        key=lambda x: x[1], reverse=True
    )[:3]
    return top3
# ------------------------------------------------------------- #####

def token_relevance_with_times(frame_scores, hyp_tokens, token_times):
    from collections import defaultdict

    if len(hyp_tokens) == 0 or len(token_times) == 0:
        return []

    F = len(frame_scores)
    token_times = list(token_times)
    # Append end frame for last token's boundary
    if token_times[-1] < F:
        token_times.append(F)

    tok_scores = []
    prev = 0
    for i, tok in enumerate(hyp_tokens):
        start = prev
        end = token_times[i] if i < len(token_times) else F
        if end <= start:
            end = start + 1  # at least one frame
        # score = frame_scores[start:end].sum().item()
        length = end - start
        if length == 0:
            continue
        score = frame_scores[start:end].sum().item() / length
        tok_scores.append(score)
        prev = end

    tot = sum(tok_scores) + 1e-8 # avoid dividing by zero
    tok_scores = [x / tot for x in tok_scores]

    top3 = sorted(zip(hyp_tokens, tok_scores), key=lambda x: x[1], reverse=True)[:3]
    return top3


#####################

# ---------- helper -----------------
def classifier_weight(cls):
    """
    Return the weight matrix (C × I) of the final linear layer that maps
    encoder dim → N_intents.
    Works whether `cls` is nn.Linear or nn.Sequential(..., nn.Linear).
    """
    if isinstance(cls, torch.nn.Linear):
        return cls.weight
    # assume the last module is the linear projection
    return next(m.weight for m in reversed(cls) if isinstance(m, torch.nn.Linear))
# ----------------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=28,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=15,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        # default="data/lang_bpe_592/libri_intent_592.model", #####
        # default="data/lang_bpe_500/bpe.model",
        default="data/updated_tokenizer_slurp/cslurp_trained_without_intents_added_intents_without_retrain.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Supported decoding methods are:
        greedy_search
        modified_beam_search
        fast_beam_search
        """,
    )

    parser.add_argument(
        "--num_active_paths",
        type=int,
        default=4,
        help="""An interger indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=4,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search""",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="""Used only when --decoding-method is
        fast_beam_search""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--num-decode-streams",
        type=int,
        default=2000,
        help="The number of streams that can be decoded parallel.",
    )

    add_model_arguments(parser)

    return parser


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
    assert x.size(1) == chunk_size, (x.size(1), chunk_size)

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


def decode_one_chunk(
    params: AttributeDict,
    model: nn.Module,
    decode_streams: List[DecodeStream],
) -> List[int]:
    """Decode one chunk frames of features for each decode_streams and
    return the indexes of finished streams in a List.

    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      decode_streams:
        A List of DecodeStream, each belonging to a utterance.
    Returns:
      Return a List containing which DecodeStreams are finished.
    """
    device = model.device
    chunk_size = int(params.chunk_size)
    left_context_len = int(params.left_context_frames)

    features = []
    feature_lens = []
    states = []
    processed_lens = []  # Used in fast-beam-search

    for stream in decode_streams:
        feat, feat_len = stream.get_feature_frames(chunk_size * 2)
        features.append(feat)
        feature_lens.append(feat_len)
        states.append(stream.states)
        processed_lens.append(stream.done_frames)

    feature_lens = torch.tensor(feature_lens, device=device)
    features = pad_sequence(features, batch_first=True, padding_value=LOG_EPS)

    # Make sure the length after encoder_embed is at least 1.
    # The encoder_embed subsample features (T - 7) // 2
    # The ConvNeXt module needs (7 - 1) // 2 = 3 frames of right padding after subsampling
    tail_length = chunk_size * 2 + 7 + 2 * 3
    if features.size(1) < tail_length:
        pad_length = tail_length - features.size(1)
        feature_lens += pad_length
        features = torch.nn.functional.pad(
            features,
            (0, 0, 0, pad_length),
            mode="constant",
            value=LOG_EPS,
        )

    states = stack_states(states)

    # encoder_out, encoder_out_lens, new_states = streaming_forward(
    #     features=features,
    #     feature_lens=feature_lens,
    #     model=model,
    #     states=states,
    #     chunk_size=chunk_size,
    #     left_context_len=left_context_len,
    # )

    # encoder_out = model.joiner.encoder_proj(encoder_out)

    # 1) run the streaming encoder
    raw_encoder_out, encoder_out_lens, new_states = streaming_forward(
        features=features,
        feature_lens=feature_lens,
        model=model,
        states=states,
        chunk_size=chunk_size,
        left_context_len=left_context_len,
    )

    # 2) intent‐head accumulation (purely from the encoder)
    # raw_encoder_out: (N, T, C)
    intent_logits = model.intent_classifier(raw_encoder_out)  # (N, T, 92)

    # mask out padding frames
    mask = torch.arange(raw_encoder_out.size(1), device=raw_encoder_out.device) \
            .unsqueeze(0) < encoder_out_lens.unsqueeze(1)    # (N, T)
    intent_logits[~mask] = 0.0

    # sum over time and accumulate per stream
    chunk_sum = intent_logits.sum(dim=1)  # (N, 92)
    for i, stream in enumerate(decode_streams):
        stream.intent_logit_sum += chunk_sum[i]
        stream.total_frames      += encoder_out_lens[i]
        # ---- record running intent confidence per frame ----
        mean_logits = stream.intent_logit_sum / stream.total_frames
        probs       = torch.softmax(mean_logits, dim=-1).cpu().tolist()
        frame_idx   = stream.total_frames  # absolute encoder‑frame count so far
        stream.intent_history.append((frame_idx, probs))
    ################################################################
    # keep full frame × intent relevance for later XAI 
    # weight_t = classifier_weight(model.intent_classifier).T

    # Project encoder outputs first
    # projected = model.intent_classifier[0](raw_encoder_out)  # (N, T, 128)
    # projected = model.intent_classifier[1](projected)        # ReLU  
    
    # weight_t = model.intent_classifier[2].weight.T ###NEW
    # weight_t = model.intent_classifier[1].weight.T ###NEW  # (128, 92)

    # frame_rels = torch.einsum(
    #     "ntc,ci->nti",                # (N,T,C)·(C,I) → (N,T,I)
    #     raw_encoder_out,              # encoder frames # (N, T, C)
    #     weight_t                      # (C, N_intents)
    # )                                 # (N,T,92)

    # proj = model.intent_classifier[0](raw_encoder_out)  # (N,T,128) ### when using bigger MLP Head
    # proj = model.intent_classifier[1](proj)             # ReLU
    # weight_t = model.intent_classifier[2].weight.T      # (128, 92)

    # 1) First projection
    proj = model.intent_classifier[0](raw_encoder_out)   # Linear(encoder_dim → 1024)
    proj = model.intent_classifier[1](proj)              

    proj = model.intent_classifier[2](proj)              
    proj = model.intent_classifier[3](proj)              
    
    proj = model.intent_classifier[4](proj)              
    proj = model.intent_classifier[5](proj)              
    proj = model.intent_classifier[6](proj)              

    proj = model.intent_classifier[7](proj)              

    proj = model.intent_classifier[8](proj)              
    
    # proj = model.intent_classifier[9](proj)              
    # proj = model.intent_classifier[10](proj)             
    # proj = model.intent_classifier[11](proj)
    # proj = model.intent_classifier[12](proj)

    # proj = model.intent_classifier[13](proj)

    # Grab final classification weight
    weight_t = model.intent_classifier[9].weight.T      # Linear(128→92)

    # Apply classifier manually
    frame_rels = torch.einsum("ntc,ci->nti", proj, weight_t)  # (N,T,92)

    # frame_rels = torch.einsum("ntc,ci->nti", projected, weight_t)  # (N, T, 92)

    for i, stream in enumerate(decode_streams):
        t = encoder_out_lens[i].item()          # valid frames in this chunk
        stream.frame_rels.append(frame_rels[i, :t].cpu())
    ################################################################

    # 3) now continuing with normal joiner projection + search
    encoder_out = model.joiner.encoder_proj(raw_encoder_out)

    if params.decoding_method == "greedy_search":
        greedy_search(model=model, encoder_out=encoder_out, streams=decode_streams)
    elif params.decoding_method == "fast_beam_search":
        processed_lens = torch.tensor(processed_lens, device=device)
        processed_lens = processed_lens + encoder_out_lens
        fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            processed_lens=processed_lens,
            streams=decode_streams,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
        )
    elif params.decoding_method == "modified_beam_search":
        modified_beam_search(
            model=model,
            streams=decode_streams,
            encoder_out=encoder_out,
            num_active_paths=params.num_active_paths,
            banned_ids=BANNED_IDS,
            # blank_penalty=3, ##### added blank penalty
        )
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")

    states = unstack_states(new_states)

    finished_streams = []
    for i in range(len(decode_streams)):
        # decode_streams[i].states = states[i]
        # decode_streams[i].done_frames += encoder_out_lens[i]

        stream = decode_streams[i] #--------------------
        stream.states = states[i] #--------------------
        stream.done_frames += encoder_out_lens[i] #--------------------

        if stream.done:
            finished_streams.append(i)

            # ----- intent post‑processing --------------------
            mean_logits = stream.intent_logit_sum / stream.total_frames
            probs       = torch.softmax(mean_logits, dim=-1)

            top3_vals, top3_ids = probs.topk(3)          # tensors
            stream.intent_mean_logits = mean_logits.cpu()
            stream.intent_probs       = probs.cpu()
            stream.intent_top3        = list(
                zip(top3_ids.tolist(), top3_vals.tolist())
            )
            # -------------------------------------------------- #####



        # if decode_streams[i].done:
        #     finished_streams.append(i)

    return finished_streams


def decode_dataset(
    cuts: CutSet,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[Tuple[List[str], List[str]]]]:
    """Decode dataset.

    Args:
      cuts:
        Lhotse Cutset containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    device = model.device

    opts = FbankOptions()
    opts.device = device
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000
    opts.mel_opts.num_bins = 80

    log_interval = 100

    decode_results = []
    # Contain decode streams currently running.
    decode_streams = []
    for num, cut in enumerate(cuts):
        # each utterance has a DecodeStream.
        initial_states = get_init_states(model=model, batch_size=1, device=device)
        decode_stream = DecodeStream(
            params=params,
            cut_id=cut.id,
            initial_states=initial_states,
            decoding_graph=decoding_graph,
            device=device,
        )
        print("Attempting to load audio from:", cut.recording.sources[0].source)

        audio: np.ndarray = cut.load_audio()
        # audio.shape: (1, num_samples)
        assert len(audio.shape) == 2
        assert audio.shape[0] == 1, "Should be single channel"
        assert audio.dtype == np.float32, audio.dtype

        # The trained model is using normalized samples
        # - this is to avoid sending [-32k,+32k] signal in...
        # - some lhotse AudioTransform classes can make the signal
        #   be out of range [-1, 1], hence the tolerance 10
        assert (
            np.abs(audio).max() <= 10
        ), "Should be normalized to [-1, 1], 10 for tolerance..."

        samples = torch.from_numpy(audio).squeeze(0)

        fbank = Fbank(opts)
        feature = fbank(samples.to(device))
        decode_stream.set_features(feature, tail_pad_len=30)
        decode_stream.ground_truth = cut.supervisions[0].text

        decode_streams.append(decode_stream)

        while len(decode_streams) >= params.num_decode_streams:
            finished_streams = decode_one_chunk(
                params=params, model=model, decode_streams=decode_streams
            )
            # for i in sorted(finished_streams, reverse=True):
            #     decode_results.append(
            #         (
            #             decode_streams[i].id,
            #             decode_streams[i].ground_truth.split(),
            #             sp.decode(decode_streams[i].decoding_result()).split(),
            #         )
            #     )
            #     del decode_streams[i]

            for i in sorted(finished_streams, reverse=True):
                    stream = decode_streams[i]
                    hyp_text = sp.decode(stream.decoding_result()) 
                    hyp_toks = hyp_text.split() 
                    # hyp_toks = [sp.id_to_piece(i) for i in stream.token_syms]

                    # —— compute final intent from the accumulated logits ——  
                    mean_logits = stream.intent_logit_sum / stream.total_frames
                    pred_id     = mean_logits.argmax().item()
                    pred_label  = intent_id2label[pred_id]
                    # ----------------------------------------------------------------

                    # ---------- intent statistics ---------------------------------
                    # (a) confidence = prob of chosen intent (= first of top‑3)
                    pred_id  = stream.intent_top3[0][0]
                    conf     = stream.intent_top3[0][1]

                    # (b) human‑readable top‑3 list
                    top3_readable = [
                        f"{intent_id2label[idx]} ({prob*100:.1f}%)"
                        for idx, prob in stream.intent_top3
                    ]

                    # (c) token‑level relevance for this chosen intent
                    # frame_rel = torch.cat(stream.frame_rels, dim=0)[:, pred_id]    # (F,)
                    # tok_rel   = token_relevance(frame_rel, hyp_toks)               # list[(tok,p)]

                    # Get concatenated frame-level relevance for the predicted intent (shape: F,)
                    frame_rel = torch.cat(stream.frame_rels, dim=0)[:, pred_id]  # (F,)
                    # Use the new token times stored during decoding
                    token_times = stream.token_times  # list of frame indices when tokens were emitted
                    # Call new relevance calculation that uses actual token emission frames
                    tok_rel = token_relevance_with_times(frame_rel, hyp_toks, token_times)  # list of (token, relevance)

                    nice_stats = (
                        f"# Confidence: {conf*100:.1f}%\n"
                        f"# Top 3 intents: {', '.join(top3_readable)}\n"
                        f"# Top 3 relevant tokens: " +
                        ", ".join([f"'{w}' {p*100:.1f}%" for w, p in tok_rel])
                    )
                    # ----------------------------------------------------------------


                    decode_results.append(
                        (
                            stream.id,
                            stream.ground_truth.split(),
                            hyp_text,
                            pred_label,      # new field for xai
                            nice_stats,  #
                            stream.intent_history #
                        )
                    )
                    del decode_streams[i]

        if num % log_interval == 0:
            logging.info(f"Cuts processed until now is {num}.")

    # decode final chunks of last sequences
    while len(decode_streams):
        finished_streams = decode_one_chunk(
            params=params, model=model, decode_streams=decode_streams
        )
        # for i in sorted(finished_streams, reverse=True):
        #     decode_results.append(
        #         (
        #             decode_streams[i].id,
        #             decode_streams[i].ground_truth.split(),
        #             sp.decode(decode_streams[i].decoding_result()).split(),
        #         )
        #     )
        #     del decode_streams[i]

        for i in sorted(finished_streams, reverse=True):
                stream = decode_streams[i]

                hyp_text = sp.decode(stream.decoding_result()) ##### NEW
                hyp_toks = hyp_text.split() ##### NEW
                # hyp_toks = [sp.id_to_piece(i) for i in stream.token_syms]

                # compute final intent from accumulated logits ——  
                mean_logits = stream.intent_logit_sum / stream.total_frames
                pred_id     = mean_logits.argmax().item()
                pred_label  = intent_id2label[pred_id]
                # ----------------------------------------------------------------

                # ---------- intent statistics ---------------------------------
                # (a) confidence = prob of chosen intent (= first of top‑3)
                pred_id  = stream.intent_top3[0][0]
                conf     = stream.intent_top3[0][1]

                # (b) human‑readable top‑3 list
                top3_readable = [
                    f"{intent_id2label[idx]} ({prob*100:.1f}%)"
                    for idx, prob in stream.intent_top3
                ]

                # (c) token‑level relevance for this chosen intent
                # frame_rel = torch.cat(stream.frame_rels, dim=0)[:, pred_id]    # (F,)
                # tok_rel   = token_relevance(frame_rel, hyp_toks)               # list[(tok,p)]

                # Get concatenated frame-level relevance for the predicted intent (shape: F,)
                frame_rel = torch.cat(stream.frame_rels, dim=0)[:, pred_id]  # (F,)
                # Use new token times stored during decoding
                token_times = stream.token_times  # list of frame indices when tokens were emitted
                # Call new relevance calculation that uses actual token emission frames
                tok_rel = token_relevance_with_times(frame_rel, hyp_toks, token_times)  # list of (token, relevance)

                nice_stats = (
                    f"# Confidence: {conf*100:.1f}%\n"
                    f"# Top 3 intents: {', '.join(top3_readable)}\n"
                    f"# Top 3 relevant tokens: " +
                    ", ".join([f"'{w}' {p*100:.1f}%" for w, p in tok_rel])
                )
                # ----------------------------------------------------------------

                decode_results.append(
                    (
                        stream.id,
                        stream.ground_truth.split(),
                        hyp_text,
                        pred_label,      # new 4th field for xai
                        nice_stats,  #
                        stream.intent_history #
                    )
                )
                del decode_streams[i]

    if params.decoding_method == "greedy_search":
        key = "greedy_search"
    elif params.decoding_method == "fast_beam_search":
        key = (
            f"beam_{params.beam}_"
            f"max_contexts_{params.max_contexts}_"
            f"max_states_{params.max_states}"
        )
    elif params.decoding_method == "modified_beam_search":
        key = f"num_active_paths_{params.num_active_paths}"
    else:
        raise ValueError(f"Unsupported decoding method: {params.decoding_method}")
    return {key: decode_results}


def save_results(
    params: AttributeDict,
    test_set_name: str,
    # results_dict: Dict[str, List[Tuple[List[str], List[str]]]],

    # Now each entry is (utt_id, ref_tokens, hyp_tokens, intent_str)
    # results_dict: Dict[str, List[Tuple[str, List[str], List[str], str]]],
    results_dict: Dict[str, List[Tuple[str, List[str], List[str], str, str, List]]],

):
    

    out_dir = params.res_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = out_dir / f"{test_set_name}-intent-top3-{params.suffix}.csv" #CSV for xai


    # 1) Write the combined CSV
    with open(combined_csv, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow([
            "audio_id",
            "ground_truth",
            "frame_number",
            "1st_intent", "1st_intent_conf",
            "2nd_intent", "2nd_intent_conf",
            "3rd_intent", "3rd_intent_conf",
        ])

        # Loop over every utterance in every key
        for key, results in results_dict.items():
            for utt_id, ref, hyp, intent, nice_stats, intent_history in sorted(results, key=lambda x: x[0]):
                gt = " ".join(ref)   # e.g. "SEND A EMAIL <email_query>"
                for frame_idx, probs in intent_history:
                    # pick top‑3 (index, prob) pairs
                    top3 = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                    intents = [ intent_id2label[i]     for i, _ in top3 ]
                    confs   = [ f"{p:.4f}"  for _, p in top3 ]   
                    writer.writerow([
                        utt_id,
                        gt,
                        frame_idx,
                        intents[0], confs[0],
                        intents[1], confs[1],
                        intents[2], confs[2],
                    ])
            

    logging.info(f"Wrote combined intent top3 CSV to {combined_csv}")


    # test_set_wers = dict()
    test_set_wers = {}
    for key, results in results_dict.items():


        # ---------- custom recogs file with xai ----------
        rec_path = params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        rec_path.parent.mkdir(parents=True, exist_ok=True)

        with open(rec_path, "w", encoding="utf-8") as f:
            # sort by utterance id for deterministic output
            for utt_id, ref, hyp, intent, nice_stats, intent_history in sorted(results, key=lambda x: x[0]):
                # hyp list + intent tag at the end
                hyp_with_intent = hyp.split() + [intent]
                f.write(f"{utt_id}:\tref={ref}\n")
                f.write(f"{utt_id}:\thyp={hyp_with_intent}\n")

                f.write(f"{nice_stats}\n")                 #show the XAI lines

        logging.info(f"The transcripts are stored in {rec_path}")

        # recog_path = (
        #     params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        # )

        # # results = sorted(results)
        # results = sorted(results, key=lambda x: x[0])

        # # Build list of (utt_id, text) tuples for store_transcripts:
        # # we join hyp_tokens into a string, then append the intent tag at end
        # texts_for_store = []
        # for utt_id, _, hyp, intent in results:
        #     hyp_str = " ".join(hyp)
        #     # e.g. store "hypothesis-text \t <intent_tag>"
        #     texts_for_store.append((utt_id, f"{hyp_str}\t{intent}"))

        # # store_transcripts(filename=recog_path, texts=results)
        # store_transcripts(filename=recog_path, texts=texts_for_store)


        # logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        #  write_error_stats expects (ref_tokens, hyp_tokens) tuples
        # wer_input = [(ref, hyp + [intent]) for _, ref, hyp, intent in results]
        
        # wer_input = [(utt_id, ref, hyp + [intent]) for utt_id, ref, hyp, intent in results]

        wer_input = []
        for utt_id, ref, hyp, intent, nice_stats, intent_history in results:
            wer_input.append((utt_id, ref, hyp.split() + [intent]))


        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w", encoding='utf-8') as f:  ###########################
            # wer = write_error_stats(
            #     f, f"{test_set_name}-{key}", results, enable_log=True
            # )
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", wer_input, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))


    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    SlurpAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    params.res_dir = params.exp_dir / "streaming" / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    assert params.causal, params.causal
    assert "," not in params.chunk_size, "chunk_size should be one value in decoding."
    assert (
        "," not in params.left_context_frames
    ), "left_context_frames should be one value in decoding."
    params.suffix += f"-chunk-{params.chunk_size}"
    params.suffix += f"-left-context-{params.left_context_frames}"

    # for fast_beam_search
    if params.decoding_method == "fast_beam_search":
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    from streaming_beam_search import build_banned_id_set   #####
    global BANNED_IDS
    BANNED_IDS = build_banned_id_set(sp)                    #####

    # <blk> and <unk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()


    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )


    

    model.to(device)
    model.eval()
    model.device = device

    decoding_graph = None
    if params.decoding_method == "fast_beam_search":
        decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    slurp = SlurpAsrDataModule(args)

    test_clean_cuts = slurp.test_clean_cuts()
    # test_other_cuts = slurp.test_other_cuts()

    # test_sets = ["test-clean", "test-other"]
    test_sets = ["test-clean"]
    # test_cuts = [test_clean_cuts, test_other_cuts]
    test_cuts = [test_clean_cuts] ###########

    for test_set, test_cut in zip(test_sets, test_cuts):
        results_dict = decode_dataset(
            cuts=test_cut,
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
        )

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
