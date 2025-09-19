# Copyright    2021-2023  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao)
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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
from scaling import ScaledLinear

from icefall.utils import add_sos, make_pad_mask #needs change
import torch.nn.functional as F
from typing import Dict, Union, OrderedDict, Any, Optional, List
import logging
import re
# from causal_conv1d.causal_conv1d import causal_conv1d_fn



# class FocalLoss(nn.Module):
#     def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = "mean"):
#         """
#         gamma: focusing parameter
#         alpha: tensor of shape (num_classes,) containing class weights
#         reduction: 'mean', 'sum', or 'none'
#         """
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # pass class_weights here
#         self.reduction = reduction

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         """
#         logits: (batch, num_classes) raw scores
#         targets: (batch,) class indices
#         mask: optional (batch,) boolean to ignore padded frames if using pooled logits
#         """
#         ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")  # (batch,)
#         pt = torch.exp(-ce_loss)  # probability of true class
#         focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # (batch,)

#         if mask is not None:
#             focal_loss = focal_loss * mask.float()  # ignore masked positions
#             denom = mask.float().sum().clamp(min=1.0)
#         else:
#             denom = logits.shape[0]

#         if self.reduction == "mean":
#             return focal_loss.sum() / denom
#         elif self.reduction == "sum":
#             return focal_loss.sum()
#         else:
#             return focal_loss

# class CausalConv1d(nn.Module):
#     def __init__(self, dim: int, width: int, activation: str = None):
#         super().__init__()

#         dim = 512 #128
#         kernel_size = 4 


#         # self.weight = nn.Parameter(torch.empty(dim, width))
#         self.weight = nn.Parameter(torch.empty(dim, kernel_size))
#         nn.init.xavier_uniform_(self.weight)
#         self.bias = nn.Parameter(torch.zeros(dim))
#         self.width = width
#         self.activation = activation

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, dim, seqlen)
#         return causal_conv1d_fn(
#             x,
#             self.weight,
#             self.bias,
#             activation=self.activation,
#         )

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    Pure Python reference implementation of CausalConv1d.
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution layer implemented using F.conv1d.
    This version does not require a C++ extension and is fully compatible with any
    PyTorch version.
    """
    def __init__(self, dim, width, activation=None):
        super().__init__()
        assert width >= 1, "width must be >= 1"
        self.dim = dim
        self.width = width
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(dim, width))
        self.bias = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=2**0.5)
        torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, x, initial_states=None, return_final_states=False):
        """
        x: (batch, dim, seqlen)
        """
        return causal_conv1d_ref(
            x,
            self.weight,
            self.bias,
            initial_states=initial_states,
            return_final_states=return_final_states,
            activation=self.activation,
        )






def extract_intent_from_supervision(text: str) -> str:
    text = text.strip()
    match = re.search(r'(<[^>]+>)$', text)
    if match:
        token = match.group(1)
        if not token.startswith("▁"):
          token = "▁" + token
        return token
    else:
        return None

some_mapping = {

  "▁<iot_hue_lightoff>": 0,
  "▁<cooking_recipe>": 1,
  "▁<recommendation_locations>": 2,
  "▁<iot_hue_lightdim>": 3,
  "▁<lists_createoradd>": 4,
  "▁<podcasts>": 5,
  "▁<transport_taxi>": 6,
  "▁<alarm_set>": 7,
  "▁<greet>": 8,
  "▁<recommendation_events>": 9,
  "▁<takeaway_query>": 10,
  "▁<currency>": 11,
  "▁<traffic>": 12,
  "▁<recommendation_movies>": 13,
  "▁<play_audiobook>": 14,
  "▁<takeaway_order>": 15,
  "▁<general_quirky>": 16,
  "▁<radio>": 17,
  "▁<iot_hue_lightup>": 18,
  "▁<play_game>": 19,
  "▁<sendemail>": 20,
  "▁<music_settings>": 21,
  "▁<factoid>": 22,
  "▁<ticket>": 23,
  "▁<iot_wemo_off>": 24,
  "▁<set>": 25,
  "▁<addcontact>": 26,
  "▁<play_podcasts>": 27,
  "▁<alarm_query>": 28,
  "▁<datetime_convert>": 29,
  "▁<email_querycontact>": 30,
  "▁<hue_lightup>": 31,
  "▁<calendar_query>": 32,
  "▁<play_music>": 33,
  "▁<calendar_set>": 34,
  "▁<quirky>": 35,
  "▁<hue_lightoff>": 36,
  "▁<calendar_remove>": 37,
  "▁<iot_coffee>": 38,
  "▁<unk>": 39,
  "▁<remove>": 40,
  "▁<volume_other>": 41,
  "▁<social_query>": 42,
  "▁<cooking_query>": 43,
  "▁<audio_volume_down>": 44,
  "▁<music_query>": 45,
  "▁<qa_maths>": 46,
  "▁<email_query>": 47,
  "▁<social_post>": 48,
  "▁<alarm_remove>": 49,
  "▁<lists_remove>": 50,
  "▁<iot_hue_lighton>": 51,
  "▁<qa_factoid>": 52,
  "▁<post>": 53,
  "▁<datetime_query>": 54,
  "▁<audio_volume_other>": 55,
  "▁<joke>": 56,
  "▁<transport_query>": 57,
  "▁<audio_volume_up>": 58,
  "▁<general_greet>": 59,
  "▁<general_joke>": 60,
  "▁<audio_volume_mute>": 61,
  "▁<events>": 62,
  "▁<news_query>": 63,
  "▁<email_sendemail>": 64,
  "▁<cleaning>": 65,
  "▁<settings>": 66,
  "▁<coffee>": 67,
  "▁<email_addcontact>": 68,
  "▁<game>": 69,
  "▁<qa_currency>": 70,
  "▁<transport_traffic>": 71,
  "▁<qa_definition>": 72,
  "▁<convert>": 73,
  "▁<createoradd>": 74,
  "▁<iot_wemo_on>": 75,
  "▁<lists_query>": 76,
  "▁<play_radio>": 77,
  "▁<weather_query>": 78,
  "▁<wemo_on>": 79,
  "▁<music_likeness>": 80,
  "▁<qa_stock>": 81,
  "▁<iot_cleaning>": 82,
  "▁<music>": 83,
  "▁<hue_lightdim>": 84,
  "▁<definition>": 85,
  "▁<iot_hue_lightchange>": 86,
  "▁<querycontact>": 87,
  "▁<transport_ticket>": 88,
  "▁<query>": 89,
  "▁<wemo_off>": 90,
  "▁<music_dislikeness>": 91
}

class AsrModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        # vocab_size: int = 592,
        use_transducer: bool = True,
        use_ctc: bool = False,
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

        ############################intent#####################################
        # layer focusing on intent classification
        # self.intent_classifier = nn.Sequential(
        #         nn.Linear(encoder_dim, 92),
        #     )
        
        # self.intent_classifier = nn.Sequential(
        #         nn.Linear(encoder_dim, 128), # 256/512
        #         nn.ReLU(), 
        #         #add conv1d (causal) (kernel 4/8/../20),
        #         # relu
        #         nn.Linear(128, 92)
        #         # more linear layer
        #     )

        self.intent_classifier = nn.Sequential(
                nn.Linear(encoder_dim, 512),
                nn.ReLU(),
                Permute(0, 2, 1),  # (batch, time, dim) -> (batch, dim, time)
                CausalConv1d(dim=512, width=4, activation="silu"),
                Permute(0, 2, 1),  # back to (batch, time, dim)
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.Linear(128, 92)
            )

        ############################intent#####################################

        # if class_weights is not None:
        #     self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        # else:
        #     self.focal_loss = None


    @torch.no_grad()
    def infer_intent_logits(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor
    ) -> torch.Tensor:
        """
        Encoder-only inference: returns mean-pooled intent logits (N, 92).
        """
        # 1) embed  encode
        x_emb, x_lens_emb = self.encoder_embed(x, x_lens)
        mask = make_pad_mask(x_lens_emb)
        x_enc = x_emb.permute(1, 0, 2)           # (T, N, C)
        enc_out, enc_lens = self.encoder(x_enc, x_lens_emb, mask)
        enc_out = enc_out.permute(1, 0, 2)       # (N, T, C)

        # 2) framewise logits
        logits = self.intent_classifier(enc_out)  # (N, T, 92)

        # 3) mask & zero-out padding
        T = logits.size(1)
        valid = torch.arange(T, device=enc_out.device).unsqueeze(0) < enc_lens.unsqueeze(1)
        logits[~valid] = 0.0

        # 4) mean-pool
        summed = logits.sum(dim=1)                # (N, 92)
        mean_pooled = summed / enc_lens.unsqueeze(1)  # (N, 92)
        return mean_pooled
    
    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, supervision_texts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)

        ############################intent#####################################
        # map the tokens to indices some_mapping
        # add masking
        # masking intent_logits*mask
        # sum/sum of masks


        # intent_tokens = [extract_intent_from_supervision(text) for text in supervision_texts]
        intent_tokens = [extract_intent_from_supervision(text) or "▁<unk>" for text in supervision_texts]

        intent_indices = [some_mapping[token] for token in intent_tokens]
        intent_labels = torch.tensor(intent_indices, device= encoder_out.device) #(N,)


        intent_logits = self.intent_classifier(encoder_out) # (N, T, 92)

        logging.info("encoder_out.shape: %s", encoder_out.shape)         # should be [496, 26, 92]
        logging.info("encoder_out_lens.shape: %s", encoder_out_lens.shape)   # should be [496]
        logging.info("intent_logits.shape: %s", intent_logits.shape)
        logging.info("x_lens.shape: %s",x_lens.shape)

        # Create mask to ignore padded frames
        max_len = intent_logits.size(1)
        mask = torch.arange(max_len, device= encoder_out.device).unsqueeze(0) < x_lens.unsqueeze(1) # Now shape: (N, T)
        intent_logits[~mask] = 0  # Set padded logits to zero
        
        # Compute mean logits over non-padded frames
        masked_intent_logits = intent_logits.sum(dim=1)  # Sum over T -> (N, 92)
        mean_intent_logits = masked_intent_logits / x_lens[:, None]  # Normalize by valid frames (N, 92)


        intent_loss = F.nll_loss(F.log_softmax(mean_intent_logits, dim=1), intent_labels) #cross entropy

        
        
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
            

        return encoder_out, encoder_out_lens, intent_loss ######

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch.amp.autocast("cuda",enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch.amp.autocast("cuda",enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        supervision_texts: Optional[List[str]] = None,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        Returns:
          Return the transducer losses and CTC loss,
          in form of (simple_loss, pruned_loss, ctc_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        # print(f"x shape: {x.shape}")        
        # # Should be 
        # # [Batch size (number of sequences/utterances), 
        # # Number of time frames (sequence length for each utterance after padding), 
        # # Feature dimension (number of features per frame, such as MFCCs or Mel-spectrogram features)]
        # print(f"x_lens shape: {x_lens.shape}")  
        # # Should be 1-dimensional tensor of shape (N,), where each element contains the number of valid frames in the corresponding utterance in x
        # print(f"Batch size (dim0): {y.dim0}")  # Number of sequences (utterances)
        # print(f"Total number of labels (tot_size(1)): {y.tot_size(1)}")  # Total labels
        # print(f"Row splits: {y.shape.row_splits(1)}")  # Row splits for each utterance

        # # To get the length of each sequence (number of labels per sequence)
        # row_splits = y.shape.row_splits(1)
        # sequence_lengths = row_splits[1:] - row_splits[:-1]
        # print(f"Lengths of sequences: {sequence_lengths}")

        # y is a ragged tensor (from the k2 library) with two axes. Its shape is [utt][label], where:
        # utt: The batch axis (number of utterances).
        # label: The sequence of target labels for each utterance (could be phonemes, characters, or subword units).
        # This means y can hold target sequences of variable lengths (e.g., the number of words or characters in each utterance might vary).
        print(f"#################################################################")  

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        # Compute encoder outputs
        encoder_out, encoder_out_lens, intent_loss = self.forward_encoder(x, x_lens, supervision_texts)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(x.device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, intent_loss
