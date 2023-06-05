import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import TransformerEncoder
from .posterior import TransformerPosterior
from .decoder import TransformerDecoder
from .prior import TransformerPrior
from .length_predictor import DenseLengthPredictor

from utils.tools import get_mask_from_lengths
from text.symbols import symbols

# from .BayesianDTW import BayesianDTW
from .BayesianPDA import BayesianPDA


import numpy as np
import pdb

class BDPVAE(nn.Module):
    """ BDPVAE-TTS """

    def __init__(self, preprocess_config, model_config):
        super(BDPVAE, self).__init__()
        self.model_config = model_config
        self.n_sample = model_config["common"]["num_samples"]
        self.mel_text_len_ratio = model_config["common"]["mel_text_len_ratio"]
        if preprocess_config["dataset"] == "popcs":
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "phone_set.json"
                ),
                "r",
            ) as f:
                n_symbols = len(json.load(f)) + 1
        else:
            n_symbols = len(symbols) + 1

        self.text_encoder = TransformerEncoder(
            n_symbols=n_symbols,
            embedding_dim=model_config["transformer"]["encoder"]["embd_dim"],
            conv_kernel=model_config["transformer"]["encoder"]["conv_kernel"],
            drop_rate=model_config["transformer"]["encoder"]["drop_rate"],
            n_blocks=model_config["transformer"]["encoder"]["n_blk"],
            attention_heads=model_config["transformer"]["encoder"]["attention_heads"],
            ffn_hidden=model_config["transformer"]["encoder"]["ffn_hidden"],
            max_len = model_config["max_seq_len"] )
        self.decoder = TransformerDecoder(
            nblk=model_config["transformer"]["decoder"]["nblk"],
            attention_dim=model_config["transformer"]["decoder"]["attention_dim"],
            attention_heads=model_config["transformer"]["decoder"]["attention_heads"],
            conv_filters=model_config["transformer"]["decoder"]["conv_filters"],
            conv_kernel=model_config["transformer"]["decoder"]["conv_kernel"],
            drop_rate=model_config["transformer"]["decoder"]["drop_rate"],
            out_dim=model_config["common"]["output_dim"],
            max_len = model_config["max_seq_len"])
        self.length_predictor = DenseLengthPredictor(
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            activation=self._get_activation(model_config["length_predictor"]["dense"]["activation"]))
        self.posterior = TransformerPosterior(
            num_mels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            post_hidden=model_config["transformer"]["posterior"]["post_hidden"],
            pos_drop_rate=model_config["transformer"]["posterior"]["pos_drop_rate"],
            n_conv =model_config["transformer"]["posterior"]["n_conv"],
            conv_kernel =model_config["transformer"]["posterior"]["conv_kernel"],
            alpha = model_config["common"]["alpha"])
        self.prior = TransformerPrior(
            n_blk=model_config["transformer"]["prior"]["n_blk"],
            channels=model_config["common"]["latent_dim"],
            n_transformer_blk=model_config["transformer"]["prior"]["n_transformer_blk"],
            embd_dim=model_config["transformer"]["encoder"]["embd_dim"],
            attention_dim=model_config["transformer"]["prior"]["attention_dim"],
            attention_heads=model_config["transformer"]["prior"]["attention_heads"],
            temperature=model_config["transformer"]["prior"]["temperature"],
            ffn_hidden=model_config["transformer"]["prior"]["ffn_hidden"],
            alpha = model_config["common"]["alpha"]
        )

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        self.pitch_embedding = None
        if model_config["add_pitch"]== True:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "stats.json"
                ),
                "r",
            ) as f: 
                f0_stats = json.load(f)
            pitch_min,pitch_max,self.mean,self.std = f0_stats['pitch']
            
            self.pitch_embedding = nn.Embedding(
                n_bins, model_config["transformer"]["encoder"]["embd_dim"],
            )
            if pitch_quantization == "log":
                self.pitch_bins = nn.Parameter(
                    torch.exp(
                        torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                    ),
                    requires_grad=False,
                )
            else:
                self.pitch_bins = nn.Parameter(
                    torch.linspace(pitch_min, pitch_max, n_bins - 1),
                    requires_grad=False,
                )

        self.dtw = BayesianPDA(alpha = model_config["common"]["alpha"])
        
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder"]["embd_dim"],
            )


    @staticmethod
    def _get_activation(activation):
        if activation == "relu":
            return nn.ReLU()
        return None

    def _compute_l2_loss(self, reconstructed, targets, lengths=None, reduce=False):
        max_time = reconstructed.shape[1]
        dim = reconstructed.shape[2]
   
        r = reconstructed.view(-1, self.n_sample, max_time, dim)
        t = targets.view(-1, self.n_sample, max_time, dim)

        if lengths is not None:
            seq_mask = get_mask_from_lengths(lengths, max_time)
            seq_mask = seq_mask.view(-1, self.n_sample, max_time)
            reshaped_lens = lengths.view(-1, self.n_sample)
            l2_loss = torch.mean(
                torch.sum(
                    torch.mean(torch.square(r - t), dim=-1) * seq_mask,
                    dim=-1) / reshaped_lens.type(torch.float32),
                dim=-1)
        else:
            l2_loss = torch.mean(torch.square(r - t), dim=[1, 2, 3])
        if reduce:
            return torch.mean(l2_loss)
        else:
            return l2_loss

    def _compute_l1_loss(self, reconstructed, targets, lengths=None, reduce=False):
        max_time = reconstructed.shape[1]
        dim = reconstructed.shape[2]
   
        r = reconstructed.view(-1, self.n_sample, max_time, dim)
        t = targets.view(-1, self.n_sample, max_time, dim)
        if lengths is not None:
            seq_mask = get_mask_from_lengths(lengths, max_time)
            seq_mask = seq_mask.view(-1, self.n_sample, max_time)
            reshaped_lens = lengths.view(-1, self.n_sample)
            l2_loss = torch.mean(
                torch.sum(
                    torch.mean(torch.square(r - t), dim=-1) * seq_mask,
                    dim=-1) / reshaped_lens.type(torch.float32),
                dim=-1)
        else:
            l2_loss = torch.mean(torch.square(r - t), dim=[1, 2, 3])
        if reduce:
            return torch.mean(l2_loss)
        else:
            return l2_loss

    @staticmethod
    def _kl_divergence(p, q, reduce=False):
        kl = torch.mean((p - q), dim=1) # (qlogp - qlogq)
        if reduce:
            return torch.mean(kl)
        else:
            return kl


    @staticmethod
    def _length_l2_loss(predicted_lengths, target_lengths, reduce=False):
        log_tgt_lengths = torch.log(target_lengths.type(torch.float32))
        log_pre_lengths = torch.log(predicted_lengths)
        if reduce:
            return torch.mean(torch.square(log_pre_lengths - log_tgt_lengths))
        else:
            return torch.square(log_pre_lengths - log_tgt_lengths)

    @staticmethod
    def _reinforce_loss(loss,logprob,reduce = False):
        reinforce_loss = loss.detach()*logprob
        if reduce:
            return torch.mean(reinforce_loss)
        else:
            return reinforce_loss


    def get_pitch_embedding(self, target, control=1.0):
        embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        return embedding

    def forward(
            self,
            speakers,
            inputs,
            text_lengths,
            max_src_len,
            f0 = None,
            # samples, 
            # spec_embs,
            mel_targets=None,
            mel_lengths=None,
            max_mel_len=None,
            reduce_loss=False,
            reduction_factor = 1,
    ):
        """
        :param speakers: speaker inputs, [batch, ]
        :param inputs: text inputs, [batch, text_max_time]
        :param text_lengths: [batch, ]
        :param max_src_len: int
        :param mel_targets: [batch, mel_max_time, mel_dim]
        :param mel_lengths: [batch, ]
        :param max_mel_len: int
        :param reduce_loss: bool
        :return: predicted mel: [batch, mel_max_time, mel_dim],
                 loss: float32,
                 samples: [batch, mel_max_time, text_max_time],
                 mask: [batch, mel_max_time, text_max_time],
                 W: [batch, mel_max_time, text_max_time],
        """
     
        # shape info
        batch_size = mel_targets.shape[0]
        mel_max_len = mel_targets.shape[1]
        text_max_len = inputs.shape[1]

        # text encoding
        text_embd = self.text_encoder(
            inputs, text_lengths, max_src_len)
	
	# Apply reduce factor if its necessary
        reduced_mels = mel_targets[:, ::reduction_factor, :]
        reduced_mel_lens = torch.div((mel_lengths + reduction_factor - 1), reduction_factor, rounding_mode='trunc')
        reduced_mel_max_len = reduced_mels.shape[1]


        # Multi speaker embedding 
        if self.speaker_emb is not None:
            text_embd = text_embd + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)

        # F0 embedding
        if self.pitch_embedding is not None:
            f0_= self.get_pitch_embedding(f0)
            f0_embd = f0_[:,::reduction_factor]

         
        # Length predictor
        predicted_lengths = self.length_predictor(
            text_embd.detach(), text_lengths)

        length_loss = self._length_l2_loss(
            predicted_lengths, mel_lengths, reduce=reduce_loss)


        # Make mask
        mask = self.dtw.W_mask(reduced_mels.shape[0],text_lengths,reduced_mel_lens)

        # Posterior encoder 
        samples, log_probs, mu, pi, W, spec_embs = self.posterior.sample(reduced_mels, 
                                                    text_embd.detach(),
                                                    text_lengths,
                                                    reduced_mel_lens,
                                                    mask, 
                                                    None)

        
        # [batch*n_sample, text_max_len, dim]
        batched_text_embd = torch.tile(
            text_embd.unsqueeze(1),
            [1, self.n_sample, 1, 1]).view(batch_size * self.n_sample, text_max_len, -1)
        batched_mel_targets = torch.tile(
            mel_targets.unsqueeze(1),
            [1, self.n_sample, 1, 1]).view(batch_size * self.n_sample, mel_max_len, -1)
        # [batch*n_sample, ]
        batched_mel_lengths = torch.tile(
            mel_lengths.unsqueeze(1),
            [1, self.n_sample]).view(-1)
        # [batch*n_sample, ]
        batched_text_lengths = torch.tile(
            text_lengths.unsqueeze(1),
            [1, self.n_sample]).view(-1)

        batched_r_mel_lengths = torch.tile(
            reduced_mel_lens.unsqueeze(1),
            [1, self.n_sample]).view(-1)

        prior_logprobs = self.prior.log_probability(z=spec_embs.detach(),
                                                    condition_inputs=batched_text_embd,
                                                    z_lengths=batched_r_mel_lengths,
                                                    condition_lengths=batched_text_lengths,
                                                    mask = mask,
                                                    omega = None)


        decoded_initial, decoded_outs, dec_att= self.decoder(samples,
                                                    batched_text_embd,
                                                    reduction_factor,
                                                    batched_r_mel_lengths,
                                                    batched_text_lengths)
        decoded_initial = decoded_initial[:, :mel_max_len, :]
        decoded_outs = decoded_outs[:, :mel_max_len, :]

        initial_l2_loss = self._compute_l2_loss(decoded_initial, batched_mel_targets,
                                                batched_mel_lengths, reduce_loss)
        l2_loss = self._compute_l2_loss(decoded_outs, batched_mel_targets,
                                        batched_mel_lengths, reduce_loss)
        l2_loss += initial_l2_loss

        reinforce_loss = log_probs

        kl_divergence = - torch.mean(prior_logprobs,-1)

        return (decoded_outs, l2_loss, torch.mean(kl_divergence), torch.mean(length_loss),reinforce_loss, samples,
                mask,text_embd,W)

    def encode_text(
            self,
            speakers,
            inputs,
            text_lengths,
            max_src_len,
            mel_targets=None,
            mel_lengths=None,
            max_mel_len=None,
            reduce_loss=False,
    ):
        # text encoding
        text_embd = self.text_encoder(
            inputs, text_lengths, max_src_len)
        return text_embd
        
    def inference(self, inputs, text_lengths, speakers = None, f0 = None, predicted_mel_lengths = None):
        
        text_embd = self.text_encoder(inputs, text_lengths,text_lengths.item())

        max_src_len = max(text_lengths)

        if self.speaker_emb is not None:
            text_embd = text_embd + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
        if self.pitch_embedding is not None:
            f0_embd = self.get_pitch_embedding(f0)

        if predicted_mel_lengths is None:
            predicted_mel_lengths = (self.length_predictor(text_embd, text_lengths)).long()
        else:
            predicted_mel_lengths = torch.tensor([predicted_mel_lengths],device = inputs.device)

        prior_latents, prior_logprobs = self.prior(predicted_mel_lengths, text_embd, text_lengths) #,f0_embd = f0_embd)

        _, predicted_mel,_ = self.decoder(
            inputs=prior_latents, text_embd=text_embd, reduction_factor = 1, z_lengths=predicted_mel_lengths,
            text_lengths=text_lengths) # f0_embd = f0_embd)

        return predicted_mel, predicted_mel_lengths, prior_latents, prior_logprobs

    def init(self, speakers, text_inputs, mel_lengths, text_lengths=None, f0=None, reduction_factor=1):
        
        max_src_len = max(text_lengths)
        text_embd = self.text_encoder(text_inputs, text_lengths, max_src_len)
        reduced_mel_lens = (mel_lengths + reduction_factor- 1) // reduction_factor
        
        if self.speaker_emb is not None:
            text_embd = text_embd + self.speaker_emb(speakers).unsqueeze(1).expand(-1, max_src_len, -1)
        # F0 embedding
        if self.pitch_embedding is not None:
            f0_embd = self.get_pitch_embedding(f0)
            f0_embd = f0_embd[:,::reduction_factor]
            # f0_embd = self.pitch_embedding(f0)

        prior_latents, prior_logprobs = self.prior.init(inputs=text_embd,
                                                        targets_lengths=reduced_mel_lens,
                                                        condition_lengths=text_lengths)
        _, predicted_mel,_ = self.decoder(inputs=prior_latents,
                                           text_embd=text_embd,
                                           reduction_factor = reduction_factor,
                                           z_lengths=reduced_mel_lens,
                                           text_lengths=text_lengths)

        return predicted_mel,text_embd

