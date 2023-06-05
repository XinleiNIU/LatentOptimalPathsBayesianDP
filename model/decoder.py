import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import LinearNorm, get_sinusoid_encoding_table

from utils.tools import get_mask_from_lengths
from .Layers import FFTBlock, PostNet
import numpy as np
import pdb


class BaseDecoder(nn.Module):
    """ P(y|x,z): decode target sequence from latent variables conditioned by x
    """

    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, inputs, text_embd, z_lengths=None, text_lengths=None,
             targets=None):
        """
        :param inputs: latent representations, [batch, max_audio_time, z_hidden]
        :param text_embd: text encodings, [batch, max_text_time, T_emb_hidden]
        :param z_lengths: [batch, ]
        :param text_lengths: [batch, ]
        :param targets: [batch, max_audio_time, out_dim]
        :return: tensor1: reconstructed acoustic features, tensor2: alignments
        """
        raise NotImplementedError

    @staticmethod
    def _compute_l1_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = targets.shape[1]
            seq_mask = get_mask_from_lengths(lengths, max_time)
            l1_loss = torch.mean(
                torch.sum(
                    torch.mean(
                        torch.abs(reconstructed - targets),
                        dim=-1) * seq_mask,
                    dim=-1) / lengths.type(torch.float32))
        else:
            l1_loss = F.l1_loss(reconstructed, targets)
        return l1_loss

    @staticmethod
    def _compute_l2_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = targets.shape[1]
            seq_mask = get_mask_from_lengths(lengths, max_time)
            l2_loss = torch.mean(
                torch.sum(
                    torch.mean(
                        torch.square(reconstructed - targets),
                        dim=-1) * seq_mask,
                    dim=-1) / lengths.type(torch.float32))
        else:
            l2_loss = F.mse_loss(reconstructed, targets)
        return l2_loss


class TransformerDecoder(BaseDecoder):
    def __init__(self, nblk, attention_dim, attention_heads,
                conv_filters, conv_kernel, drop_rate,out_dim, max_len):
        super(TransformerDecoder, self).__init__()
        
        self.out_dim = out_dim
        self.mel_linear = LinearNorm(attention_dim, out_dim)
        self.postnet = PostNet()
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(max_len, attention_dim).unsqueeze(0),
            requires_grad=False,
        )
        self.out_projection = LinearNorm(attention_dim, out_dim*2)
        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    attention_dim, attention_heads, attention_dim//attention_heads, attention_dim//attention_heads, 
                    conv_filters, conv_kernel, dropout=drop_rate, name = 'decoder-attention-{}'.format(i)
                )
                for i in range(nblk)
            ]
        )

    def forward(self, inputs, text_embd, reduction_factor = 1, z_lengths=None, text_lengths=None, f0_embd = None,return_attns=True):
        """
            text_embd :[batch,N,512]
            z_lengths : mel_length
        """

        batch_size = inputs.shape[0]
        if z_lengths is not None:
            max_len = z_lengths.max()
            
        else:
            max_len = inputs.sum(-1).max()
        
        mask = ~get_mask_from_lengths(z_lengths)
        # enc_seq, extend_mel_len = self.length_regulator(text_embd, inputs, max_len)
        enc_seq = torch.matmul(inputs,text_embd)

        if f0_embd is not None:
            enc_seq = enc_seq + f0_embd
        
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) #[batch,T,T]
       
        dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        slf_attn_mask = slf_attn_mask[:, :, :max_len]
        mask = mask[:,:max_len]
        alignemnts = {}
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
           
            if return_attns:
                alignemnts[dec_layer.name] =dec_output
                
        # initial_outs = self.mel_linear(dec_output)
        initial_outs = self.out_projection(dec_output)[:, :, : reduction_factor * self.out_dim]
        initial_outs = initial_outs.reshape(batch_size, max_len * reduction_factor, self.out_dim)
        postnet_output = self.postnet(initial_outs) 

        # outputs = self.residual_projection(postnet_output)+ initial_outs
        outputs = postnet_output + initial_outs
        return initial_outs, outputs, alignemnts
