import torch
import torch.nn as nn
from torch.nn import functional as F
# from math import sqrt

from .utils import get_sinusoid_encoding_table
from .Layers import FFTBlock
import transformer.Constants as Constants
from utils.tools import get_mask_from_lengths
import pdb
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEncoder(nn.Module):
    """ Encoder """

    def __init__(self, n_symbols, embedding_dim, conv_kernel,
                 drop_rate, n_blocks,attention_heads, ffn_hidden,max_len):
        super(TransformerEncoder, self).__init__()

        n_position = max_len + 1
        n_src_vocab = n_symbols
        d_word_vec = embedding_dim
        n_layers = n_blocks
        n_head = attention_heads
        d_k = d_v = (
            embedding_dim
            // attention_heads
        )
        d_model = embedding_dim
        d_inner = ffn_hidden
        kernel_size = conv_kernel
        dropout = drop_rate

        self.max_seq_len = max_len
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD
        )
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def plot(self,enc_output):
        import matplotlib.pyplot as plt
        text_1 = enc_output.cpu().detach().numpy()
        fig_sample, axes = plt.subplots(1, 1, squeeze=False)
        axes[0][0].imshow(text_1[0].T,origin="lower")
        plt.savefig("./text_infer.png")


    def forward(self, src_seq, src_lens, max_src_len, return_attns=False):

        if max_src_len is None:
            max_src_len = src_lens.item()
        mask = ~get_mask_from_lengths(src_lens, max_src_len)

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
    
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
          
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        return enc_output