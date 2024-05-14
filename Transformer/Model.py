from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Decode.HardDecision import hard_decision # sign to binary
from Encode.Modulator import bpsk_modulator
import math


class Embedding(nn.Module):
    def __init__(self, encoded, d_model, parity_matrix, device):
        super(Embedding, self).__init__()
        self.device = device
        self.parity = parity_matrix.squeeze(0)
        self.src_embed = nn.Parameter(torch.empty((encoded + self.parity.size(0), d_model))).to(self.device)
        nn.init.xavier_uniform_(self.src_embed)

    def forward(self, noised_signal):
        magnitude = torch.abs(noised_signal)
        binary = hard_decision(torch.sign(noised_signal), self.device)
        syndrome = torch.matmul(binary, self.parity.T) % 2
        syndrome = bpsk_modulator(syndrome)
        emb = torch.cat([magnitude, syndrome], -1).unsqueeze(-1)
        emb = self.src_embed.unsqueeze(0) * emb
        return emb

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x)

class SublayerConnection(nn.Module): # Residual NN for MH-Sa and Feed Fwd output place and Norm Function
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        x0 = self.norm(x)
        x0 = sublayer(x0)
        x0 = self.dropout(x0)
        x0 = x0 + x
        return x0

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout, device):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # embedding dimension must be divisible by number of heads
        self.d_k = d_model // h
        self.h = h # number of heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, query, key, value, mask):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.tensor(-1e9, dtype=torch.float, device=self.device))
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        return x

class ECC_Transformer(nn.Module):
    def __init__(self, n_head, d_model, encoded, pc_matrix, N_dec, dropout, device):
        super(ECC_Transformer, self).__init__()
        self.d_model = d_model
        self.parity_matrix = pc_matrix
        self.encoded = encoded
        self.device = device

        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, d_model, dropout, self.device)
        ff = PositionwiseFeedForward(d_model, d_model*4, dropout)

        self.input_embed = Embedding(self.encoded, self.d_model, self.parity_matrix, self.device)
        self.encoderlayer = EncoderLayer(self.d_model, c(attn), c(ff), dropout)
        self.decoder = Decoder(self.encoderlayer, N_dec) # N_dec: encoder layers 复制次数
        self.oned_final_embed = torch.nn.Sequential(*[nn.Linear(self.d_model, 1)]) # make 32 channel to 1 channel. Convert to original channel
        self.out_fc = nn.Linear(self.encoded + self.parity_matrix.size(0), self.encoded) # Convert 10(7+3) to 7(encoded codeword)

        self.src_mask = self.get_mask(self.encoded, self.parity_matrix)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_embed(x)
        x = self.encoderlayer(x, self.src_mask)
        x = self.decoder(x, self.src_mask)
        x = self.oned_final_embed(x).squeeze(-1)
        x = self.out_fc(x)
        return x

    def get_mask(self, n, pc_matrix, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(n, pc_matrix):
            mask_size = n + pc_matrix.size(0)
            mask = torch.eye(mask_size, mask_size)
            for ii in range(pc_matrix.size(0)):
                idx = torch.where(pc_matrix[ii] > 0)[0]
                for jj in idx:
                    for kk in idx:
                        if jj != kk: # < could decrease a little complexity
                            mask[jj, kk] += 1
                            mask[kk, jj] += 1
                            mask[n + ii, jj] += 1
                            mask[jj, n + ii] += 1
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(n, pc_matrix).to(self.device)
        return src_mask