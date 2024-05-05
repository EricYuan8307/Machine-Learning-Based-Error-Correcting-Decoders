from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Decode.HardDecision import hard_decision # sign to binary
from Encode.Modulator import bpsk_modulator # Binary to sign
from Codebook.CodebookMatrix import ParitycheckMatrix

class Embedding(nn.Module):
    def __init__(self, encoded, d_model, parity_matrix, device):
        super(Embedding, self).__init__()
        self.device = device
        self.parity = parity_matrix.squeeze(0)
        self.src_embed = nn.Parameter(torch.empty((encoded + self.parity.size(0), d_model)))

    def forward(self, noised_signal):
        magnitude = torch.abs(noised_signal)
        binary = hard_decision(torch.sign(noised_signal), self.device)
        syndrome = torch.matmul(binary, self.parity.T) % 2
        syndrome = bpsk_modulator(syndrome)
        emb = torch.cat([magnitude, syndrome], 2)
        return torch.matmul(emb, self.src_embed)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class decoder(nn.Module):
    def __init__(self, layer, N):
            super(decoder, self).__init__()
            self.layers = clones(layer, N)
            self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
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

