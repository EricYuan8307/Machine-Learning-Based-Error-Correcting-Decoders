from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Decode.HardDecision import hard_decision # sign to binary
from Encode.Modulator import bpsk_modulator # Binary to sign
from Codebook.CodebookMatrix import ParitycheckMatrix

class InitialEmbedding(nn.Module):
    def __init__(self, encoded, d_model, parity_matrix, device):
        super(InitialEmbedding, self).__init__()
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

