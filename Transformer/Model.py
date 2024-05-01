from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Decode.HardDecision import hard_decision


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class decoder(nn.Module):
