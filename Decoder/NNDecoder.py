import torch
import torch.nn.functional as F
import torch.nn as nn

class SingleLableNNDecoder(nn.Module):
    def __init__(self):

        super().__init__()
