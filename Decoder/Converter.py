import torch
import torch.nn.functional as F
import torch.nn as nn

def BinarytoDecimal(binary_tensor, device):
    decimal_values = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1], dtype=torch.float, device=device)), dim=-1)
    decimal_values = decimal_values.squeeze()

    return decimal_values


class DecimaltoBinary(nn.Module):
    def __init__(self, device):

        super().__init__()
        self.device = device
        self.B = torch.tensor([[[0, 0, 0, 0],
                               [1, 1, 1, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [0, 0, 1, 1],
                               [0, 1, 0, 0],
                               [0, 1, 0, 1],
                               [0, 1, 1, 0],
                               [0, 1, 1, 1],
                               [1, 0, 0, 0],
                               [1, 0, 0, 1],
                               [1, 0, 1, 0],
                               [1, 0, 1, 1],
                               [1, 1, 0, 0],
                               [1, 1, 0, 1],
                               [1, 1, 1, 1]],], device=self.device, dtype=torch.float) # torch.Size([1, 16, 4])


    def forward(self, decimal_tensor):


        return



