import torch
import torch.nn as nn


# Multi-label Neural Network Decoder:
def MLNN_decision(bitcodeword, mps_device):
    tensor_1 = torch.tensor(1, dtype=torch.float, device=mps_device)
    tensor_0 = torch.tensor(0, dtype=torch.float, device=mps_device)
    estimated_bits = torch.where(bitcodeword >= 0.5, tensor_1, tensor_0)

    return estimated_bits


# Single-label Neural Network Decoder:
class DecimaltoBinary(nn.Module):
    def __init__(self, codebook):

        super().__init__()
        self.B = codebook


    def forward(self, decimal_tensor):
        decimal_tensor = decimal_tensor
        Binary_output = self.B[decimal_tensor]

        return Binary_output


def BinarytoDecimal(binary_tensor):
    binary_tensor = binary_tensor.squeeze()  # Remove extra dimensions
    powers_of_two = 2 ** torch.arange(binary_tensor.size(-1) - 1, -1, -1, dtype=torch.float, device=binary_tensor.device)
    decimal_values = torch.sum(binary_tensor * powers_of_two, dim=-1)
    return decimal_values