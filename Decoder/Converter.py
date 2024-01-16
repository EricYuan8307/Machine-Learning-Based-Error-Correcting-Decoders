import torch
import torch.nn as nn

# Single-label Neural Network Decoder:
class DecimaltoBinary(nn.Module):
    def __init__(self, device):

        super().__init__()
        self.device = device
        self.B = torch.tensor([[0, 0, 0, 0],
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
                               [1, 1, 1, 1]], device=self.device, dtype=torch.float) # torch.Size([1, 16, 4])


    def forward(self, decimal_tensor):
        decimal_tensor = decimal_tensor.unsqueeze(1)
        decimal_tensor = torch.argmax(decimal_tensor, dim=2)
        Binary_output = self.B[decimal_tensor]

        return Binary_output

def BinarytoDecimal(binary_tensor, device):
    decimal_values = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1], dtype=torch.float, device=device)), dim=-1)
    decimal_values = decimal_values.squeeze()

    return decimal_values


# Multi-label Neural Network Decoder:
def MLNN_decision(bitcodeword, mps_device):
    tensor_1 = torch.tensor(1, dtype=torch.float, device=mps_device)
    tensor_0 = torch.tensor(0, dtype=torch.float, device=mps_device)
    estimated_bits = torch.where(bitcodeword >= 0.5, tensor_1, tensor_0)

    return estimated_bits

