import torch

def hard_decision(bitcodeword, device):
    tensor_1 = torch.tensor(1, dtype=torch.float, device=device)
    tensor_0 = torch.tensor(0, dtype=torch.float, device=device)
    estimated_bits = torch.where(bitcodeword > 0, tensor_1, tensor_0)

    return estimated_bits