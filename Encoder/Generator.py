import torch

def generator(nr_codewords, device):
    bits = torch.randint(0, 2, size=(nr_codewords, 1, 4), dtype=torch.float, device=device)

    return bits