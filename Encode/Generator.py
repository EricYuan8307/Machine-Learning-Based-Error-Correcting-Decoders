import torch

def generator(nr_codewords, bits, device):
    codewords = torch.randint(0, 2, size=(nr_codewords, 1, bits), dtype=torch.float, device=device)

    return codewords