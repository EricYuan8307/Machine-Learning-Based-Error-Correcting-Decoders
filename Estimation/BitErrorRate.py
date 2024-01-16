import torch

def calculate_ber(transmitted_bits, origin_bits):
    # Ensure that both tensors have the same shape
    assert transmitted_bits.shape == origin_bits.shape, "Shapes of transmitted and received bits must be the same."

    # Calculate the bit errors
    errors = (transmitted_bits != origin_bits).sum().item()

    # Calculate the Bit Error Rate (BER)
    ber = errors / transmitted_bits.numel()

    return ber, errors

# transmitted_bits = torch.tensor([[[1, 0, 1, 1]], [[0, 1, 0, 0,]]], dtype=torch.int)
# origin_bits = torch.tensor([[[1, 1, 0, 1]], [[0, 1, 0, 0,]]], dtype=torch.int)
#
# ber, error = calculate_ber(transmitted_bits, origin_bits)