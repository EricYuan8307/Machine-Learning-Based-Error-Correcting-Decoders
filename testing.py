import torch

def generator(nr_codewords, bits, device):
    codewords = torch.randint(0, 2, size=(nr_codewords, 1, bits), dtype=torch.float, device=device)

    return codewords

def generator_ECCT(nr_codewords, bits, device):
    codewords = torch.randint(0, 2, size=(nr_codewords, bits), dtype=torch.float, device=device)

    return codewords

def calculate_ber(transmitted_bits, origin_bits):
    """
       Calculate the Block Error Rate (BLER).

       Args:
       predicted (torch.Tensor): The predicted tensor.
       target (torch.Tensor): The target tensor.

       Returns:
       float: The BER.
       int: errors
       """
    # Ensure that both tensors have the same shape
    assert transmitted_bits.shape == origin_bits.shape, "Shapes of transmitted and received bits must be the same."

    # Calculate the bit errors
    errors = (transmitted_bits != origin_bits).sum().item()

    # Calculate the Bit Error Rate (BER)
    ber = errors / transmitted_bits.numel()

    return ber, errors

def calculate_bler(predicted, target):
    """
    Calculate the Block Error Rate (BLER).

    Args:
    predicted (torch.Tensor): The predicted tensor.
    target (torch.Tensor): The target tensor.

    Returns:
    float: The BLER.
    """
    # Check if predicted and target tensors have the same shape
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target tensors must have the same shape")

    error = 0.0

    # Calculate number of erroneous blocks
    block_errors = torch.sum(predicted != target, dim=-2)  # Count bit differences for each block
    block_errors = (block_errors > 0).sum().item()  # Count blocks with at least one bit difference

    # Calculate total number of blocks, which is the size of the first dimension
    total_blocks = predicted.size(0)

    # Calculate BLER
    bler = block_errors / total_blocks

    return bler, block_errors

device = "cpu"
nr_codewords = 10
bits = 16
codewords2 = generator_ECCT(nr_codewords, bits, device)
codewords2_2 = generator_ECCT(nr_codewords, bits, device)
ber, errors = calculate_ber(codewords2, codewords2_2)
# print("codewords2 ber",ber, errors)
# bler, errors = calculate_bler(codewords2, codewords2_2)
# print("codewords2 bler",bler, errors)
print(codewords2_2.shape)
print(codewords2_2.shape[-1])
codewords1 = generator(nr_codewords, bits, device)
codewords2 = generator(nr_codewords, bits, device)
print(codewords2.shape)
print(codewords2.shape[-1])
# ber, errors = calculate_ber(codewords1, codewords2)
# print("codewords1 ber",ber, errors)
# bler, errors = calculate_bler(codewords1, codewords2)
# print("codewords1 bler",bler, errors)
