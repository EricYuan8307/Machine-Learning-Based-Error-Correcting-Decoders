import torch

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

    # Calculate number of erroneous blocks
    erroneous_blocks = torch.sum(predicted != target).item()

    # Calculate total number of blocks, which is the size of the first dimension
    total_blocks = predicted.size(0)

    # Calculate BLER
    bler = erroneous_blocks / total_blocks

    return bler, erroneous_blocks

# device = torch.device("cpu")
# transmitted_bits = torch.tensor([[[1, 0, 1, 1]], [[0, 1, 0, 0,]]], dtype=torch.int)
# origin_bits = torch.tensor([[[1, 1, 0, 1]], [[0, 1, 0, 0,]]], dtype=torch.int)
#
# # ber, error = calculate_ber(transmitted_bits, origin_bits)
# bler, error_num = calculate_bler(transmitted_bits, origin_bits)
# print(f"BLER: {bler}")
# print(f"error nun: {error_num}")