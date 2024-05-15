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

    error = 0.0

    # Calculate number of erroneous blocks
    block_errors = torch.sum(predicted != target, dim=-1)  # Count bit differences for each block
    block_errors = (block_errors > 0).sum().item()  # Count blocks with at least one bit difference

    # Calculate total number of blocks, which is the size of the first dimension
    total_blocks = predicted.size(0)

    # Calculate BLER
    bler = block_errors / total_blocks

    return bler, block_errors