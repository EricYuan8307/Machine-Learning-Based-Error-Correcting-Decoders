def calculate_ber(transmitted_bits, origin_bits):
    # Ensure that both tensors have the same shape
    assert transmitted_bits.shape == origin_bits.shape, "Shapes of transmitted and received bits must be the same."

    # Calculate the bit errors
    errors = (transmitted_bits != origin_bits).sum().item()

    # Calculate the Bit Error Rate (BER)
    ber = errors / (transmitted_bits.numel()*origin_bits.shape[2])

    return ber, errors