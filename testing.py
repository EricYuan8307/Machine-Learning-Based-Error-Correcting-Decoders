import torch

# Parameters
n = 7  # Codeword length
k = 4  # Message length
snr_db = torch.tensor(7)  # Signal-to-noise ratio in dB
max_iterations = 10  # Maximum number of iterations for ISD
q = 1  # Quantization level

# Generate a random message
message = torch.randint(0, 2, (k,)).float()  # Convert to float
print("Original Message:", message)

# Simple parity-check matrix for (7,4) Hamming code
H = torch.tensor([
    [1, 1, 1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0, 0, 1]
], dtype=torch.float32)

# Generator matrix for (7,4) Hamming code
G = torch.tensor([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
], dtype=torch.float32)

# Encode the message
codeword = message @ G % 2
print("Encoded Codeword:", codeword)

# Add noise to the codeword
snr = 10**(snr_db / 10)
sigma = torch.sqrt(1 / (2 * snr))
noise = sigma * torch.randn(n)
received = codeword + noise
print("Received Codeword with Noise:", received)

# Quantize the received signal based on q
def quantize(signal, q):
    if q == 0:
        return torch.round(signal)  # Hard decision
    elif q == 1:
        return torch.sign(signal) * torch.clamp(torch.abs(signal), 0, 1)  # Simple soft decision
    elif q == 2:
        # For q=2, quantize to 4 levels: strong 0, weak 0, weak 1, strong 1
        levels = torch.tensor([-1.5, -0.5, 0.5, 1.5])
        quantized = torch.zeros_like(signal)
        for i, level in enumerate(levels):
            quantized[torch.abs(signal - level) == torch.min(torch.abs(signal - levels))] = level
        return quantized
    else:
        raise ValueError("Unsupported quantization level")

# ISD Decoding with quantization
def isd_decode(received, H, max_iterations, q):
    r = received.clone()
    for _ in range(max_iterations):
        syndrome = torch.matmul(H, torch.round(r)) % 2
        if torch.all(syndrome == 0):
            break
        # Compute the reliability (soft decision)
        reliability = 1 / (1 + torch.exp(-2 * r / sigma))
        error_pos = torch.argmax(reliability)
        r[error_pos] = 1 - r[error_pos]
        r = quantize(r, q)
    return torch.round(r)

decoded = isd_decode(received, H, max_iterations, q)
print("Decoded Codeword:", decoded)
