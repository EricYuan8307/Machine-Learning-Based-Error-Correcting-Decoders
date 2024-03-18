import torch

def AWGN(signal, snr_dB, device):
    # Add Gaussian noise to the signal
    noise_power = 10 ** (snr_dB / 10)
    noise = torch.sqrt(1 / (2 * noise_power)) * torch.randn(signal.shape, device=device)
    noised_signal = signal + noise

    return noised_signal