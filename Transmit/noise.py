import torch

def AWGN(signal, snr_dB, mps_device):
    # Add Gaussian noise to the signal
    noise_power = torch.tensor(10 ** (snr_dB / 10), device=mps_device)
    noise = torch.sqrt(1 / (2 * noise_power)) * torch.randn(signal.shape, device=mps_device)
    noised_signal = signal + noise

    return noised_signal