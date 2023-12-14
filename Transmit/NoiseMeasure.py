import torch

def NoiseMeasure(noised_signal, beforenoise_signal):

    # Noise Measurment
    ch_noise = noised_signal - beforenoise_signal
    # Calculate noise power
    noise_power = torch.mean(ch_noise ** 2)

    # Calculate practical SNR
    practical_snr = 10 * torch.log10(1 / (noise_power * 2.0))

    return practical_snr