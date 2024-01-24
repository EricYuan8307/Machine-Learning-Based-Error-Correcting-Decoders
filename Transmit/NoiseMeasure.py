import torch

def NoiseMeasure(noised_signal, modulated_signal):

    # Noise Measurment
    ch_noise = noised_signal - modulated_signal
    # Calculate noise power
    noise_power = torch.mean(ch_noise ** 2)

    # Calculate practical SNR
    practical_snr = 10 * torch.log10(1 / (noise_power * 2.0)) + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float))

    return practical_snr

def NoiseMeasure_BPSK(noised_signal, modulated_signal):

    # Noise Measurment
    ch_noise = noised_signal - modulated_signal
    # Calculate noise power
    noise_power = torch.mean(ch_noise ** 2)

    # Calculate practical SNR
    practical_snr = 10 * torch.log10(1 / (noise_power * 2.0))

    return practical_snr