import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

def AWGN(signal, snr_dB):
    # Add Gaussian noise to the signal
    noise_power = torch.tensor(10 ** (snr_dB / 10)).to(mps_device)
    noise = torch.sqrt(1 / (2 * noise_power)) * torch.randn(signal.shape).to(mps_device)
    noised_signal = signal + noise

    return noised_signal