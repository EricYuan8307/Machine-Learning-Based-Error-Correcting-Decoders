import torch
import numpy as np

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

# LLR Calculaton:
def llr(signal, snr):
    """
    Calculate Log Likelihood Ratio (LLR) for a simple binary symmetric channel.

    Args:
        signal (torch.Tensor): Received signal from BPSK.
        noise_std (float): Standard deviation of the noise.

    Returns:
        llr: Log Likelihood Ratio (LLR) values.
    """

    # BPSK modulation
    # Calculate channel noise standard deviation
    ch_noise_sigma = torch.sqrt(torch.tensor(1 / (10 ** (snr / 10.0)) / 2.0))
    # ch_noise_sigma = torch.tensor(10 ** (snr / 10.0))


    # Calculate LLR
    # llr = signal * 2.0 * (ch_noise_sigma).to(mps_device)

    # noise_std = torch.sqrt(torch.tensor((1.0)/(10 ** (snr / 10)/2)))

    # Calculate the LLR
    llr = 2 * signal / (ch_noise_sigma**2)

    return llr

# received_signal = torch.tensor(1)
# snr_value = 10.0
#
# llr_values = llr(received_signal, snr_value)
# print(f"LLR Values: {llr_values.item()}")


# received_signal = 1
# snr_value = 10
# llr_values = llr(received_signal, snr_value)
# print(llr_values)