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

    noise_std = torch.sqrt(torch.tensor((1.0)/(10 ** (snr / 10)/2))).to(mps_device)

    # Calculate the LLR
    llr = 2 * signal / (noise_std**2)

    return llr

# received_signal = torch.tensor(1)
# snr_value = 20.0
#
# llr_values = llr(received_signal, snr_value)
# print(f"LLR Values: {llr_values.item()}")

#
# # received_signal = 1
# # snr_value = 10
# # llr_values = llr(received_signal, snr_value)
# # print(llr_values)