import torch

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

    # Assuming Binary Phase Shift Keying (BPSK) modulation
    noise_std = torch.sqrt(torch.tensor(10 ** (snr / 10))).to(mps_device)

    # Calculate the LLR
    llr = 2 * signal * noise_std

    # return llr_values, llr
    return llr