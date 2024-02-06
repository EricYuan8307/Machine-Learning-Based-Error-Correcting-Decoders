import torch

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

    # Calculate channel noise standard deviation
    ch_noise_sigma = torch.sqrt(torch.tensor(1 / (10 ** (snr / 10.0)) / 2.0))

    # Calculate the LLR
    llr = 2 * signal / (ch_noise_sigma**2)

    return llr
