import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

# BPSK Modulator and Add Noise After Modulator
class bpsk_modulator(torch.nn.Module):
    def __init__(self):
        """
        Use BPSK to compress the data, which is easily to transmit.

        Args:
            codeword: data received from the Hamming(7,4) encoder(Tensor)

        Returns:
            data: Tensor contain all data modulated and add noise
        """
        super(bpsk_modulator, self).__init__()

    def forward(self, codeword, snr_dB):
        # data = torch.tensor(data, dtype=float)
        bits = codeword.to(dtype=torch.float).to(mps_device)
        bits = 2 * bits - 1

        # Add Gaussian noise to the signal
        noise_power = torch.tensor(10 ** (snr_dB / 10)).to(mps_device)
        noise = torch.sqrt(1 / (2 * noise_power)) * torch.randn(bits.shape).to(mps_device)
        noised_signal = bits + noise

        return noised_signal