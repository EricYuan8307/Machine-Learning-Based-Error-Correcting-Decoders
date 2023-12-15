import torch

# BPSK Modulator and Add Noise After Modulator
class bpsk_modulator(torch.nn.Module):
    def __init__(self):
        """
        Use BPSK to compress the data, which is easily to transmit.

        Args:
            codeword: data received from the Hamming(7,4) encoder(Tensor)

        Returns:
            bits: Tensor contain all data modulated and add noise
        """
        super(bpsk_modulator, self).__init__()

    def forward(self, codeword, mps_device):
        bits = codeword.to(dtype=torch.float).to(mps_device)
        bits = bits.to(dtype=torch.int)
        bits = 2 * bits - 1

        return bits