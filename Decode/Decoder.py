import torch

class PCC_decoder(torch.nn.Module):
    def __init__(self, decoderMatrix):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            H: decoder matrix.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final 12-bit codeword.
        """

        super(PCC_decoder, self).__init__()
        self.H = decoderMatrix

    def forward(self, input):

        result = torch.matmul(input, self.H.T)

        return result