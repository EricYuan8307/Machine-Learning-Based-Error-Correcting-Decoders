import torch

class Hamming74decoder(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            r: decoder matrix.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final 4-bit codeword.
        """

        super(Hamming74decoder, self).__init__()
        self.r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    def forward(self, input):

        result = torch.matmul(input.to(torch.float), self.r.T).to(torch.int)

        return result