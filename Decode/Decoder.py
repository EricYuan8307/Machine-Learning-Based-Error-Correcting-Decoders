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
        self.H = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    def forward(self, input):

        result = torch.matmul(input, self.H.T).to(torch.int)

        return result


class Parity10_5decoder(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            r: decoder matrix.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final 4-bit codeword.
        """

        super(Parity10_5decoder, self).__init__()
        self.H = torch.tensor([[1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 1, 0, 0, 0, 0, 1]],device=mps_device, dtype=torch.float)

    def forward(self, input):

        result = torch.matmul(input, self.H.T)

        return result.to(torch.int)