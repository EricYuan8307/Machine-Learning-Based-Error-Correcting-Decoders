import torch
from torch import nn

class Hamming74decoder(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            C: Use this matrix to calculate the closest hamming distance.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result.
        """

        super(Hamming74decoder, self).__init__()
        self.r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    def forward(self, input):

        result = torch.matmul(input.to(torch.float), self.r.T).to(torch.int)
        result = result.unsqueeze(1)

        return result


# device = (torch.device("mps") if torch.backends.mps.is_available()
#           else (torch.device("cuda") if torch.backends.cuda.is_available()
#                 else torch.device("cpu")))
#
# # llr = torch.tensor([[[1, -1, 0.5, -2, 5, 6, 1.2]],
# #                      [[-0.1, 1, -0.2, 0.5, -5, 6, -11]]], dtype=torch.float, device=device)
#
# llr = torch.randint(-2, 2, size=(2, 1, 7), dtype=torch.float, device=device)
# print("llr:",llr)
#
# decoder = Hamming74decoder(device)
# bitresult = decoder(llr)
#
# print("sdcodebook:",bitresult)
# print("sddecoder:",bitresult.shape)