import torch
from torch import nn
import torch.nn.functional as F

class HardDecisionML(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            C: Use this matrix to calculate the closest hamming distance.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result.
        """
        super(HardDecisionML, self).__init__()
        self.C = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                               [1, 1, 1, 0, 0, 0, 0],
                               [1, 0, 0, 1, 1, 0, 0],
                               [0, 1, 1, 1, 1, 0, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [1, 0, 1, 1, 0, 1, 0],
                               [1, 1, 0, 0, 1, 1, 0],
                               [0, 0, 1, 0, 1, 1, 0],
                               [1, 1, 0, 1, 0, 0, 1],
                               [0, 0, 1, 1, 0, 0, 1],
                               [0, 1, 0, 0, 1, 0, 1],
                               [1, 0, 1, 0, 1, 0, 1],
                               [1, 0, 0, 0, 0, 1, 1],
                               [0, 1, 1, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1],], device=mps_device, dtype=torch.float)

    def forward(self, harddecision):
        # Calculate Hamming distances
        distances = torch.sum(harddecision == self.C, dim=2)

        # Find indices of minimum distances
        most_like = torch.argmax(distances, dim=1)

        # Information Replace
        harddecision_output = self.C[most_like].unsqueeze(1)

        return harddecision_output


class SoftDecisionML(nn.Module):
    def __init__(self, mps_device):
        """
        soft-decision Maximum Likelihood Estimation

        Args:
            C: codebook after BPSK.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result(closest euclidean distance).
        """
        super(SoftDecisionML, self).__init__()
        self.sd_c = torch.tensor([[-1, -1, -1, -1, -1, -1, -1],
                               [ 1,  1,  1, -1, -1, -1, -1],
                               [ 1, -1, -1,  1,  1, -1, -1],
                               [-1,  1,  1,  1,  1, -1, -1],
                               [-1,  1, -1,  1, -1,  1, -1],
                               [ 1, -1,  1,  1, -1,  1, -1],
                               [ 1,  1, -1, -1,  1,  1, -1],
                               [-1, -1,  1, -1,  1,  1, -1],
                               [ 1,  1, -1,  1, -1, -1,  1],
                               [-1, -1,  1,  1, -1, -1,  1],
                               [-1,  1, -1, -1,  1, -1,  1],
                               [ 1, -1,  1, -1,  1, -1,  1],
                               [ 1, -1, -1, -1, -1,  1,  1],
                               [-1,  1,  1, -1, -1,  1,  1],
                               [-1, -1, -1,  1,  1,  1,  1],
                               [ 1,  1,  1,  1,  1,  1,  1]], device=mps_device, dtype=torch.float)

    def forward(self,llr):
        # Compute the distance between each input vector and each codeword(euclidean distance)
        distances = torch.cdist(llr, self.sd_c.unsqueeze(0))

        # Calculate softmax over the negative distances
        soft_assignments = F.softmax(-distances, dim=2)
        most_like = torch.argmax(soft_assignments, dim=2)
        softdecision_output = self.sd_c[most_like]

        return softdecision_output



# device = (torch.device("mps") if torch.backends.mps.is_available()
#           else (torch.device("cuda") if torch.backends.cuda.is_available()
#                 else torch.device("cpu")))
#
# llr = torch.tensor([[[1,  1,  1, -1, -1, -1, -1]],
#                      [[1, -1, -1,  1,  1, -1, -1]]], dtype=torch.float, device=device)
#
# # llr = torch.randint(-2, 2, size=(2, 1, 7), dtype=torch.float, device=device)
# print("llr:",llr)
#
# hddecoder = HardDecisionML(device)
# hdcodebook = hddecoder(llr)
#
# sddecoder = SoftDecisionML(device)
# sdcodebook = sddecoder(llr)
#
# print("hdcodebook:",hdcodebook)
# print("sdcodebook:",sdcodebook)