import torch
from torch import nn
import torch.nn.functional as F

class HardDecisionML74(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            C: Use this matrix to calculate the closest hamming distance.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result.
        """
        super(HardDecisionML74, self).__init__()
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

class SoftDecisionML74(nn.Module):
    def __init__(self, mps_device):
        """
        soft-decision Maximum Likelihood Estimation

        Args:
            C: codebook after BPSK.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result(closest Euclidean distance).
        """
        super(SoftDecisionML74, self).__init__()
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

    def forward(self,signal):
        # Compute the distance between each input vector and each codeword(euclidean distance)
        distances = torch.cdist(signal, self.sd_c.unsqueeze(0))

        # Calculate softmax over the negative distances
        soft_assignments = F.softmax(-distances, dim=2)
        most_like = torch.argmax(soft_assignments, dim=2)
        softdecision_output = self.sd_c[most_like]

        return softdecision_output

class SoftDecisionML10_5(nn.Module):
    def __init__(self, mps_device):
        """
        soft-decision Maximum Likelihood Estimation

        Args:
            C: codebook after BPSK.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result(closest Euclidean distance).
        """
        super(SoftDecisionML10_5, self).__init__()
        self.sd_c = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1,  1,  1, -1, -1, -1,  1],
                                  [-1, -1, -1,  1, -1, -1, -1, -1,  1,  1],
                                  [-1, -1, -1,  1,  1,  1, -1, -1,  1, -1],
                                  [-1, -1,  1, -1, -1, -1, -1,  1,  1, -1],
                                  [-1, -1,  1, -1,  1,  1, -1,  1,  1,  1],
                                  [-1, -1,  1,  1, -1, -1, -1,  1, -1,  1],
                                  [-1, -1,  1,  1,  1,  1, -1,  1, -1, -1],
                                  [-1,  1, -1, -1, -1, -1,  1,  1, -1, -1],
                                  [-1,  1, -1, -1,  1,  1,  1,  1, -1,  1],
                                  [-1,  1, -1,  1, -1, -1,  1,  1,  1,  1],
                                  [-1,  1, -1,  1,  1,  1,  1,  1,  1, -1],
                                  [-1,  1,  1, -1, -1, -1,  1, -1,  1, -1],
                                  [-1,  1,  1, -1,  1,  1,  1, -1,  1,  1],
                                  [-1,  1,  1,  1, -1, -1,  1, -1, -1,  1],
                                  [-1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
                                  [ 1, -1, -1, -1, -1,  1,  1, -1, -1, -1],
                                  [ 1, -1, -1, -1,  1, -1,  1, -1, -1,  1],
                                  [ 1, -1, -1,  1, -1,  1,  1, -1,  1,  1],
                                  [ 1, -1, -1,  1,  1, -1,  1, -1,  1, -1],
                                  [ 1, -1,  1, -1, -1,  1,  1,  1,  1, -1],
                                  [ 1, -1,  1, -1,  1, -1,  1,  1,  1,  1],
                                  [ 1, -1,  1,  1, -1,  1,  1,  1, -1,  1],
                                  [ 1, -1,  1,  1,  1, -1,  1,  1, -1, -1],
                                  [ 1,  1, -1, -1, -1,  1, -1,  1, -1, -1],
                                  [ 1,  1, -1, -1,  1, -1, -1,  1, -1,  1],
                                  [ 1,  1, -1,  1, -1,  1, -1,  1,  1,  1],
                                  [ 1,  1, -1,  1,  1, -1, -1,  1,  1, -1],
                                  [ 1,  1,  1, -1, -1,  1, -1, -1,  1, -1],
                                  [ 1,  1,  1, -1,  1, -1, -1, -1,  1,  1],
                                  [ 1,  1,  1,  1, -1,  1, -1, -1, -1,  1],
                                  [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],], device=mps_device, dtype=torch.float)

    def forward(self,signal):
        # Compute the distance between each input vector and each codeword(euclidean distance)
        distances = torch.cdist(signal, self.sd_c.unsqueeze(0))

        # Calculate softmax over the negative distances
        soft_assignments = F.softmax(-distances, dim=2)
        most_like = torch.argmax(soft_assignments, dim=2)
        softdecision = self.sd_c[most_like]

        return softdecision

class SoftDecisionML16_5(nn.Module):
    def __init__(self, mps_device):
        """
        soft-decision Maximum Likelihood Estimation

        Args:
            C: codebook after BPSK.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result(closest Euclidean distance).
        """
        super(SoftDecisionML16_5, self).__init__()
        self.sd_c = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                  [-1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1],
                                  [-1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1],
                                  [-1, -1, -1,  1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1, -1],
                                  [-1, -1,  1, -1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1,  1],
                                  [-1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
                                  [-1, -1,  1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1, -1, -1, -1],
                                  [-1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1],
                                  [-1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1, -1,  1],
                                  [-1,  1, -1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
                                  [-1,  1, -1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1, -1],
                                  [-1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1,  1,  1],
                                  [-1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
                                  [-1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1],
                                  [-1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
                                  [-1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1],
                                  [ 1, -1, -1, -1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1,  1],
                                  [ 1, -1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1],
                                  [ 1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1, -1,  1, -1],
                                  [ 1, -1, -1,  1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1, -1,  1],
                                  [ 1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1, -1],
                                  [ 1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1,  1, -1,  1],
                                  [ 1, -1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1],
                                  [ 1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1, -1, -1],
                                  [ 1,  1, -1, -1, -1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1],
                                  [ 1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1],
                                  [ 1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1,  1],
                                  [ 1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1],
                                  [ 1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1,  1],
                                  [ 1,  1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1],
                                  [ 1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1,  1, -1,  1, -1],
                                  [ 1,  1,  1,  1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1,  1]], device=mps_device, dtype=torch.float)

    def forward(self,signal):
        # Compute the distance between each input vector and each codeword(euclidean distance)
        distances = torch.cdist(signal, self.sd_c.unsqueeze(0))

        # Calculate softmax over the negative distances
        soft_assignments = F.softmax(-distances, dim=2)
        most_like = torch.argmax(soft_assignments, dim=2)
        softdecision = self.sd_c[most_like]

        return softdecision

