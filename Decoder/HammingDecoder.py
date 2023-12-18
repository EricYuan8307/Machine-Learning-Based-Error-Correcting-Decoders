import torch
from torch import nn
import torch.nn.functional as F
from Encoder.BPSK import bpsk_modulator

class HammingDecoder(torch.nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            C: Use this matrix to calculate the closest hamming distance.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result.
        """

        super(HammingDecoder, self).__init__()
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

        self.r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    def forward(self, harddecision):
        # Calculate Hamming distances
        distances = torch.sum(harddecision == self.C, dim=2)

        # Find indices of minimum distances
        most_like = torch.argmax(distances, dim=1)

        # Information Replace
        harddecision_final = self.C[most_like]

        result = torch.matmul(harddecision_final.to(torch.float), self.r.T).to(torch.int)
        result = result.unsqueeze(1)

        return result

class SoftDecisionDecoder(nn.Module):
    def __init__(self, mps_device):
        """
        hard-decision Maximum Likelihood Estimation

        Args:
            C: Use this matrix to calculate the closest hamming distance.
            mps_device: Move Data on Specific device for computing(GPU).

        Returns:
            result: The final estimate result.
        """

        super(SoftDecisionDecoder, self).__init__()
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
                               [1, 1, 1, 1, 1, 1, 1], ], device=mps_device, dtype=torch.float)

        self.r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1], ], device=mps_device, dtype=torch.float)

    def forward(self,llr):

        modulator = bpsk_modulator()
        sd_c = modulator(self.C, device).to(torch.float)

        # Compute the distance between each input vector and each codeword
        # Here, using euclidean distance for simplicity, but this can be adapted
        # to a more suitable distance metric for log-likelihoods
        distances = torch.cdist(llr, sd_c.unsqueeze(0))

        # Calculate softmax over the negative distances (as softmax is exp(-distance))
        # to represent the probability of each codeword being the correct one
        soft_assignments = F.softmax(-distances, dim=2)
        print("soft-assignment", soft_assignments)

        # most_like = torch.argmin(soft_assignments, dim=1)
        #
        # harddecision_final = self.C[most_like]
        #
        # result = torch.matmul(harddecision_final.to(torch.float), self.r.T).to(torch.int)
        #
        # result = result.unsqueeze(1)
        #
        # return result



device = (torch.device("mps") if torch.backends.mps.is_available()
          else (torch.device("cuda") if torch.backends.cuda.is_available()
                else torch.device("cpu")))

# llr = torch.tensor([[[1, 1, 0, 1, 5, 6, 1.2]],
#                      [[0, 1, 0, 0, 5, 6, 11]]], dtype=torch.float, device=device)

llr = torch.randint(-2, 2, size=(10, 1, 7), dtype=torch.float, device=device)
# print(llr)

sddecoder = SoftDecisionDecoder(device)
sdcodebook = sddecoder(llr)
print("llr:",llr)
# print("sdcodebook:",sdcodebook)