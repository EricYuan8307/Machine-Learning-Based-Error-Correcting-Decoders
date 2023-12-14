import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))


class HammingDecoder(torch.nn.Module):
    def __init__(self,):
        """
        LDPC Belief Propagation.

        Args:
            H: Low density parity code for building tanner graph.

        Returns:
            estimated_bits: the output result from belief propagation.
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
        min_distance = torch.argmax(distances, dim=1)

        # Information Replace
        harddecision1 = self.C[min_distance]

        result = torch.matmul(harddecision1.to(torch.float), self.r.T).to(torch.int)
        result = result.unsqueeze(1)

        return result
