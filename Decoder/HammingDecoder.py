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
            llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.

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
                               [1, 1, 1, 1, 1, 1, 1],], device=mps_device)

        self.r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    def forward(self, harddecision):
        # Calculate Hamming distances
        distances = torch.sum(harddecision != self.C, dim=2)

        # Find indices of minimum distances
        min_distance = torch.argmin(distances, dim=1)

        # Information Replace
        harddecision = self.C[min_distance]

        result = torch.matmul(harddecision.to(torch.float), self.r.T).to(torch.int)

        return result

# # hd_output = torch.randint(0, 2, size=(3, 1, 7), dtype=torch.int).to(mps_device)
# # print("hdout0",hd_output)
# hd_output1 = torch.tensor([[[0, 0, 0, 0, 0, 0, 0]],
#                           [[1, 1, 1, 0, 0, 0, 0]],
#                           [[1, 0, 0, 1, 1, 0, 0]]],).to(mps_device)
# # print("hd_output1",hd_output1)
#
#
# decoder = HammingDecoder()
# result = decoder(hd_output1)
# print("result",result)