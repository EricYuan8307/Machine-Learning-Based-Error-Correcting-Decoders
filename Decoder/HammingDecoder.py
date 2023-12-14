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

        # print("input harddecision shape: ", harddecision.shape)

        # Calculate Hamming distances
        # print("C", self.C.shape)
        distances = torch.sum(harddecision == self.C, dim=2)
        # print("distances",distances)

        # Find indices of minimum distances
        min_distance = torch.argmax(distances, dim=1)
        # print("min distance",min_distance)

        # Information Replace
        harddecision1 = self.C[min_distance]
        # print("after hard-decision",harddecision.shape)

        result = torch.matmul(harddecision1.to(torch.float), self.r.T).to(torch.int)
        result = result.unsqueeze(1)

        # bits = torch.randint(0, 2, size=(2, 1, 7), dtype=torch.int).to(mps_device)
        # result2 = torch.matmul(bits.to(torch.float), self.r.T).to(torch.int)
        # print("result2", result2.shape)


        return result


# hd_output1 = torch.tensor([[[1, 0, 1, 0, 1, 0, 1]],
#                           [[1, 0, 0, 1, 1, 0, 0]]],).to(mps_device)
# # print("hd_output1",hd_output1)
# #
# #
# decoder = HammingDecoder()
# result = decoder(hd_output1)
# print("result",result)