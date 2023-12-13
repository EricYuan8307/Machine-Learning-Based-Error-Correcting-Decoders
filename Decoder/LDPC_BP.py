import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, llr):
        """
        LDPC Belief Propagation.

        Args:
            H: Low density parity code for building tanner graph.
            llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.

        Returns:
            estimated_bits: the output result from belief propagation.
        """

        super(LDPCBeliefPropagation, self).__init__()
        self.llr = llr
        self.H = torch.tensor([[1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0],
                               [0, 1, 0, 0, 1, 1, 0],
                               [1, 0, 0, 1, 0, 0, 1], ], device=mps_device)
        self.num_check_nodes, self.num_variable_nodes = self.H.shape
        self.channel = llr.shape[2]

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes, self.channel),
                                          dtype=torch.float).to(mps_device)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes, self.channel),
                                           dtype=torch.float).to(mps_device)

    def forward(self, max_iter):
        # start_time = time.time()
        for iteration in range(max_iter):

            # Variable to check node messages
            for i in range(self.num_variable_nodes):
                for j in range(self.num_check_nodes):
                    # Compute messages from variable to check nodes
                    connected_checks = self.H[j, :] == 1
                    product = torch.prod(torch.tanh(0.5 * self.messages_v_to_c[connected_checks, j]), dim=0,
                                         keepdim=True)
                    self.messages_v_to_c[i, j] = torch.sign(self.llr[j]) * product

            # Check to variable node messages
            for i in range(self.num_check_nodes):
                for j in range(self.num_variable_nodes):
                    # Compute messages from check to variable nodes
                    connected_vars = self.H[:, j] == 1
                    sum_msgs = torch.sum(self.messages_c_to_v[connected_vars, i]) - self.messages_v_to_c[j, i]
                    self.messages_c_to_v[i, j] = 2 * torch.atan(torch.exp(0.5 * sum_msgs))

        # Calculate the final estimated bits and only take first four bits
        estimated_bits = torch.sign(self.llr) * torch.prod(torch.tanh(0.5 * self.messages_c_to_v))

        return estimated_bits