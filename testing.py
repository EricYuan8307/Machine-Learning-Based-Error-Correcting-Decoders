import torch



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
                               [1, 0, 0, 1, 0, 0, 1], ])
        self.num_check_nodes, self.num_variable_nodes = self.H.shape
        self.channel = llr.shape[2]

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes, self.channel), dtype=torch.float)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes, self.channel), dtype=torch.float)

    def forward(self, max_iter):
        for iteration in range(max_iter):
            # Variable to check node messages
            for i in range(self.num_variable_nodes):
                for j in range(self.num_check_nodes):
                    # Compute messages from variable to check nodes
                    connected_checks = self.H[j, :] == 1
                    product = torch.prod(torch.tanh(0.5 * self.messages_v_to_c[connected_checks, j]),dim=0, keepdim=True)
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
        estimated_bits = torch.where(estimated_bits > 0, torch.tensor(1), torch.tensor(0))
        estimated_bits = estimated_bits[:, :, 0:4]

        return estimated_bits


# %%
# input data LLR with 7-bit message
nr_codewords = 1000000
llr_output = torch.randint(-14, 14, size=(nr_codewords, 1, 7), dtype=torch.float) # torch.Size([10, 1, 7])

# # Define LDPC parameters
# H = torch.tensor([ [1, 1, 1, 0, 0, 0, 0],
#                    [0, 0, 1, 1, 1, 0, 0],
#                    [0, 1, 0, 0, 1, 1, 0],
#                    [1, 0, 0, 1, 0, 0, 1],])
iter = 10
ldpc_bp = LDPCBeliefPropagation(llr_output)

#loop all the llr and get result.
# for i in range(llr_output.shape[0]):
#     bp_data = llr_output[i]
estimated_result = ldpc_bp(iter)
final_result = estimated_result

print(final_result)