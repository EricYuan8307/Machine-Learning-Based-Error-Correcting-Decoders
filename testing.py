import torch
mps_device = torch.device("mps")


class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, H):
        """
        LDPC Belief Propagation.

        Args:
            H: Low density parity code for building tanner graph.
            llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.

        Returns:
            estimated_bits: the output result from belief propagation.
        """

        super(LDPCBeliefPropagation, self).__init__()
        self.H = H
        self.num_check_nodes, self.num_variable_nodes = H.shape

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes), dtype=torch.float).to(mps_device)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes), dtype=torch.float).to(mps_device)

    def forward(self, llr, max_iter):
        for iteration in range(max_iter):
            # Variable to check node messages
            for i in range(self.num_variable_nodes):
                for j in range(self.num_check_nodes):
                    # Compute messages from variable to check nodes
                    connected_checks = self.H[j, :] == 1
                    product = torch.prod(torch.tanh(0.5 * self.messages_v_to_c[connected_checks, j]))
                    self.messages_v_to_c[i, j] = torch.sign(llr[i]) * product

            # Check to variable node messages
            for i in range(self.num_check_nodes):
                for j in range(self.num_variable_nodes):
                    # Compute messages from check to variable nodes
                    connected_vars = self.H[:, j] == 1
                    sum_msgs = torch.sum(self.messages_c_to_v[connected_vars, i]) - self.messages_v_to_c[j, i]
                    self.messages_c_to_v[i, j] = 2 * torch.atan(torch.exp(0.5 * sum_msgs))

        # Calculate the final estimated bits and only take first four bits
        estimated_bits = torch.sign(llr) * torch.prod(torch.tanh(0.5 * self.messages_c_to_v), dim=0)
        tensor_1 = torch.tensor(1, device=mps_device)
        tensor_0 = torch.tensor(0, device=mps_device)
        estimated_bits = torch.where(estimated_bits > 0, tensor_1, tensor_0)
        estimated_bits = estimated_bits[0:4]

        return estimated_bits


# %%
# Define LDPC parameters
H = torch.tensor([ [1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 0, 1],], device=mps_device)
iter = 10
ldpc_bp = LDPCBeliefPropagation(H)

llr_output = torch.tensor([-10.7472,  10.0925, -12.2140, -11.3412,  10.6539,  -8.9250,  10.3458]).to(mps_device)

# Store the final result from LDPC
tensor_size = torch.Size([1, 4])
final_result = torch.zeros(tensor_size).to(mps_device)

#loop all the llr and get result.
# for i in range(llr_output.shape[0]):
#     bp_data = llr_output[i]
estimated_result = ldpc_bp(llr_output, iter)
final_result = estimated_result

print(final_result)