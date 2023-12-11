import torch

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, H, max_iter=50):
        super(LDPCBeliefPropagation, self).__init__()
        self.H = H
        self.max_iter = max_iter
        self.num_check_nodes, self.num_variable_nodes = H.shape

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes),dtype=torch.float)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes),dtype=torch.float)

    def forward(self, llr):
        for iteration in range(self.max_iter):
            # Variable to check node messages
            for i in range(self.num_variable_nodes):
                for j in range(self.num_check_nodes):
                    # Compute messages from variable to check nodes
                    connected_checks = self.H[j, :] == 1
                    product0 = 0.5 * self.messages_v_to_c[connected_checks, j]
                    product = torch.prod(torch.tanh(product0))
                    # product = torch.prod(torch.tanh(0.5 * self.messages_v_to_c[connected_checks, i]))
                    self.messages_v_to_c[i, j] = torch.sign(llr[i]) * product

            # Check to variable node messages
            for i in range(self.num_check_nodes):
                for j in range(self.num_variable_nodes):
                    # Compute messages from check to variable nodes
                    connected_vars = self.H[:, j] == 1
                    sum_msg0 = self.messages_c_to_v[connected_vars, i]
                    sum_msgs = torch.sum(sum_msg0) - self.messages_v_to_c[j, i]
                    # sum_msgs = torch.sum(self.messages_v_to_c[connected_vars, i]) - self.messages_v_to_c[j, i]
                    self.messages_c_to_v[i, j] = 2 * torch.atan(torch.exp(0.5 * sum_msgs))

        # Calculate the final estimated bits
        estimated_bits = torch.sign(llr) * torch.prod(torch.tanh(0.5 * self.messages_c_to_v), dim=0)
        estimated_bits = torch.where(estimated_bits>0, torch.tensor(0), torch.tensor(1))

        return estimated_bits

# Example usage:
# Define LDPC parameters 
H = torch.tensor([ [1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 0, 1],])

llr_demodulator_output = torch.tensor([-0.0558,  0.0314,  0.0457,  0.0374, -0.0477, -0.0378,  0.0647])

ldpc_bp = LDPCBeliefPropagation(H)
estimated_bits = ldpc_bp(llr_demodulator_output)

print("LLR Demodulator:", llr_demodulator_output)
print("Estimated Bits:", estimated_bits)
