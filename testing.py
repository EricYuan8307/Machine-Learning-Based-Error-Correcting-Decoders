import torch

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, H):
        super(LDPCBeliefPropagation, self).__init__()
        self.H = H
        self.num_check_nodes, self.num_variable_nodes = H.shape

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes),dtype=torch.float)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes),dtype=torch.float)

    def forward(self, llr, max_iter=50):
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

        # Calculate the final estimated bits
        estimated_bits = torch.sign(llr) * torch.prod(torch.tanh(0.5 * self.messages_c_to_v), dim=0)
        estimated_bits = torch.where(estimated_bits>0, torch.tensor(1), torch.tensor(0))
        estimated_bits = estimated_bits[0:4]

        return estimated_bits

# Example usage:
# Define LDPC parameters 
H = torch.tensor([ [1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 0, 1],])
iter = 10
llr_demodulator_output = torch.tensor([-10.7472,  10.0925, -12.2140, -11.3412,  10.6539,  -8.9250,  10.3458])

ldpc_bp = LDPCBeliefPropagation(H)
estimated_bits = ldpc_bp(llr_demodulator_output, iter)

print("LLR Demodulator:", llr_demodulator_output)
print("Estimated Bits:", estimated_bits)
print("[0, 1, 0, 0, 1, 0, 1]")
