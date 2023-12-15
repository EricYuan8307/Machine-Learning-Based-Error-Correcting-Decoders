import torch

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, device):
        """
        LDPC Belief Propagation.

        Args:
            H: Low density parity code for building tanner graph.
            llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.

        Returns:
            estimated_bits: the output result from belief propagation.
        """

        super(LDPCBeliefPropagation, self).__init__()
        self.H = torch.tensor([[1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0],
                               [0, 1, 0, 0, 1, 1, 0],
                               [1, 0, 0, 1, 0, 0, 1], ],dtype=torch.float, device=device)
        self.num_check_nodes, self.num_variable_nodes = self.H.shape

    def forward(self, llr, max_iters):
        # Initialize messages
        messages_v_to_c = llr.repeat(1, self.num_check_nodes, 1)
        messages_c_to_v = torch.zeros_like(messages_v_to_c)

        for iteration in range(max_iters):
            # Variable Node Update
            messages_v_to_c = self.variable_node_update(llr, messages_c_to_v)

            # Check Node Update
            messages_c_to_v = self.check_node_update(messages_v_to_c)

        # Final decision
        sums = messages_c_to_v.sum(dim=1).unsqueeze(1)
        llr_final = llr + sums
        return llr_final

    def variable_node_update(self, llr, messages_c_to_v):
        # Update messages from variable nodes to check nodes
        total = llr + messages_c_to_v.sum(dim=1, keepdim=True)
        # result = total - messages_c_to_v

        return total

    def check_node_update(self, messages_v_to_c):
        # # Update messages from check nodes to variable nodes
        alpha = torch.sign(messages_v_to_c)
        beta = torch.abs(messages_v_to_c)

        # Sum of phi(beta_ij) for all i' (excluding the current variable i)
        # # We temporarily set the current variable's message to zero to exclude it from the sum
        original_beta = beta.clone()
        beta[:, :] = 0
        sum_phi_beta = self.phi(beta).sum(dim=1, keepdim=True)

        # # Now we restore the original beta values
        beta = original_beta
        #
        # # Compute the phi of sum_phi_beta, which is the same for all edges connected to the check node
        phi_sum_phi_beta = self.phi(sum_phi_beta)

        # The final message is the product of alpha and the phi of the sum of phi(beta)
        # for each edge, we exclude the current variable's contribution by subtracting its phi(beta)
        messages_c_to_v = alpha * (phi_sum_phi_beta - self.phi(beta))
        # messages_c_to_v = torch.prod(alpha) * phi_sum_phi_beta

        # return result
        return messages_c_to_v

    def phi(self, x):
        # Avoid division by zero and numerical instability when x is very small
        return -torch.log(torch.tanh(x / 2))


# device = torch.device("mps")
#
# llr_output = torch.tensor([[[1, 1, 1, 1, 1, 1, 1]],
#                            [[-1, -2, -3, -4, -5, -6, -7]], ], dtype=torch.float,
#                           device=device)  # torch.Size([2, 1, 7])
#
# # result = torch.sum(llr_output, dim=1)
# # print(result)
#
# iter = 5

# ldpc_bp = LDPCBeliefPropagation(device)
# LDPC_result = ldpc_bp(llr_output, iter)  # LDPC
# print(llr_output)
# print(LDPC_result)
