import torch
from Decoder.HardDecision import hard_decision
from Estimation.BitErrorRate import calculate_ber

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
        self.H = torch.tensor([[[1, 1, 1, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1, 0, 0],
                               [0, 1, 0, 0, 1, 1, 0],
                               [1, 0, 0, 1, 0, 0, 1], ], ],dtype=torch.float, device=device)
        self.num_check_nodes = self.H.shape[1]
        self.num_variable_nodes = self.H.shape[2]

    def forward(self, llr, max_iters, device):
        # Initial values
        messages_v_to_c = llr * self.H
        new_msg = torch.zeros(llr.shape, dtype=torch.float, device=device)
        messages_c_to_v = torch.zeros(llr.shape, dtype=torch.float, device=device)

        # print("initial llr", llr)

        for iteration in range(max_iters):

            messages_c_to_v = messages_c_to_v + llr # torch.Size([2, 1, 7])

            check_node = torch.sum(messages_c_to_v, dim=2) #torch.Size([2, 4])
            q_ij = self.H * check_node.unsqueeze(2)
            q_iij = q_ij - messages_c_to_v # V_{i'j} # torch.Size([2, 4, 7])

            A_mj = torch.sum(self.phi(q_iij, device),dim=2) # torch.Size([2, 4])
            phi_A_mj = torch.log(torch.abs(torch.div((torch.exp(A_mj) -1), (torch.exp(A_mj) + 1))) + 1e-9)  # torch.Size([2, 4])

            # Check nodes to variable nodes
            phi_A_mj_mask = self.H * phi_A_mj.unsqueeze(2) #torch.Size([2, 4, 7])
            # print("phi_A_mj_mask",phi_A_mj_mask)

            # S_{mj} calculate the sign in each check node
            sign1 = torch.sign(q_iij)
            masked_sign_phi = sign1 * self.H + (1 - self.H)
            sign = masked_sign_phi.prod(dim=2) # torch.Size([2, 4])
            sign_mask = self.H * sign.unsqueeze(2)
            # print("masked_sign_phi", sign_mask)


            messages_c_to_v = sign_mask*phi_A_mj_mask # torch.Size([2, 4, 7])
            # print("messages_c_to_v", messages_c_to_v.shape)

            # R_output = self.H * R.unsqueeze(2) # torch.Size([2, 4, 7])
            # print("R_output",R_output.shape)
            sum_messages_c_to_v = torch.sum(messages_c_to_v, dim=1).unsqueeze(1) # torch.Size([2, 1, 7])
            # print("sum_R",sum_messages_c_to_v)

            # Variable nodes to check nodes
            total_llr = llr + sum_messages_c_to_v
            # print("total_llr",total_llr)
            # print("total_llr", total_llr.shape)

            messages_v_to_c = self.H * (total_llr) - messages_c_to_v
            print("next_messages_v_to_c",messages_v_to_c)

            # Update messages from check nodes to variable nodes
            messages_c_to_v = messages_v_to_c

            # messages_v_to_c = qj - R_output
            # print(messages_v_to_c.shape)

        return total_llr

    def phi(self, x, device):
        # result = torch.where(self.H == 1, torch.log(torch.abs(torch.tanh(x))), torch.tensor(0., device=device))
        ex = torch.div((torch.exp(x) -1), (torch.exp(x) + 1))
        abs_ex = torch.abs(ex)
        result = torch.where(self.H == 1, torch.log(abs_ex + 1e-9), torch.tensor(0., device=device))

        return result


device = torch.device("cpu")

# llr_output = torch.tensor([[[1, 1, 1, 1, 1, 1, 1]],
#                            [[-11, -2, -3, -4, -5, -6, -7]], ], dtype=torch.float,
#                           device=device)  # torch.Size([2, 1, 7])
llr_output = torch.randn(1, 1, 7, dtype=torch.float, device=device)

# result = torch.sum(llr_output, dim=1)
# print(result)

iter = 2

ldpc_bp = LDPCBeliefPropagation(device)
LDPC_result = ldpc_bp(llr_output, iter ,device)  # LDPC

# HD_result = hard_decision(LDPC_result, device)
# HD_input = hard_decision(llr_output, device)
#
# print("HD_input", HD_input)
# print("HD_result", HD_result)
#
# print("BER", calculate_ber(HD_result,HD_input))



