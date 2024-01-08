import torch
from Decoder.HardDecision import hard_decision

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
        self.H = torch.tensor([[[1, 0, 1, 0, 1, 0, 1],
                               [0, 1, 1, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1, 1, 1], ], ],dtype=torch.float, device=device)
        self.num_check_nodes = self.H.shape[1]
        self.num_variable_nodes = self.H.shape[2]
        self.device = device

    def forward(self, llr, max_iters):
        # Initial values
        messages_v_to_c = torch.zeros(llr.shape, dtype=torch.float, device=self.device)

        # print("initial llr", llr)

        for iteration in range(max_iters):
            #  From variable nodes to check nodes
            llr_update = self.H * (llr + messages_v_to_c)
            log_llr_update = self.phi(llr_update)
            sum_log_llr = self.H * torch.sum(log_llr_update)
            sum_log_llr_update = sum_log_llr - log_llr_update
            phi_sum_log_llr_updat = self.phi(sum_log_llr_update)

            sign1 = torch.sign(llr_update)
            masked_sign_phi = sign1 * self.H
            masked_sign_phi_ones = masked_sign_phi + (1 - self.H)
            sign = masked_sign_phi_ones.prod(dim=2) # torch.Size([2, 4])
            sign_mask = self.H * sign.unsqueeze(2)
            sign_mask_update = torch.div(sign_mask, masked_sign_phi_ones)

            # From check nodes to variable nodes
            messages_c_to_v = -phi_sum_log_llr_updat * sign_mask_update

            sum_messages_c_to_v = torch.sum(messages_c_to_v,dim=1)
            final_llr = llr + sum_messages_c_to_v

            est = hard_decision(final_llr, self.device).to(torch.float) # torch.Size([1, 1, 7])
            mult = torch.matmul(est, self.H.transpose(1, 2))%2

            if torch.all(mult == torch.zeros(mult.shape, device=self.device)):
                break
            else: messages_v_to_c = self.H * final_llr - messages_c_to_v


        return final_llr

    def phi(self, x):
        """Stable computation of log(tanh(x/2)) for belief propagation."""
        # Avoid division by zero
        result0 =torch.tanh(x/2)
        result1 = torch.abs(result0)
        result = torch.where(self.H == 1, torch.log(result1), torch.tensor(0.0, device=self.device))
        return result


# device = torch.device("cpu")
#
# # llr_output = torch.tensor([[[20.4128,  -16.3902,  -19.1344,  -15.7405,  -26.6343,   23.9271,   21.8500]],
# #                            [[  92.018,  -20.977,  -13.301, -176.342, -154.045,  -58.012,  -11.695]],
# #                            ], dtype=torch.float,device=device)  # torch.Size([2, 1, 7])
# nr_number = int(1e6)
# llr_output = torch.randn(nr_number, 1, 7, dtype=torch.float, device=device)
# # print("llr_output", llr_output)
# result = torch.zeros(llr_output.shape)
#
# for i in range(nr_number):
#     input = llr_output[i]
#     iter = 5
#     start = time.time()
#     ldpc_bp = LDPCBeliefPropagation(device)
#     LDPC_result = ldpc_bp(input, iter) # LDPC
#     end = time.time()
#     result[i] = LDPC_result
#
#     print("time_elapsed", end - start)
#
# print(result)
#
# # print("LDPC_result: ", LDPC_result)
# #
# # HD_result = hard_decision(LDPC_result, device)
# # HD_input = hard_decision(llr_output, device)
# #
# #
# # print("HD_input", HD_input.shape)
# # print("HD_result", HD_result.shape)
# #
# # print("BER", calculate_ber(HD_result,HD_input))
