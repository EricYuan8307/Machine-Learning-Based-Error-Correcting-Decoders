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
                               [0, 0, 0, 1, 1, 1, 1]]], dtype=torch.float64, device=device)
        # self.H = torch.tensor([[[1, 1, 0, 1, 1, 0, 0],
        #                        [1, 0, 1, 1, 0, 1, 0],
        #                        [0, 1, 1, 1, 0, 0, 1]]], dtype=torch.float64, device=device)
        self.num_check_nodes = self.H.shape[1]
        self.num_variable_nodes = self.H.shape[2]
        self.device = device

    def forward(self, llr, max_iters):
        # Initial values
        messages_v_to_c = torch.zeros(llr.shape, dtype=torch.float64, device=self.device)
        # print("initial llr", llr)

        for iteration in range(max_iters):
            #  From variable nodes to check nodes
            llr_update = self.H * llr if iteration == 0 else messages_v_to_c
            # llr_update = self.H * llr
            log_llr_update = self.phi(llr_update)
            sum_log_llr = torch.sum(log_llr_update, dim=2).unsqueeze(2)
            masked_sum_log_llr = self.H * sum_log_llr
            sum_log_llr_update = masked_sum_log_llr - log_llr_update
            phi_sum_log_llr_update = self.phi(sum_log_llr_update)

            sign0 = torch.sign(llr_update)
            masked_sign = sign0 * self.H
            masked_sign_ones = masked_sign + (1 - self.H)
            sign = masked_sign_ones.prod(dim=2) # torch.Size([2, 4])
            sign_mask = self.H * sign.unsqueeze(2)
            sign_mask_update = torch.div(sign_mask, masked_sign_ones)

            # From check nodes to variable nodes
            messages_c_to_v = -phi_sum_log_llr_update * sign_mask_update

            sum_messages_c_to_v = torch.sum(messages_c_to_v,dim=1)
            llr_total = llr + sum_messages_c_to_v

            est = hard_decision(llr_total, self.device).to(torch.float64) # torch.Size([1, 1, 7])
            mult = torch.matmul(est, self.H.transpose(1, 2))%2

            if torch.all(mult == torch.zeros(mult.shape, device=self.device)):
                break
            else:
                M = self.H * llr_total
                messages_v_to_c = M - messages_c_to_v


        return llr_total

    def phi(self, x):
        """Stable computation of log(tanh(x/2)) for belief propagation."""
        # Avoid division by zero
        result0 =torch.tanh(x/2)
        result1 = torch.abs(result0)
        result = torch.where(self.H == 1, torch.log(result1), torch.tensor(0.0, device=self.device))
        return result


# device = torch.device("cpu")
#
# llr_output = torch.tensor([[[-0.227861747916372,-0.371519925363314,-0.212513607051216,0.232155258660031,0.254121391049793,0.323201541013432,-0.094486115781860]]], dtype=torch.float,device=device)
# # llr_output = torch.tensor([[[20.4128, -16.3902, -19.1344, -15.7405, -26.6343, 23.9271, 21.8500]]], dtype=torch.float,device=device)  # torch.Size([2, 1, 7])
#
# iter = 5
# ldpc_bp = LDPCBeliefPropagation(device)
# LDPC_result = ldpc_bp(llr_output, iter) # LDPC
#
# print("LDPC_result: ", LDPC_result)
