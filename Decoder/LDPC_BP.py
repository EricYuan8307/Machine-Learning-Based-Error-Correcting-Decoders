import torch
from Decoder.HardDecision import hard_decision
from Estimation.BitErrorRate import calculate_ber

# class LDPCBeliefPropagation(torch.nn.Module):
#     def __init__(self, device):
#         """
#         LDPC Belief Propagation.
#
#         Args:
#             H: Low density parity code for building tanner graph.
#             llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.
#
#         Returns:
#             estimated_bits: the output result from belief propagation.
#         """
#
#         super(LDPCBeliefPropagation, self).__init__()
#         self.H = torch.tensor([[[1, 0, 1, 0, 1, 0, 1],
#                                [0, 1, 1, 0, 0, 1, 1],
#                                [0, 0, 0, 1, 1, 1, 1], ], ],dtype=torch.float, device=device)
#         self.num_check_nodes = self.H.shape[1]
#         self.num_variable_nodes = self.H.shape[2]
#
#     def forward(self, llr, max_iters, device):
#         # Initial values
#         messages_v_to_c = llr * self.H
#         new_msg = torch.zeros(llr.shape, dtype=torch.float, device=device)
#         messages_c_to_v = torch.zeros(llr.shape, dtype=torch.float, device=device)
#
#         # print("initial llr", llr)
#
#         for iteration in range(max_iters):
#             # Check nodes to variable nodes
#             messages_c_to_v = messages_c_to_v + llr # torch.Size([2, 1, 7])
#
#             check_node = torch.sum(messages_c_to_v, dim=2) #torch.Size([2, 4])
#             q_ij = self.H * check_node.unsqueeze(2)
#             q_iij = q_ij - messages_c_to_v # V_{i'j} # torch.Size([2, 4, 7])
#
#             A_mj = torch.sum(self.phi(q_iij, device),dim=2) # torch.Size([2, 4])
#             phi_A_mj = torch.log(torch.abs(torch.div((torch.exp(A_mj) -1), (torch.exp(A_mj) + 1))) + 1e-9)  # torch.Size([2, 4])
#             phi_A_mj_mask = self.H * phi_A_mj.unsqueeze(2) #torch.Size([2, 4, 7])
#             # print("phi_A_mj_mask",phi_A_mj_mask)
#
#             # S_{mj} calculate the sign in each check node
#             sign1 = torch.sign(q_iij)
#             masked_sign_phi = sign1 * self.H + (1 - self.H)
#             sign = masked_sign_phi.prod(dim=2) # torch.Size([2, 4])
#             sign_mask = self.H * sign.unsqueeze(2)
#             # print("masked_sign_phi", sign_mask)
#
#             messages_c_to_v = sign_mask*phi_A_mj_mask # torch.Size([2, 4, 7])
#             # print("messages_c_to_v", messages_c_to_v.shape)
#
#             # Variable nodes to check nodes
#             sum_messages_c_to_v = torch.sum(messages_c_to_v, dim=1).unsqueeze(1) # torch.Size([2, 1, 7])
#             # print("sum_R",sum_messages_c_to_v)
#
#             total_llr = llr + sum_messages_c_to_v
#             messages_v_to_c = self.H * (total_llr) - messages_c_to_v
#             # print("next_messages_v_to_c",messages_v_to_c)
#
#             # Update messages from check nodes to variable nodes
#             messages_c_to_v = messages_v_to_c
#
#         return total_llr
#
#     def phi(self, x, device):
#         # result = torch.where(self.H == 1, torch.log(torch.abs(torch.tanh(x))), torch.tensor(0., device=device))
#         ex = torch.div((torch.exp(x) -1), (torch.exp(x) + 1))
#         abs_ex = torch.abs(ex)
#         result = torch.where(self.H == 1, torch.log(abs_ex + 1e-9), torch.tensor(0., device=device))
#
#         return result

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, device):
        """
        Initialize the LDPC Belief Propagation Decoder.

        Args:
            H (Tensor): Parity-check matrix of the LDPC code.
        """
        super(LDPCBeliefPropagation, self).__init__()
        self.H = torch.tensor([[1, 0, 1, 0, 1, 0, 1],
                                [0, 1, 1, 0, 0, 1, 1],
                                [0, 0, 0, 1, 1, 1, 1]],dtype=torch.float, device=device)
        self.device = device

    def phi(self, x):
        """Stable computation of log(tanh(x/2)) for belief propagation."""
        # Avoid division by zero
        return 2.0 * torch.atanh(torch.tanh(x / 2.0))

    def forward(self, llr, max_iters):
        """
        Perform belief propagation decoding.

        Args:
            llr (Tensor): Log-likelihood ratios of shape (batch_size, 1, num_variable_nodes).
            max_iters (int): Maximum number of iterations.

        Returns:
            Tensor: Estimated bit values for each sample in the batch.
        """
        batch_size, _, num_variable_nodes = llr.shape
        num_check_nodes, _ = self.H.shape

        # Initialize messages for each sample in the batch
        messages_c_to_v = torch.zeros(batch_size, num_check_nodes, num_variable_nodes, device=self.device)

        for _ in range(max_iters):
            # Reshape LLR and messages for batch processing
            llr_expanded = llr.expand_as(messages_c_to_v)
            messages_c_to_v_expanded = messages_c_to_v.expand_as(llr_expanded)

            # Check nodes to variable nodes
            tanh_llr = torch.tanh((llr_expanded + messages_c_to_v_expanded) / 2)
            prod_tanh = torch.prod(tanh_llr ** self.H, dim=2, keepdim=True)
            print("prod_tanh", prod_tanh)
            messages_c_to_v = self.phi((prod_tanh / (tanh_llr + 1e-10)) ** self.H)

            # Variable nodes to check nodes
            total_llr = llr.squeeze(1) + torch.sum(messages_c_to_v, dim=1)
            messages_v_to_c = total_llr.unsqueeze(1) - messages_c_to_v

            # Update messages from check nodes to variable nodes
            messages_c_to_v = messages_v_to_c

        # Make final decision
        total_llr = llr.squeeze(1) + torch.sum(messages_c_to_v, dim=1)
        total_llr = total_llr.unsqueeze(1)
        return total_llr

device = torch.device("cpu")

# llr_output = torch.tensor([[[-160.1754,  -50.0127,  -40.4241,   30.8286,   40.4707,  140.4069,   1.706]],
#         [[  92.018,  -20.977,  -13.301, -176.342, -154.045,  -58.012,  -11.695]]], dtype=torch.float,
#                           device=device)  # torch.Size([2, 1, 7])
llr_output = torch.randn(1, 1, 7, dtype=torch.float, device=device)*10
print("llr_output", llr_output)

iter = 2

ldpc_bp = LDPCBeliefPropagation(device)
LDPC_result = ldpc_bp(llr_output, iter)  # LDPC

HD_result = hard_decision(LDPC_result, device)
HD_input = hard_decision(llr_output, device)

# print("HD_input", HD_input)
# print("HD_result", HD_result)
#
# print("BER", calculate_ber(HD_result,HD_input))



