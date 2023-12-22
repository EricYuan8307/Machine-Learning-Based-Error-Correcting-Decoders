import torch
from Decoder.HardDecision import hard_decision
from Estimation.BitErrorRate import calculate_ber

class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, H, device):
        """
        Initialize the LDPC Belief Propagation Decoder.

        Args:
            H (Tensor): Parity-check matrix of the LDPC code.
        """
        super(LDPCBeliefPropagation, self).__init__()
        self.H = H.to(device)
        self.device = device

    def phi(self, x):
        """Stable computation of log(tanh(x/2)) for belief propagation."""
        # Avoid division by zero
        return 2.0 * torch.atanh(torch.tanh(x / 2.0))

    def forward(self, llr, max_iters=10):
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


# Example usage
device = torch.device("cpu")

H = torch.tensor([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]], dtype=torch.float, device=device)  # Example parity-check matrix

ldpc_decoder = LDPCBeliefPropagation(H, device=device)

# Example LLR input for a batch of 1000 samples
llr_output = torch.tensor([[[20.4128,  -16.3902,  -19.1344,  -15.7405,  -26.6343,   23.9271,   21.8500]],
                           # [[  92.018,  -20.977,  -13.301, -176.342, -154.045,  -58.012,  -11.695]],
                           ], dtype=torch.float,device=device)  # torch.Size([2, 1, 7])
# print(llr_input)

# Decoding
ldpc_decoder = LDPCBeliefPropagation(H, device)
decoded_bits = ldpc_decoder(llr_output)
# print("Decoded Bits:", decoded_bits)

HD_result = hard_decision(decoded_bits, device)
HD_input = hard_decision(llr_output, device)

print("HD_input", HD_input.shape)
print("HD_result", HD_result.shape)

print("BER", calculate_ber(HD_result,HD_input))
