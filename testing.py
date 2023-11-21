import torch
import torch.nn as nn

class LDPCDecoder(nn.Module):
    def __init__(self, H, max_iterations=10):
        super(LDPCDecoder, self).__init__()
        self.H = H
        self.max_iterations = max_iterations

    def forward(self, received_codeword):
        batch_size, _ = received_codeword.size()
        _, n = self.H.size()

        # Convert received_codeword to log likelihood ratios (LLRs)
        llrs = 2 * received_codeword / (1 - received_codeword)

        for iteration in range(self.max_iterations):
            # Compute check node values
            check_node_values = torch.matmul(self.H, llrs.t()).t()

            # Compute updated variable node values
            updated_llrs = llrs + check_node_values

            # Hard decision: LLR to binary
            decoded_codeword = torch.sign(updated_llrs).long()

            # Check if the decoded codeword satisfies parity checks
            syndromes = torch.matmul(decoded_codeword, self.H.t()) % 2

            # Check if all syndromes are zero (no errors)
            if torch.sum(syndromes) == 0:
                break

            # Compute variable node values for the next iteration
            llrs = updated_llrs

        return decoded_codeword

# Example usage
# Define LDPC code matrix (H matrix) and received codeword
H = torch.tensor([[1, 0, 1, 1, 0, 0],
                  [0, 1, 1, 0, 1, 0],
                  [1, 1, 0, 0, 0, 1]])

received_codeword = torch.tensor([[1, 0, 1, 0, 1, 1]])

# Create LDPC decoder
ldpc_decoder = LDPCDecoder(H, max_iterations=10)

# Decode the received codeword
decoded_codeword = ldpc_decoder(received_codeword)

print("Received Codeword:", received_codeword)
print("Decoded Codeword:", decoded_codeword)
