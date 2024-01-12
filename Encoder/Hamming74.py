import torch

# Hamming(7,4) Encoder
class hamming_encoder(torch.nn.Module):
    def __init__(self, device):
        """
        Use Hamming(7,4) to encode the data.

        Args:
            data: data received from the Hamming(7,4) encoder(Tensor)
            generator matrix: generate the parity code

        Returns:
            encoded data: 4 bits original info with 3 parity code.
        """
        super(hamming_encoder, self).__init__()

        # Define the generator matrix for Hamming(7,4)
        self.generator_matrix = torch.tensor([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],], dtype=torch.float, device=device)

    def forward(self, input_data):
        # Ensure input_data has shape (batch_size)
        # assert input_data.size(0) == self.generator_matrix.shape[1], "Input data must have same generator matrix row number bits."

        # Perform matrix multiplication to encode the data
        result_tensor = torch.matmul(input_data.to(torch.float), self.generator_matrix.t()) % 2

        return result_tensor

# device = (torch.device("mps") if torch.backends.mps.is_available()
#                                     else (torch.device("cuda") if torch.backends.cuda.is_available()
#                                           else torch.device("cpu")))
# bits = torch.tensor([[[1, 1, 0, 1]],
#                          [[0, 1, 0, 0,]]], dtype=torch.int, device=device)
#
# encoder = hamming_encoder(device)
# print(encoder(bits))