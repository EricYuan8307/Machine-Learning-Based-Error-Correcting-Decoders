import torch

# Hamming(7,4) Encoder:
class hamming74_encoder(torch.nn.Module):
    def __init__(self, device):
        """
        Use Hamming(7,4) to encode the data.

        Args:
            data: data received from the Hamming(7,4) encoder(Tensor)
            generator matrix: generate the parity code

        Returns:
            encoded data: 4 bits original info with 3 parity code.
        """
        super(hamming74_encoder, self).__init__()

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
        # Perform matrix multiplication to encode the data
        result_tensor = torch.matmul(input_data.to(torch.float), self.generator_matrix.t()) % 2

        return result_tensor


# (10,5)Parity Check Code Encoder:
class Parity10_5_encoder(torch.nn.Module):
    def __init__(self, device):
        """
        Use Hamming(7,4) to encode the data.

        Args:
            data: data received from the Hamming(7,4) encoder(Tensor)
            generator matrix: generate the parity code

        Returns:
            encoded data: 4 bits original info with 3 parity code.
        """
        super(Parity10_5_encoder, self).__init__()

        # Define the generator matrix for Hamming(7,4)
        self.generator_matrix = torch.tensor([
            [1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0],], dtype=torch.float, device=device)

    def forward(self, input_data):
        # Perform matrix multiplication to encode the data
        result_tensor = torch.matmul(input_data.to(torch.float), self.generator_matrix.t()) % 2

        return result_tensor



device = torch.device("cpu")
encoder = Parity10_5_encoder(device)

from Encode.Generator import generator
code = generator(10, 5, device)
print("code", code)
encoded_codeword = encoder(code)
print("encoded_codeword", encoded_codeword)
