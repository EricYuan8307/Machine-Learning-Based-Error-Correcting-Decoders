import torch

class PCC_encoders(torch.nn.Module):
    def __init__(self, generator_matrix):
        """
        Use matrix to encode the data.

        Args:
            data: data received from the signal
            generator matrix: generate the parity code

        Returns:
            encoded data
        """
        super(PCC_encoders, self).__init__()

        # Define the generator matrix
        self.generator_matrix = generator_matrix

    def forward(self, input_data):
        # Perform matrix multiplication to encode the data
        result_tensor = torch.matmul(input_data, self.generator_matrix.t()) % 2

        return result_tensor