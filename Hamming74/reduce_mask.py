import torch

class MaskMatrix(torch.nn.Module):
    def __init__(self, device):
        """
        Initializes the SLNN, N=7 mask.

        Args:
            device: The device to run the encoder on.

        Returns:
            Specific the mask between input layer and hidden layer
        """
        super(MaskMatrix, self).__init__()

        # Define the device
        self.device = device

    def forward(self, edge_delete, encoded, hidden_size):
        if edge_delete == 0 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1]], dtype=torch.float, device=self.device)

        if edge_delete == 9 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1, 1, 1],
                                [0, 1, 1, 1, 1, 1, 1],
                                [1, 0, 1, 1, 1, 1, 0],
                                [1, 0, 1, 1, 1, 1, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 14 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 1, 1, 1],
                                [0, 0, 1, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [0, 1, 1, 1, 1, 0, 1],
                                [1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 1, 1, 1, 1, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 19 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0],
                                [0, 0, 0, 1, 1, 1, 1],
                                [0, 0, 0, 1, 1, 1, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [0, 1, 1, 0, 1, 0, 0],
                                [1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 1, 1, 1, 1, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 24 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0],
                                [0, 0, 0, 1, 1, 1, 1],
                                [0, 0, 0, 0, 1, 1, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1, 0, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 29 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 1, 0, 1, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [1, 0, 1, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 1, 0, 0, 0, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 34 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[1, 0, 1, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [1, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 1, 1, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0, 1]], dtype=torch.float, device=self.device)

        elif edge_delete == 39 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=self.device)

        elif edge_delete == 40 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                                [0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [1, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=self.device)

        elif edge_delete == 41 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=self.device)

        elif edge_delete == 42 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=self.device)

        elif edge_delete == 43 and encoded == 7 and hidden_size == 7:
            mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 1, 0],
                                [0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=self.device)

        return mask