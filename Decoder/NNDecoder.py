import torch.nn as nn


class SingleLabelNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)
        # self.softmax = nn.LogSoftmax(dim=2) # dim=2 performance worse than dim=0

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

class MultiLabelNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x