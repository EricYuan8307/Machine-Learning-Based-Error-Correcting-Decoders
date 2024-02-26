import torch
import torch.nn as nn


class SingleLabelNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

class SingleLabelNNDecoder_nonfully(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mask):

        super().__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.mask = mask

    def forward(self, x):
        x = self.hidden(x)
        x = torch.matmul(x, self.mask)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

class MultiLabelNNDecoder1(nn.Module):
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

class MultiLabelNNDecoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden0 = nn.Linear(input_size, hidden_size[0])
        self.hidden1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size[1], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden0(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x

class MultiLabelNNDecoder3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden0 = nn.Linear(input_size, hidden_size[0])
        self.hidden1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.hidden2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size[2], output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden0(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)

        return x