import torch
import torch.nn as nn


class SingleLabelNNDecoder1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
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

class SingleLabelNNDecoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.hidden0 = nn.Linear(input_size, hidden_size[0])
        self.hidden1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size[1], output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x = self.hidden0(x)
        x = self.relu(x)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

class SingleLabelNNDecoder_nonfully(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, mask):

        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.mask = mask
        self.apply_mask(mask)
        self.register_hooks()

    def apply_mask(self, mask):
        # ensure mask's shape is same as hidden weight shape
        assert mask.shape == self.hidden.weight.shape
        with torch.no_grad():
            self.hidden.weight *= mask

    def register_hooks(self):
        # 为权重注册一个钩子，每次梯度计算后应用掩码
        self.hidden.weight.register_hook(lambda grad: grad * self.mask)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

class MultiLabelNNDecoder1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x) # In author's article
        x = self.output(x)
        x = self.sigmoid(x)

        return x

    # def __init__(self, input_size, hidden_size, output_size):
    #
    #     super().__init__()
    #     self.fc1 = nn.Linear(input_size, hidden_size)
    #     self.relu = nn.ReLU()
    #     self.softmax = nn.LogSoftmax(dim=2)
    #     self.fc2 = nn.Linear(hidden_size, output_size)
    #     self.sigmoid = nn.Sigmoid()
    #
    # def forward(self, x):
    #     x = self.fc1(x)
    #     # x = self.relu(x) # In author's article
    #     x = self.softmax(x) # for Maximum Likelihood
    #     x = self.fc2(x)
    #     x = self.sigmoid(x)
    #
    #     return x

class MultiLabelNNDecoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
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

class MultiLabelNNDecoder_N(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=2)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.softmax(x) # for Maximum Likelihood
        x = self.output(x)
        x = self.sigmoid(x)

        return x