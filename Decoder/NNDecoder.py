import torch
import torch.nn as nn
import torch.optim as optim


class SingleLableNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = x.view(x.shape[0], input_size, 2) # torch.Size([5, 1, 14]) to torch.Size([5, 7, 2])
        x = self.softmax(x)
        x = torch.argmax(x, dim=2).unsqueeze(1).to(torch.float).requires_grad_(True) # torch.Size([5, 1, 7])

        return x


# device = (torch.device("mps") if torch.backends.mps.is_available()
#           else (torch.device("cuda") if torch.backends.cuda.is_available()
#                 else torch.device("cpu")))
device = torch.device("cpu")

# Define your dataset and dataloader
# Example data and labels (replace with your own dataset)
nr = 5
data = torch.randn(size=(nr, 1, 7), dtype=torch.float, device=device, requires_grad=True)
# data = torch.ones(size=(2, 1, 1), dtype=torch.float, device=device)
# data = torch.tensor([[[2, 0], [0, 1]],
#                      # [[0, 1],[0, 1]],
#                      ], dtype=torch.float, device=device)
print("data shape: ", data.shape)
labels = torch.randint(0, 2, size=(nr, 1, 7), dtype=torch.float, device=device)

# Hyperparameters
input_size = data.shape[2]
hidden_size = data.shape[2]
output_size = 2*data.shape[2]
learning_rate = 0.01
epochs = 10000



# Create an instance of the SimpleNN class
model = SingleLableNNDecoder(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model(data)
    # outputs = torch.tensor(outputs, requires_grad=True)
    # x = torch.argmax(x, dim=2).unsqueeze(1).to(torch.float) # torch.Size([5, 1, 7])

    # Compute the loss
    loss = criterion(outputs, labels)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

# # Testing the model
# with torch.no_grad():
#     test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
#     predicted_labels = model(test_data)
#     predicted_labels = (predicted_labels > 0.5).float()  # Threshold the outputs
#
#     print("Predicted labels:")
#     print(predicted_labels)