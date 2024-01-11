import torch
import torch.nn as nn
import torch.optim as optim

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Estimation.BitErrorRate import calculate_ber
from Decoder.HammingDecoder import Hamming74decoder
from Decoder.NNDecoder import SingleLableNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure

device = (torch.device("mps") if torch.backends.mps.is_available()
                                    else (torch.device("cuda") if torch.backends.cuda.is_available()
                                          else torch.device("cpu")))
    # device = torch.device("cpu")

# Create an instance of the SimpleNN class
model = SingleLableNNDecoder(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    outputs = model(data)

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