import torch
import torch.nn as nn

class SingleLableNNDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()
        self.input_size = input_size
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = x.view(x.shape[0], self.input_size, 2) # torch.Size([5, 1, 14]) to torch.Size([5, 7, 2])
        x = self.softmax(x)
        x = torch.argmax(x, dim=2).unsqueeze(1).to(torch.float).requires_grad_(True) # torch.Size([5, 1, 7])

        return x


import torch
import torch.nn as nn
import torch.optim as optim

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Decoder.NNDecoder import SingleLableNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure


def main():
    # snr = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]
    snr = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5]

    for i in range(len(snr)):
        snr_dB = snr[i]
        print("snr_dB = ", snr_dB)
        encoder = hamming_encoder(device)

        # ML:
        bits_info = generator(nr_codeword, device)
        encoded_codeword = encoder(bits_info)
        modulated_signal = bpsk_modulator(encoded_codeword)
        noised_signal = AWGN(modulated_signal, snr_dB, device)

        practical_snr = NoiseMeasure(noised_signal, modulated_signal)

        # Create an instance of the SimpleNN class
        model = SingleLableNNDecoder(noised_signal.shape[2], hidden_size, 2*noised_signal.shape[2]).to(device)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            outputs = model(noised_signal)

            # Compute the loss
            loss = criterion(outputs, encoded_codeword)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {practical_snr}, Epoch [{epoch + 1}/{epochs}], BER: {loss.item()/100}')

        # # Testing the model
        # with torch.no_grad():
        #     test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        #     predicted_labels = model(test_data)
        #     predicted_labels = (predicted_labels > 0.5).float()  # Threshold the outputs
        #
        #     print("Predicted labels:")
        #     print(predicted_labels)

if __name__ == '__main__':
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")

    # Hyperparameters
    hidden_size = 7
    learning_rate = 0.01
    epochs = 10000
    nr_codeword = int(40)

    main()