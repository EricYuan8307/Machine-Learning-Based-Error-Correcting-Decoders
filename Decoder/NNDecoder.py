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
        x = self.softmax(x)
        # x = torch.argmax(x, dim=2).unsqueeze(1).to(torch.float).requires_grad_(True) # torch.Size([5, 1, 7])

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

def training(snr, nr_codeword, epochs, learning_rate, hidden_size, device):

    for i in range(len(snr)):
        snr_dB = snr[i]

        # Encoder:
        encoder = hamming_encoder(device)

        bits_info = generator(nr_codeword, device)
        encoded_codeword = encoder(bits_info)
        modulated_signal = bpsk_modulator(encoded_codeword)
        noised_signal = AWGN(modulated_signal, snr_dB, device)

        practical_snr = NoiseMeasure(noised_signal, modulated_signal)

        # NN structure:
        input_size = noised_signal.shape[2]
        output_size = torch.pow(torch.tensor(2, device=device), bits_info.shape[2]) # 2^x
        label = BinarytoDecimal(bits_info)

        # Create an instance of the SimpleNN class
        model = SingleLableNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            print('inputs: ', noised_signal.shape) # torch.Size([1000, 1, 7])
            print('labels: ', label.shape) # torch.Size([1000])
            # Forward pass
            outputs = model(noised_signal)
            print('outputs: ', outputs.shape) # torch.Size([1000, 1, 16])

            # Compute the loss
            loss = criterion(outputs, label)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {practical_snr}, Epoch [{epoch + 1}/{epochs}], BER: {1 - loss.item()/100}')

    #     # Testing the model
    #     with torch.no_grad():
    #         test_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    #         predicted_labels = model(test_data)
    #         predicted_labels = (predicted_labels > 0.5).float()  # Threshold the outputs
    #
    #         print("Predicted labels:")
    #         print(predicted_labels)
    #
    # def img_loss(train_losses, test_losses, file_name):
    #     num_epochs = len(train_losses)
    #     place = "images/" + file_name
    #     # plot the loss
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.plot(train_losses, label='Training loss')
    #     ax.plot(test_losses, label='Testing loss')
    #     ax.set_xlim(0, num_epochs - 1)
    #
    #     # axis labels
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.savefig(place)
    #     plt.show()

def BinarytoDecimal(binary_tensor):
    decimal_values = torch.sum(binary_tensor * (2 ** torch.arange(binary_tensor.shape[-1], dtype=torch.float)), dim=-1)
    decimal_values = decimal_values.squeeze()

    return decimal_values

def main():
    snr = torch.arange(0, 9.5, 0.5)
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")

    # Hyperparameters
    hidden_size = 7
    learning_rate = 1e-4
    epochs = 1
    nr_codeword = int(1e3)

    training(snr, nr_codeword, epochs, learning_rate, hidden_size, device)



if __name__ == '__main__':
    main()