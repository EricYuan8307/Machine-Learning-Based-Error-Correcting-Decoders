import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Decoder.NNDecoder import SingleLableNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure
from Estimation.BitErrorRate import calculate_ber
from Decoder.Converter import BinarytoDecimal, DecimaltoBinary

def training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device):

    for i in range(len(snr)):
        snr_dB = snr[i]

        # Transmitter:
        encoder = hamming_encoder(device)
        bits_info = generator(nr_codeword, device)
        encoded_codeword = encoder(bits_info)
        modulated_signal = bpsk_modulator(encoded_codeword)
        noised_signal = AWGN(modulated_signal, snr_dB, device)
        snr_measure = NoiseMeasure(noised_signal, modulated_signal)

        # NN structure:
        input_size = noised_signal.shape[2]
        output_size = torch.pow(torch.tensor(2, device=device), bits_info.shape[2]) # 2^x
        label = BinarytoDecimal(bits_info,device).to(torch.long)

        # Create an instance of the SimpleNN class
        model = SingleLableNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noised_signal).squeeze(1)

            # Compute the loss
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {snr_measure}, Epoch [{epoch + 1}/{epochs}], loss: {loss.item()}')

        tobinary = DecimaltoBinary(device)
        SLNN_final = tobinary(outputs)

        BER_SLNN, error_num_SLNN = calculate_ber(SLNN_final, bits_info)  # BER calculation
        print(f"SLNN: When SNR is {snr_measure} and signal number is {nr_codeword}, error number is {error_num_SLNN} and BER is {BER_SLNN}")






def main():
    snr = torch.arange(4, 9.5, 0.5)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))
    # device = torch.device("cpu")

    # Hyperparameters
    hidden_size = 7
    batch_size = 32
    learning_rate = 1e-2
    epochs = 10000
    nr_codeword = int(1e6)

    training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device)

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




if __name__ == '__main__':
    main()