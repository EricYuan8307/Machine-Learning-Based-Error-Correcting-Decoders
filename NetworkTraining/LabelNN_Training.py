import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Decoder.NNDecoder import SingleLabelNNDecoder, MultiLabelNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure
from Estimation.BitErrorRate import calculate_ber
from Decoder.Converter import BinarytoDecimal, DecimaltoBinary, MLNN_decision

def SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device):

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
        model = SingleLabelNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(noised_signal).squeeze(1)

            # Compute the loss
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            end_time = time.time()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {snr_measure}, Epoch [{epoch + 1}/{epochs}], loss: {loss.item()}, time: {end_time - start_time}')

        tobinary = DecimaltoBinary(device)
        SLNN_final = tobinary(outputs)

        BER_SLNN, error_num_SLNN = calculate_ber(SLNN_final, bits_info)  # BER calculation
        print(f"SLNN: When SNR is {snr_measure} and signal number is {nr_codeword}, error number is {error_num_SLNN} and BER is {BER_SLNN}")


def MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device):

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
        output_size = bits_info.shape[2]

        # Create an instance of the SimpleNN class
        model = MultiLabelNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(noised_signal)

            # Compute the loss
            loss = criterion(outputs, bits_info)
            loss.backward()
            optimizer.step()

            end_time = time.time()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {snr_measure}, Epoch [{epoch + 1}/{epochs}], loss: {loss.item()}, time: {end_time - start_time}')

        MLNN_final = MLNN_decision(outputs, device)
        BER_MLNN, error_num_MLNN = calculate_ber(MLNN_final, bits_info)  # BER calculation
        print(f"MLNN: When SNR is {snr_measure} and signal number is {nr_codeword}, error number is {error_num_MLNN} and BER is {BER_MLNN}")



def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda:2")

    # Hyperparameters
    snr = torch.arange(0, 9.5, 0.5)
    hidden_size = 7
    batch_size = 32
    learning_rate = 1e-2
    epochs = 10000
    nr_codeword = int(1e2)

    # SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device)
    MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, device)


if __name__ == '__main__':
    main()