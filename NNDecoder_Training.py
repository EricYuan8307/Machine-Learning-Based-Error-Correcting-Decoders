import os
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Decoder.NNDecoder import SingleLabelNNDecoder, MultiLabelNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure
from Estimation.BitErrorRate import calculate_ber
from Decoder.Converter import BinarytoDecimal, DecimaltoBinary, MLNN_decision

def SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, directory_path, result, device):

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
        trainset = TensorDataset(noised_signal, label)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)

        # Create an instance of the SimpleNN class
        model = SingleLabelNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            start_time = time.time()
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                # Forward pass
                outputs = model(inputs).squeeze(1)
                # output_test = torch.argmax(outputs, dim=1)

                # Compute the loss
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            end_time = time.time()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {snr_measure}, Epoch [{epoch + 1}/{epochs}], loss: {loss.item()}, time: {end_time - start_time}')

        tobinary = DecimaltoBinary(device)
        SLNN_final = tobinary(outputs)

        BER_SLNN, error_num_SLNN = calculate_ber(SLNN_final, bits_info)  # BER calculation
        print(f"SLNN: When SNR is {snr_measure} and signal number is {nr_codeword}, error number is {error_num_SLNN} and BER is {BER_SLNN}")
        result[0, i] = BER_SLNN

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Get the current timestamp as a string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename with the timestamp
    csv_filename = f"SLNN_BER_result_{current_time}.csv"

    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result, delimiter=' ,')



def MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, directory_path, result, device):

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
            # Forward pass
            outputs = model(noised_signal)

            # Compute the loss
            optimizer.zero_grad()
            loss = criterion(outputs, bits_info)
            loss.backward()
            optimizer.step()

            end_time = time.time()

            if (epoch + 1) % 1000 == 0:
                print(f'When SNR is {snr_measure}, Epoch [{epoch + 1}/{epochs}], loss: {loss.item()}, time: {end_time - start_time}')

        MLNN_final = MLNN_decision(outputs, device)
        BER_MLNN, error_num_MLNN = calculate_ber(MLNN_final, bits_info)  # BER calculation
        print(f"MLNN: When SNR is {snr_measure} and signal number is {nr_codeword}, error number is {error_num_MLNN} and BER is {BER_MLNN}")
        result[0, i] = BER_MLNN

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Get the current timestamp as a string
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename with the timestamp
    csv_filename = f"MLNN_BER_result_{current_time}.csv"

    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result, delimiter=' ,')



def main():
    # Device Setting
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))
    # device = torch.device("cpu")
    # device = torch.device("cuda:2")

    # Hyperparameters
    snr = torch.arange(4, 6.5, 0.5)
    SLNN_hidden_size = 7
    MLNN_hidden_size = 100
    batch_size = 64
    learning_rate = 1e-2
    epochs = 10000
    nr_codeword = int(1e7)
    # 如果是在主目录子文件夹下，就需要使用absloyte path, 当BER_estimate在主目录中，所以Reference address就行。
    SLNN_directory_path = "/Result/Model/SLNN"
    MLNN_directory_path = "/Result/Model/MLNN"

    result_save = np.zeros((1, len(snr)))

    SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, SLNN_hidden_size, SLNN_directory_path, result_save, device)
    # MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, MLNN_hidden_size, MLNN_directory_path,result_save, device)


if __name__ == '__main__':
    main()