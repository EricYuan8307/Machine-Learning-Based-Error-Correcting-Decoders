import os
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
from Decoder.Converter import BinarytoDecimal
from earlystopping import EarlyStopping

def SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, model_path, patience, delta, device):

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
        label = BinarytoDecimal(bits_info, device).to(torch.int64)
        SLNN_trainset = TensorDataset(noised_signal, label)
        SLNN_trainloader = torch.utils.data.DataLoader(SLNN_trainset, batch_size, shuffle=True)
        SLNN_testloader = torch.utils.data.DataLoader(SLNN_trainset, batch_size, shuffle=False)

        # Create an instance of the SimpleNN class
        model = SingleLabelNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Early Stopping
        early_stopping = EarlyStopping(patience, delta, snr_dB)

        # Single-Label Neural Network Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(SLNN_trainloader, 0):
                inputs, labels = data

                # Forward pass
                outputs = model(inputs).squeeze(1)

                # Compute the loss
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'SLNN: SNR{snr_dB}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

            # Testing loop
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for data in SLNN_testloader:
                    inputs, labels = data

                    # Forward pass
                    outputs = model(inputs).squeeze(1)

                    # Compute the loss
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                    # Calculate accuracy
                    predicted = torch.argmax(outputs, dim=1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

            print(f'SLNN Testing - SNR{snr_dB} - Loss: {running_loss / len(SLNN_testloader):.3f}')

            # Early Stopping
            if early_stopping(running_loss, model, model_path):
                print('SLNN: Early stopping')
                print(f'SLNN: Stop at total val_loss is {running_loss}')
            else:
                print("SLNN: Continue Training")



def MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, model_path, patience, delta, device):

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
        MLNN_trainset = TensorDataset(noised_signal, bits_info)
        MLNN_trainloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=True)
        MLNN_testloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=False)

        # Create an instance of the SimpleNN class
        model = MultiLabelNNDecoder(input_size, hidden_size, output_size).to(device)

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        # Early Stopping
        early_stopping = EarlyStopping(patience, delta, snr_dB)

        # Multi-Label Neural Network Training loop
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(MLNN_trainloader, 0):
                inputs, labels = data

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'MLNN: SNR{snr_dB}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

            # Testing loop
            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            with torch.no_grad():
                for data in MLNN_testloader:
                    inputs, labels = data

                    # Forward pass
                    outputs = model(inputs)

                    # Compute the loss
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

                    # Calculate accuracy
                    total_correct += (outputs == labels).sum().item()
                    total_samples += labels.size(0)

            print(f'MLNN Testing - SNR{snr_dB} - Loss: {running_loss / len(MLNN_testloader):.3f}')

            # Early Stopping
            if early_stopping(running_loss, model, model_path):
                print('MLNN: Early stopping')
                print(f'MLNN: Stop at total val_loss is {running_loss}')
            else:
                print("MLNN: Continue Training")

        # # Save MLNN model with specific SNR and time
        # os.makedirs(model_path, exist_ok=True)
        # torch.save(model.state_dict(), f"{model_path}MLNN_model_BER{snr_dB}.pth")



def main():
    # # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    # Hyperparameters
    SLNN_snr = torch.arange(0.0, 6.5, 0.5)
    MLNN_snr = torch.arange(0.0, 6.5, 0.5)
    SLNN_hidden_size = 7
    MLNN_hidden_size = 10
    batch_size = 64
    learning_rate = 1e-2
    epochs = 250
    nr_codeword = int(1e8)
    patience = 4
    delta = 0.001

    # Save model
    SLNN_model_path = "Result/Model/SLNN/"
    MLNN_model_path = "Result/Model/MLNN/"

    SLNN_training(SLNN_snr, nr_codeword, epochs, learning_rate, batch_size, SLNN_hidden_size, SLNN_model_path, patience, delta, device)
    MLNN_training(MLNN_snr, nr_codeword, epochs, learning_rate, batch_size, MLNN_hidden_size, MLNN_model_path, patience, delta, device)


if __name__ == '__main__':
    main()