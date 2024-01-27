import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from Encoder.Generator import generator
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Transmit.noise import AWGN
from Decoder.NNDecoder import SingleLabelNNDecoder, MultiLabelNNDecoder
from Transmit.NoiseMeasure import NoiseMeasure
from Decoder.Converter import BinarytoDecimal
from earlystopping import SLNN_EarlyStopping, MLNN_EarlyStopping

def SLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, model_path, patience, delta, device):

    for i in range(len(snr)):
        snr_save = i/2
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
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Define lists to store loss values
        SLNN_train_losses = []
        SLNN_test_losses = []

        # Early Stopping
        early_stopping = SLNN_EarlyStopping(patience, delta, snr_save)

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
                if i % 1000 == 999:  # Print every 100 mini-batches
                    print(f'SLNN: SNR{snr_save}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
                    running_loss = 0.0

            # Calculate the average training loss for this epoch
            avg_train_loss = running_loss / len(SLNN_trainloader)
            SLNN_train_losses.append(avg_train_loss)

            # Testing loop
            running_loss = 0.0

            with torch.no_grad():
                for data in SLNN_testloader:
                    inputs, labels = data

                    # Forward pass
                    outputs = model(inputs).squeeze(1)

                    # Compute the loss
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            # Calculate the average testing loss for this epoch
            avg_test_loss = running_loss / len(SLNN_testloader)
            SLNN_test_losses.append(avg_test_loss)

            print(f'SLNN Testing - SNR{snr_save} - Loss: {running_loss / len(SLNN_testloader):.9f}')

            # Early Stopping
            if early_stopping(running_loss, model, model_path):
                print('SLNN: Early stopping')
                print(f'SLNN: SNR={snr_save} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
                break
            else:
                print(f"SLNN: Continue Training")


def MLNN_training(snr, nr_codeword, epochs, learning_rate, batch_size, hidden_size, model_path, patience, delta, device):

    for i in range(len(snr)):
        snr_save = i/2
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
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Define lists to store loss values
        MLNN_train_losses = []
        MLNN_test_losses = []

        # Early Stopping
        early_stopping = MLNN_EarlyStopping(patience, delta, snr_save)

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
                if i % 1000 == 999:  # Print every 100 mini-batches
                    print(f'MLNN: SNR{snr_save}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
                    running_loss = 0.0

            # Calculate the average training loss for this epoch
            avg_train_loss = running_loss / len(MLNN_trainloader)
            MLNN_train_losses.append(avg_train_loss)

            # Testing loop
            running_loss = 0.0

            with torch.no_grad():
                for data in MLNN_testloader:
                    inputs, labels = data

                    # Forward pass
                    outputs = model(inputs)

                    # Compute the loss
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            # Calculate the average testing loss for this epoch
            avg_test_loss = running_loss / len(MLNN_testloader)
            MLNN_test_losses.append(avg_test_loss)

            print(f'MLNN Testing - SNR{snr_save} - Loss: {running_loss/len(MLNN_testloader):.9f}')


            # Early Stopping
            if early_stopping(running_loss, model, model_path):
                print('MLNN: Early stopping')
                print(f'MLNN: SNR={snr_save} Stop at total val_loss is {running_loss/len(MLNN_testloader)} and epoch is {epoch}')
                break
            else:
                print("MLNN: Continue Training")



def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    # Hyperparameters
    SLNN_snr = torch.arange(0, 8.5, 0.5)
    SLNN_snr = SLNN_snr + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float)) # for SLNN article
    MLNN_snr = torch.arange(0, 8, 0.5)
    MLNN_snr = MLNN_snr + 10 * torch.log10(torch.tensor(4 / 7, dtype=torch.float)) # for MLNN article

    SLNN_hidden_size = 7
    MLNN_hidden_size = 100
    batch_size = 64
    learning_rate = 1e-2
    epochs = 500
    nr_codeword = int(1e6)

    # Early Stopping # Guess same number of your output
    SLNN_patience = 16
    MLNN_patience = 4
    delta = 0.001

    # Save model
    current_time = datetime.now().strftime("%m-%d_%H-%M-%S")
    SLNN_model_path = f"Result/Model/SLNN_{current_time}/"
    MLNN_model_path = f"Result/Model/MLNN_{current_time}/"

    SLNN_training(SLNN_snr, nr_codeword, epochs, learning_rate, batch_size, SLNN_hidden_size, SLNN_model_path, SLNN_patience, delta, device)
    # MLNN_training(MLNN_snr, nr_codeword, epochs, learning_rate, batch_size, MLNN_hidden_size, MLNN_model_path, MLNN_patience, delta, device)




if __name__ == '__main__':
    main()