import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Decode.NNDecoder import SingleLabelNNDecoder1, MultiLabelNNDecoder2, MultiLabelNNDecoder3
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import BinarytoDecimal
from earlystopping import EarlyStopping

from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders

def SLNN_training1(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr, device)
    snr_measure = NoiseMeasure(noised_signal, modulated_signal, bits, encoded).to(torch.int)

    # NN structure:
    input_size = noised_signal.shape[2]
    output_size = torch.pow(torch.tensor(2), bits_info.shape[2]) # 2^x
    label = BinarytoDecimal(bits_info).to(torch.int64)
    SLNN_trainset = TensorDataset(noised_signal, label)
    SLNN_trainloader = torch.utils.data.DataLoader(SLNN_trainset, batch_size, shuffle=True)
    SLNN_testloader = torch.utils.data.DataLoader(SLNN_trainset, batch_size, shuffle=False)

    # Create an instance of the SimpleNN class
    model = SingleLabelNNDecoder1(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define lists to store loss values
    SLNN_train_losses = []
    SLNN_test_losses = []

    # Early Stopping
    early_stopping = EarlyStopping(patience, delta)

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
                print(f'{NN_type}:Hidden Size:{hidden_size}, SNR{snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
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

        print(f'{NN_type} Testing - Hidden Size:{hidden_size} - SNR{snr_measure} - Loss: {running_loss / len(SLNN_testloader):.9f}')

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def MLNN_training2(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr, device)
    snr_measure = NoiseMeasure(noised_signal, modulated_signal, bits, encoded).to(torch.int)

    # NN structure:
    input_size = noised_signal.shape[2]
    output_size = bits_info.shape[2]
    MLNN_trainset = TensorDataset(noised_signal, bits_info)
    MLNN_trainloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=True)
    MLNN_testloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=False)

    # Create an instance of the SimpleNN class
    model = MultiLabelNNDecoder2(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define lists to store loss values
    MLNN_train_losses = []
    MLNN_test_losses = []

    # Early Stopping
    early_stopping = EarlyStopping(patience, delta)

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
                print(f'{NN_type}: Hidden Size:{hidden_size}, SNR{snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
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

        print(f'{NN_type} Testing - SNR{snr_measure} - Loss: {running_loss/len(MLNN_testloader):.9f}')


        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(MLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def MLNN_training3(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr, device)
    snr_measure = NoiseMeasure(noised_signal, modulated_signal, bits, encoded).to(torch.int)

    # NN structure:
    input_size = noised_signal.shape[2]
    output_size = bits_info.shape[2]
    MLNN_trainset = TensorDataset(noised_signal, bits_info)
    MLNN_trainloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=True)
    MLNN_testloader = torch.utils.data.DataLoader(MLNN_trainset, batch_size, shuffle=False)

    # Create an instance of the SimpleNN class
    model = MultiLabelNNDecoder3(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Define lists to store loss values
    MLNN_train_losses = []
    MLNN_test_losses = []

    # Early Stopping
    early_stopping = EarlyStopping(patience, delta)

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
                print(f'{NN_type}: Hidden Size:{hidden_size}, SNR{snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
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

        print(f'{NN_type} Testing - SNR{snr_measure} - Loss: {running_loss/len(MLNN_testloader):.9f}')


        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(MLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    # Hyperparameters
    NN_type = "SLNN"
    hidden_num = 1
    hidden_size = [24, 25, 26, 27, 28]
    batch_size = 64
    learning_rate = 1e-2
    epochs = 500

    nr_codeword = int(1e7)
    bits = 10
    encoded = 26
    encoding_method = "Parity" # "Hamming", "Parity", "BCH"

    snr = torch.tensor(0.0, dtype=torch.float, device=device)
    snr = snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article


    # Early Stopping # Guess same number of your output
    SLNN_patience = torch.pow(torch.tensor(2), bits)
    MLNN_patience = bits
    delta = 0.001

    # model Path:
    model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/"
    model_name = f"{NN_type}_{hidden_size}"


    if NN_type == "SLNN" and hidden_num == 1:
        for i in range(len(hidden_size)):
            SLNN_training1(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size[i],
                          model_save_path, model_name, NN_type, SLNN_patience, delta, device)

    elif NN_type == "MLNN" and hidden_num == 2:
        # Train MLNN model with two hidden layer
        for i in range(len(hidden_size)):
            MLNN_training2(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size[i],
                           model_save_path, model_name, NN_type, MLNN_patience, delta, device)

    elif NN_type == "MLNN" and hidden_num == 3:
        # Train MLNN model with three hidden layers
        for i in range(len(hidden_size)):
            MLNN_training3(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size[i],
                           model_save_path, model_name, NN_type, MLNN_patience, delta, device)


if __name__ == '__main__':
    main()