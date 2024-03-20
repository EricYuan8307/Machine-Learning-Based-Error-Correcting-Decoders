import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Decode.NNDecoder import SingleLabelNNDecoder1, SingleLabelNNDecoder2, MultiLabelNNDecoder2, MultiLabelNNDecoder3
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import BinarytoDecimal
from earlystopping import EarlyStopping

from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders

def SLNN_training1(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
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
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

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

        scheduler.step(avg_test_loss)

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def SLNN_training2(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
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
    model = SingleLabelNNDecoder2(input_size, hidden_size, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

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

        scheduler.step(avg_test_loss)

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def MLNN_training2(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
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
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

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

        scheduler.step(avg_test_loss)

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(MLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def MLNN_training3(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, model_save_path, model_name, NN_type, patience, delta, device):
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
    optimizer = optim.SGD(model.parameters(), learning_rate, momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

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

        scheduler.step(avg_test_loss)

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
    NeuralNetwork_type = ["SLNN"] # ["SLNN", "MLNN"]
    SLNN_hidden_size1 = [24, 25, 26, 27, 28]
    SLNN_hidden_size2 = [[25, 25], [100, 20], [20, 100], [100, 25], [25, 100]]
    MLNN_hidden_size = [[1000, 500], [2000, 1000], [2000, 1000, 500]]
    batch_size = 64
    learning_rate = 1e-2
    momentum = 0.9
    epochs = 500

    nr_codeword = int(1e6)
    bits = 10
    encoded = 26
    encoding_method = "Parity" # "Hamming", "Parity", "BCH"

    snr = torch.tensor(0.0, dtype=torch.float, device=device)
    snr = snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    # Early Stopping # Guess same number of your output
    patience = encoded
    # SLNN_patience = torch.pow(torch.tensor(2), bits)
    # MLNN_patience = bits
    delta = 0.001

    for NN_type in NeuralNetwork_type:
        # model Path:
        model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/"

        if NN_type == "SLNN" :
            for i in range(len(SLNN_hidden_size1)):
                model_name = f"{NN_type}_hiddenlayer{SLNN_hidden_size1[i]}"
                SLNN_training1(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, SLNN_hidden_size1[i],
                              model_save_path, model_name, NN_type, patience, delta, device)

            for j in range(len(SLNN_hidden_size2)):
                model_name = f"{NN_type}_hiddenlayer{SLNN_hidden_size2[j]}"
                SLNN_training2(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, SLNN_hidden_size2[j],
                              model_save_path, model_name, NN_type, patience, delta, device)

        elif NN_type == "MLNN":
            # Train MLNN model with two hidden layer
            for k in range(len(MLNN_hidden_size)):
                if len(MLNN_hidden_size[k]) == 2:
                    model_name = f"{NN_type}_hiddenlayer{MLNN_hidden_size[k]}"
                    MLNN_training2(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, MLNN_hidden_size[k],
                                   model_save_path, model_name, NN_type, patience, delta, device)

                elif len(MLNN_hidden_size[k]) == 3:
                    model_name = f"{NN_type}_hiddenlayer{MLNN_hidden_size[k]}"
                    MLNN_training3(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate,momentum, batch_size, MLNN_hidden_size[k],
                                   model_save_path, model_name, NN_type, patience, delta, device)


if __name__ == '__main__':
    main()