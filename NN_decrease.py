import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from generating import all_codebook_NonML
from Encode.Generator import generator
from Encode.Encoder import PCC_encoders
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Decode.NNDecoder import SingleLabelNNDecoder_nonfully
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import BinarytoDecimal
from earlystopping import EarlyStopping



def SLNN_training1(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size,
                   model_load_pth, model_save_path, model_name, NN_type, patience, delta, mask, device):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)

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
    model = SingleLabelNNDecoder_nonfully(input_size, hidden_size, output_size, mask).to(device)
    model.load_state_dict(torch.load(model_load_pth))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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
            print(
                f'{NN_type}: Hidden Size:{hidden_size}, SNR={snr_measure} Stop at total val_loss is {running_loss / len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")


def main():
    # Device Setting
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.cuda.is_available()
                    else torch.device("cpu")))
    # device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    nr_codeword = int(1e6)
    bits = 10
    encoded = 26
    encoding_method = "Parity"  # "Hamming", "Parity", "BCH",
    NeuralNetwork_type = "SLNN"  # ["SLNN", "MLNN"]

    SLNN_hidden_size1 = 24
    # SLNN_hidden_size2 = [[25, 25], [100, 20], [20, 100], [100, 25], [25, 100]]
    MLNN_hidden_size = [[1000, 500], [2000, 1000], [2000, 1000, 500]]
    batch_size = 64
    learning_rate = 1e-2
    momentum = 0.9
    epochs = 500

    snr = torch.tensor(0.0, dtype=torch.float, device=device)
    snr = snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))  # for SLNN article

    # Early Stopping # Guess same number of your output
    patience = encoded
    delta = 0.001

    edge_deletes = [551] # Edge delete
    orders = torch.arange(0, 74, 1)
    # orders = [0]

    for edge_delete in edge_deletes:
        for order in orders:
            # model Path:
            model_load_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{SLNN_hidden_size1}_ft_{device}/{NeuralNetwork_type}_deleted{edge_delete}_order{order}.pth"
            model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{SLNN_hidden_size1}_ft_{device}/"

            # calculate the mask
            model = torch.load(model_load_path)
            mask = (model['hidden.weight'] != 0).int()

            # Train SLNN with different hidden layer neurons
            model_name = f"{NeuralNetwork_type}_deleted{edge_delete}_order{order}_trained"
            SLNN_training1(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, SLNN_hidden_size1,
                          model_load_path, model_save_path, model_name, NeuralNetwork_type, patience, delta, mask, device)


if __name__ == '__main__':
    main()