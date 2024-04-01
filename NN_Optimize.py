import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Decode.NNDecoder import SingleLabelNNDecoder_nonfully
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import BinarytoDecimal
from earlystopping import EarlyStopping

from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders


def SLNN_training(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, edge_delete,
                  model_load_pth, model_save_path, model_name, patience, delta, mask, order, device):
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
    model = SingleLabelNNDecoder_nonfully(input_size, hidden_size, output_size, mask).to(device)
    model.load_state_dict(torch.load(model_load_pth))

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
                print(f'SLNN Edge Deleted:{edge_delete}, order:{order}, SNR={snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
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

        print(f'SLNN Testing - Edge Deleted:{edge_delete} - order:{order}, - SNR{snr_measure} - Loss: {running_loss / len(SLNN_testloader):.9f}')

        scheduler.step(avg_test_loss)

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print('SLNN: Early stopping')
            print(f'SLNN Edge Deleted:{edge_delete}, order:{order}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print("SLNN: Continue Training")


def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    hidden_size = 24
    batch_size = 64
    learning_rate = 1e-2
    epochs = 500
    nr_codeword = int(1e6)
    bits = 10
    encoded = 26
    encoding_method = "Parity"
    edge_delete_range = [642] # 449, 470, 490, 501, 530, 540, 551, 561, 571, 581, 593, 601, 610, 611, 612, 613, 614, 617, 618, 619
    # order = torch.arange(1, 113, 1).to(torch.int)
    order = None
    NN_type = "SLNN"
    momentum = 0.9

    SLNN_snr = torch.tensor(0, dtype=torch.float, device=device) # SLNN training
    SLNN_snr = SLNN_snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))  # for SLNN article

    # Early Stopping
    patience = encoded
    delta = 0.001

    for edge_delete in edge_delete_range:
        # model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{hidden_size}_deleted_{device}/{NN_type}_deleted{edge_delete}.pth"
        # if not os.path.exists(model_pth):
        #     continue

        Load_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{hidden_size}_deleted_{device}/{NN_type}_deleted{edge_delete}.pth"
        model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{hidden_size}_ft_{device}/"
        model_name = f"SLNN_deleted{edge_delete}_trained"

        # Train SLNN with different hidden layer neurons
        model = torch.load(Load_path)
        mask = (model['hidden.weight'] != 0).int()

        SLNN_training(SLNN_snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, momentum, batch_size, hidden_size, edge_delete,
                      Load_path, model_save_path, model_name, patience, delta, mask, order, device)

    # for j in range(len(order)):
    #     mask =
    #     Load_path = f"Result/Model/SLNN_edgedeleted{edge_delete}_output.weight_cpu/SLNN7_edgedeleted{edge_delete}_order{order[j]}.pth"
    #     model_save_path = f"Result/Model/SLNN_edgedeleted{edge_delete}_trained_output.weight_{device}_BER{snr}/"
    #     model_name = f"SLNN_edgedeleted{edge_delete}_order{order[j]}"
    #
    #     # Train SLNN with different hidden layer neurons
    #     SLNN_training(SLNN_snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, SLNN_hidden_size, edge_delete,
    #                   Load_path, model_save_path, model_name, SLNN_patience, delta, mask, order[j], device)


if __name__ == '__main__':
    main()