import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Decode.NNDecoder import SingleLabelNNDecoder_nonfully
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import BinarytoDecimal
from earlystopping import SLNN_EarlyStopping

from generating import all_codebook
from Encode.Encoder import PCC_encoders
from Hamming74.reduce_mask import MaskMatrix



def SLNN_training(snr, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size, edge_delete,
                  model_load_pth, model_save_path, patience, delta, mask, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)

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

    # Define lists to store loss values
    SLNN_train_losses = []
    SLNN_test_losses = []

    # Early Stopping
    early_stopping = SLNN_EarlyStopping(patience, delta, snr_measure, edge_delete)

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
                print(f'SLNN Edge Deleted:{edge_delete}, SNR={snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
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

        print(f'SLNN Testing - Edge Deleted:{edge_delete} - SNR{snr_measure} - Loss: {running_loss / len(SLNN_testloader):.9f}')

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path):
            print('SLNN: Early stopping')
            print(f'SLNN Edge Deleted:{edge_delete}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(SLNN_testloader)} and epoch is {epoch}')
            break
        else:
            print("SLNN: Continue Training")


def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    SLNN_hidden_size = 7
    batch_size = 32
    learning_rate = 1e-2
    epochs = 500
    nr_codeword = int(1e6)
    bits = 4
    encoded = 7
    # edge_delete = [9, 14, 19, 24, 29, 34, 39, 40, 41, 42] # Edge delete
    edge_delete = [41, 42] # Edge delete
    masks = MaskMatrix(device)


    SLNN_snr = torch.tensor(0.0, dtype=torch.float, device=device) # SLNN training
    SLNN_snr = SLNN_snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))  # for SLNN article

    # Early Stopping
    SLNN_patience = 16
    delta = 0.001

    for i in range(len(edge_delete)):
        mask = masks(edge_delete[i], encoded, SLNN_hidden_size)
        # model Path:
        Load_path = f"Result/Model/SLNN_modified_neuron7_cpu_hidden.weight/SLNN_model_modified_hiddenlayer7_threshold{edge_delete[i]}_BER0.pth"
        SLNN_reduce_save_path = f"Result/Model/SLNN_decrease_hidden.weight_{device}/"

        # Train SLNN with different hidden layer neurons
        SLNN_training(SLNN_snr, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, SLNN_hidden_size, edge_delete[i],
                      Load_path, SLNN_reduce_save_path, SLNN_patience, delta, mask, device)


if __name__ == '__main__':
    main()