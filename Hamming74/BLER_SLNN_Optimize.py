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
from earlystopping import EarlyStopping

from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders


def SLNN_training(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, hidden_size, edge_delete,
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


def Mask40(order, device):
    if order == 1:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 2:
        mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 3:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 4:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 5:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 6:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 7:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 8:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 9:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 10:
        mask = torch.tensor([[0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 1, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [1, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask

def Mask42(device):
    mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask

def Mask43(order, device):
    if order == 1:
        mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 2:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 3:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 4:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 5:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 6:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    if order == 7:
        mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1],
                             [0, 1, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device)

    return mask


def main():
    # Device Setting
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
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
    encoding_method = "Hamming"
    edge_delete = 43
    order = torch.arange(1, 113, 1).to(torch.int)

    snr = 0

    SLNN_snr = torch.tensor(snr, dtype=torch.float, device=device) # SLNN training
    SLNN_snr = SLNN_snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))  # for SLNN article

    # Early Stopping
    SLNN_patience = 16
    delta = 0.001

    for j in range(len(order)):
        mask = Mask42(device)
        Load_path = f"Result/Model/SLNN_edgedeleted{edge_delete}_output.weight_cpu/SLNN7_edgedeleted{edge_delete}_order{order[j]}.pth"
        model_save_path = f"Result/Model/SLNN_edgedeleted{edge_delete}_trained_output.weight_{device}_BER{snr}/"
        model_name = f"SLNN_edgedeleted{edge_delete}_order{order[j]}"

        # Train SLNN with different hidden layer neurons
        SLNN_training(SLNN_snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, SLNN_hidden_size, edge_delete,
                      Load_path, model_save_path, model_name, SLNN_patience, delta, mask, order[j], device)


if __name__ == '__main__':
    main()