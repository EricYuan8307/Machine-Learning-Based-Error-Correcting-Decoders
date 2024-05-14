import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encode.Generator import generator_ECCT
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Transformer.Model import ECC_Transformer
from Codebook.CodebookMatrix import ParitycheckMatrix
from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK
from earlystopping import EarlyStopping
from Decode.HardDecision import hard_decision

def ECCT_Training(snr, method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, model_save_path, model_name, NN_type, patience, delta, n_dec, n_head, d_model, dropout, device):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator_ECCT(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr, device)
    snr_measure = NoiseMeasure(noised_signal, modulated_signal, bits, encoded).to(torch.int)

    # Transformer:
    noised_signal = noised_signal.squeeze(1)
    # bits_info = bits_info.squeeze(1)
    encoded_codeword = encoded_codeword.squeeze(1)
    compare = noised_signal * modulated_signal
    compare = hard_decision(torch.sign(compare), device)

    H = ParitycheckMatrix(encoded, bits, method, device).squeeze(0).T
    ECCT_trainset = TensorDataset(noised_signal, compare)
    ECCT_trainloader = torch.utils.data.DataLoader(ECCT_trainset, batch_size, shuffle=True)
    ECCT_testloader = torch.utils.data.DataLoader(ECCT_trainset, batch_size, shuffle=False)

    # ECCT model:
    model = ECC_Transformer(n_head, d_model, encoded, H, n_dec, dropout, device).to(device)

    # Define the loss function and optimizer
    criterion = F.binary_cross_entropy_with_logits
    optimizer = optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Define lists to store loss values
    ECCT_train_losses = []
    ECCT_test_losses = []

    # Early Stopping
    early_stopping = EarlyStopping(patience, delta)

    # ECCT Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(ECCT_trainloader, 0):
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
                print(f'{NN_type}: Head Num:{n_head}, Encoding Dim:{d_model}, SNR{snr_measure}, Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 1000:.9f}')
                running_loss = 0.0

        # Calculate the average training loss for this epoch
        avg_train_loss = running_loss / len(ECCT_trainloader)
        ECCT_train_losses.append(avg_train_loss)

        # Testing loop
        running_loss = 0.0

        with torch.no_grad():
            for data in ECCT_testloader:
                inputs, labels = data

                # Forward pass
                outputs = model(inputs)

                # Compute the loss
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        # Calculate the average testing loss for this epoch
        avg_test_loss = running_loss / len(ECCT_testloader)
        ECCT_test_losses.append(avg_test_loss)

        print(f'{NN_type} Testing - {NN_type}: Head Num:{n_head}, Encoding Dim:{d_model}, SNR{snr_measure} - Loss: {running_loss/len(ECCT_testloader):.9f}')

        scheduler.step(avg_test_loss)

        # Early Stopping
        if early_stopping(running_loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Head Num:{n_head}, Encoding Dim:{d_model}, SNR={snr_measure} Stop at total val_loss is {running_loss/len(ECCT_testloader)} and epoch is {epoch}')
            break
        else:
            print(f"{NN_type}: Continue Training")

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    set_seed(42)

    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    NN_type = "ECCT"
    nr_codeword = int(1e6)
    bits = 51
    encoded = 63
    encoding_method = "BCH" # "Hamming", "Parity", "BCH",

    n_decoder = 6 # decoder iteration times
    n_head = 8 # head number
    dropout = 0 # dropout rate
    d_model = 128 # input embedding dimension

    epochs = 1000
    learning_rate = 0.001
    batch_size = 128
    patience = 10
    delta = 0.001

    model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/"
    model_name = f"{NN_type}_h{n_head}_d{d_model}"

    snr = torch.tensor(0.0, dtype=torch.float, device=device) # for EsN0 (dB)
    snr = snr + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for EbN0 (dB)

    ECCT_Training(snr, encoding_method, nr_codeword, bits, encoded, epochs, learning_rate, batch_size, model_save_path,
                  model_name, NN_type, patience, delta, n_decoder, n_head, d_model, dropout, device)

if __name__ == "__main__":
    main()