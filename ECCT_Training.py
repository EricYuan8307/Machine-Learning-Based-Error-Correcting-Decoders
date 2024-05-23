import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Encode.Generator import generator_ECCT
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN_ECCT
from Transformer.Model import ECC_Transformer
from Codebook.CodebookMatrix import ParitycheckMatrix
from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders
from earlystopping import EarlyStopping
from Metric.ErrorRate import calculate_ber, calculate_bler

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def BLER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

def train(model, train_loader, optimizer, epoch, LR, device):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    for batch_idx, (m, x, y, magnitude, syndrome) in enumerate(train_loader):
        z_mul = (y * bpsk_modulator(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        BLER = calculate_bler(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += BLER * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: learning rate={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples

def ECCT_Training(model, std, method, nr_codeword, bits, encoded, learning_rate, batch_size, epoch, device):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)
    H = ParitycheckMatrix(encoded, bits, method, device).squeeze(0).T

    # ECCT Input:
    bits_info, encoded_codeword, noised_signal, magnitude, syndrome = prepossing(std, H, encoder_matrix, nr_codeword, bits, device)

    # Transformer:
    ECCT_trainset = TensorDataset(bits_info, encoded_codeword, noised_signal, magnitude, syndrome)
    ECCT_trainloader = torch.utils.data.DataLoader(ECCT_trainset, batch_size, shuffle=True)

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), learning_rate)

    model.train()
    cum_loss = cum_ber = cum_bler = cum_samples = 0
    for batch_idx, (m, x, y, magnitude, syndrome) in enumerate(ECCT_trainloader):
        z_mul = (y * bpsk_modulator(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        ber, _ = calculate_ber(x_pred, x)
        bler, _ = calculate_bler(x_pred, x)

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_bler += bler * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx + 1) % 100 == 0 or batch_idx == len(ECCT_trainloader) - 1:
            print(
                f'Epoch:{epoch} Batch {batch_idx + 1}/{len(ECCT_trainloader)}: learning rate={learning_rate:.2e}, '
                f'Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} BLER={cum_bler / cum_samples:.2e}')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_bler / cum_samples

def ECCT_testing(model, std_range, method, nr_codeword, bits, encoded, batch_size, device, min_bler):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)
    H = ParitycheckMatrix(encoded, bits, method, device).squeeze(0).T
    ECCT_testloader_list = []

    # Transformer:
    for i in range(len(std_range)):
        # ECCT Input:
        bits_info, encoded_codeword, noised_signal, magnitude, syndrome = prepossing(std_range[i], H, encoder_matrix,
                                                                                     nr_codeword, bits, device)
        ECCT_testset = TensorDataset(bits_info, encoded_codeword, noised_signal, magnitude, syndrome)
        ECCT_testloader = torch.utils.data.DataLoader(ECCT_testset, batch_size, shuffle=False)
        ECCT_testloader_list.append(ECCT_testloader)

    # Define the loss function and optimizer
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_bler_list, cum_samples_all = [], [], [], []
    with torch.no_grad():
        for ii, test_loader in enumerate(ECCT_testloader_list):
            test_loss = test_ber = test_bler = cum_count = 0.
            while True:
                (m, x, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bpsk_modulator(x))
                z_pred = model(magnitude, syndrome)
                loss, x_pred = model.loss(-z_pred, z_mul, y)

                test_loss += loss.item() * x.shape[0]

                ber, _ = calculate_ber(x_pred, x)
                bler, _ = calculate_bler(x_pred, x)

                test_ber += ber * x.shape[0]
                test_bler += bler * x.shape[0]
                cum_count += x.shape[0]
                # if (min_bler > 0 and test_bler > min_bler and cum_count > 1e5) or cum_count >= 1e9:
                #     if cum_count >= 1e9:
                #         print(f'Number of samples threshold reached for EbN0:{std_range[ii]}')
                #     else:
                #         print(f'BLER count threshold reached for EbN0:{std_range[ii]}')
                #     break
            # cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_bler_list.append(test_bler / cum_count)
            print(f'BER={test_loss_ber_list[-1]:.2e}')
        ###
        print('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno) in (zip(test_loss_list, std_range))]))
        print('Test BLER ' + '{}: {:.2e}'.format(ebno, elem) for (elem, ebno) in
              (zip(test_loss_bler_list, std_range)))
        print('Test BER ' + '{}: {:.2e}'.format(ebno, elem) for (elem, ebno) in
              (zip(test_loss_ber_list, std_range)))
    return test_loss_list, test_loss_ber_list, test_loss_bler_list

def sign_to_bin(x):
    return 0.5 * (1 - x) # 0.5*(x+1)

def prepossing(std, H, encoder_matrix, nr_codeword, bits, device):
    # ECCT Input:
    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator_ECCT(nr_codeword, bits, device)  # m
    encoded_codeword = encoder(bits_info)  # x
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN_ECCT(modulated_signal, std, device)  # y
    magnitude = torch.abs(noised_signal)
    syndrome = torch.matmul(sign_to_bin(torch.sign(noised_signal)), H.T) % 2
    syndrome = bpsk_modulator(syndrome)

    return bits_info, encoded_codeword, noised_signal, magnitude, syndrome

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * torch.log10(2 * rate)
    return torch.sqrt(1. / (10. ** (snr / 10.)))

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
    nr_codeword = int(1e5)
    bits = 51
    encoded = 63
    encoding_method = "BCH" # "Hamming", "Parity", "BCH",

    n_decoder = 6 # decoder iteration times
    dropout = 0 # dropout rate
    n_head = 8  # head number
    d_model = 128 # input embedding dimension
    epochs = 1000
    learning_rate = 0.001
    batch_size = 128
    test_batch_size = 1024
    patience = 10
    delta = 0.001
    H = ParitycheckMatrix(encoded, bits, encoding_method, device).squeeze(0).T

    model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/"
    model_name = f"{NN_type}_h{n_head}_d{d_model}"

    snr = torch.tensor(8.0, dtype=torch.float, device=device) # for EsN0 (dB)
    snr = torch.sqrt(1. / (10. ** (snr / 10.)))

    test_std_range = torch.arange(4, 7, 1)
    test_std_range = torch.sqrt(1. / (10. ** (test_std_range / 10.)))

    early_stopping = EarlyStopping(patience, delta)

    model = ECC_Transformer(n_head, d_model, encoded, H, n_decoder, dropout, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)


    for epoch in range(1, epochs + 1):
        loss, ber, bler = ECCT_Training(model, snr, encoding_method, nr_codeword, bits, encoded, learning_rate, batch_size, epoch, device)
        scheduler.step()
        if early_stopping(loss, model, model_save_path, model_name):
            print(f'{NN_type}: Early stopping')
            print(f'{NN_type}: Stop at loss is {loss} and epoch is {epoch}, ber:{ber}, ber:{bler}')
            break
        else:
            print(f"{NN_type}: Continue Training")
        if epoch % 300 == 0 or epoch in [1, epochs]:
            test_loss_list, test_ber_list, test_bler_list = ECCT_testing(model, test_std_range, encoding_method,
                                                                         nr_codeword, bits, encoded, test_batch_size, device, min_bler=100)


if __name__ == "__main__":
    main()