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
from Metric.ErrorRate import calculate_ber, calculate_bler


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * torch.log10(2 * rate)
    return torch.sqrt(1. / (10. ** (snr / 10.)))

def sign_to_bin(x):
    return 0.5 * (1 - x) # 0.5*(x+1)

class ECC_Dataset():
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        ss = random.choice(self.sigma)
        z = torch.randn(self.code.n) * ss
        y = bpsk_modulator(x) + z
        magnitude = torch.abs(y)
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)).long(),
                                self.pc_matrix) % 2
        syndrome = bpsk_modulator(syndrome)
        return m.float(), x.float(), y.float(), magnitude.float(), syndrome.float()


##################################################################
##################################################################

def train(model, device, train_loader, optimizer, epoch, LR):
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
        ber = calculate_ber(x_pred, x.to(device))
        fer = calculate_bler(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx+1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################

def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (m, x, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bpsk_modulator(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += calculate_ber(x_pred, x.to(device)) * x.shape[0]
                test_fer += calculate_bler(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                if (min_FER > 0 and test_fer > min_FER and cum_count > 1e5) or cum_count >= 1e9:
                    if cum_count >= 1e9:
                        print(f'Number of samples threshold reached for EbN0:{EbNo_range_test[ii]}')
                    else:
                        print(f'FER count threshold reached for EbN0:{EbNo_range_test[ii]}')
                    break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(f'Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}')
        ###
        print('\nTest Loss ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno) in (zip(test_loss_list, EbNo_range_test))]))
        print('Test FER ' + '{}: {:.2e}'.format(ebno, elem) for (elem, ebno) in (zip(test_loss_fer_list, EbNo_range_test)))
        print('Test BER ' + '{}: {:.2e}'.format(ebno, elem) for (elem, ebno)in (zip(test_loss_ber_list, EbNo_range_test)))
    return test_loss_list, test_loss_ber_list, test_loss_fer_list


def main():
    set_seed(42)

    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    # device = torch.device("cpu")
    device = torch.device("cuda")

    NN_type = "ECCT"
    nr_codeword = int(1e6)
    bits = 51
    encoded = 63
    encoding_method = "BCH"  # "Hamming", "Parity", "BCH",

    n_decoder = 6  # decoder iteration times
    dropout = 0  # dropout rate

    n_head = 8  # head number
    d_model = 128  # input embedding dimension
    epochs = 1000
    learning_rate = 0.001
    batch_size = 128
    patience = 10
    delta = 0.001

    model_save_path = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/"
    model_name = f"{NN_type}_h{n_head}_d{d_model}"


    model = ECC_Transformer(n_head, d_model, encoded, H, n_decoder, dropout, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    EbNo_range_test = torch.arange(4, 7, 1)
    EbNo_range_train = torch.arange(2, 8, 1)
    std_train = [EbN0_to_std(ii, bits / encoded) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, bits / encoded) for ii in EbNo_range_test]
    ECC_dataset = ECC_Dataset(code, std_train, len=batch_size * 1000, zero_cw=True)
    train_dataloader = DataLoader(ECC_dataset, batch_size=int(batch_size), shuffle=True)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], len=int(args.test_batch_size), zero_cw=False),
                                       batch_size=int(args.test_batch_size), shuffle=False) for ii in range(len(std_test))]


    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
        if epoch % 300 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, EbNo_range_test)

##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='BCH',
                        choices=['Hamming', 'BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=51)
    parser.add_argument('--code_n', type=int, default=63)
    parser.add_argument('--standardize', action='store_true')

    # model args
    parser.add_argument('--N_dec', type=int, default=6) # decoder is concatenation of N decoding layers of self-attention and feedforward layers and interleaved by normalization layers
    parser.add_argument('--d_model', type=int, default=32) # Embedding dimension
    parser.add_argument('--h', type=int, default=4) # multihead attention heads

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    class Code():
        pass
    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, _ = all_codebook_NonML(args.code_type, code.k, code.n, )
    H = ParitycheckMatrix(encoded, bits, method, device).squeeze(0).T


    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code
    ####################################################################
    model_dir = os.path.join('Results_ECCT',
                             args.code_type + '__Code_n_' + str(
                                 args.code_n) + '_k_' + str(
                                 args.code_k) +
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir

    main()
