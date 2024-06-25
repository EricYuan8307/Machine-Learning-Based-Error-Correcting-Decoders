import torch
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from Transformer.Codes_article import *
from generating import all_codebook_NonML

from torch.optim.lr_scheduler import CosineAnnealingLR
from Transformer.Model_ca import ECC_Transformer
from Codebook.CodebookMatrix import ParitycheckMatrix

def count_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, device):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1).to(device)
        self.pc_matrix = code.pc_matrix.transpose(0, 1).to(device)
        self.device = device

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        m = torch.randint(0, 2, (1, self.code.k), dtype=torch.float, device=self.device).squeeze()
        x = torch.matmul(m, self.generator_matrix) % 2
        ss = random.choice(self.sigma)
        z = torch.randn(self.code.n, device=self.device) * ss
        y = bin_to_sign(x) + z
        magnitude = sign_to_bin(torch.sign(y))
        syndrome = torch.matmul(sign_to_bin(torch.sign(y)), self.pc_matrix) % 2
        syndrome = bin_to_sign(syndrome)
        return m.float(), x.float(), z.float(), y.float(), magnitude.float(), syndrome.float()


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    for batch_idx, (m, x, z, y, magnitude, syndrome) in enumerate(train_loader):
        z_mul = (y * bin_to_sign(x))
        z_pred = model(magnitude.to(device), syndrome.to(device))
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()

        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1:
            print(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = [], [], [], []
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.
            while True:
                (m, x, z, y, magnitude, syndrome) = next(iter(test_loader))
                z_mul = (y * bin_to_sign(x))
                z_pred = model(magnitude.to(device), syndrome.to(device))
                loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))

                test_loss += loss.item() * x.shape[0]

                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
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
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_list, EbNo_range_test))]))
        print('Test FER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_fer_list, EbNo_range_test))]))
        print('Test BER ' + ' '.join(
            ['{}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ber_list, EbNo_range_test))]))
    return test_loss_list, test_loss_ber_list, test_loss_fer_list


def main(args):
    code = args.code

    #################################
    model = ECC_Transformer(args, dropout=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # logging.info(model)
    # logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')

    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    ECC_dataset = ECC_Dataset(code, std_train, args.batch_size * 1000, device)
    train_dataloader = DataLoader(ECC_dataset, batch_size=int(args.batch_size), shuffle=True)
    test_dataloader_list = [DataLoader(ECC_Dataset(code, [std_test[ii]], int(args.test_batch_size), device),
                                       batch_size=int(args.test_batch_size), shuffle=False) for ii in range(len(std_test))]

    total_trainable_params = count_trainable_parameters(model)
    print(f"Total trainable parameters: {total_trainable_params}")

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, fer = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), os.path.join(args.model_path, f"{args.model_name}.pth"))
            print(f"best model save: Loss={loss:.2e} BER={ber:.2e} FER={fer:.2e}")
        else:
            print("continue training")

        if epoch % 30 == 0 or epoch in [1, args.epochs]:
            test(model, device, test_dataloader_list, EbNo_range_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CrossMPT')
    parser.add_argument('--model_type', type=str, default='CrossMPT')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='BCH', choices=['Hamming', 'BCH', 'POLAR', 'LDPC'])
    parser.add_argument('--code_k', type=int, default=16)
    parser.add_argument('--code_n', type=int, default=31)
    parser.add_argument('--standardize', action='store_true')

    # model args
    parser.add_argument('--N_dec', type=int, default=6) # decoder is concatenation of N decoding layers of self-attention and feedforward layers and interleaved by normalization layers
    parser.add_argument('--d_model', type=int, default=128) # Embedding dimension
    parser.add_argument('--h', type=int, default=8) # multihead attention heads

    args = parser.parse_args()
    set_seed(args.seed)

    class Code():
        pass
    code = Code()
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cuda")
    # device = torch.device("cpu")
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    code.generator_matrix, _ = all_codebook_NonML(args.code_type, args.code_k, args.code_n, device)
    code.pc_matrix = ParitycheckMatrix(args.code_n, args.code_k, args.code_type, device).squeeze(0).T
    args.code = code

    args.model_path = f"Result/Model/{args.code_type}{args.code_n}_{args.code_k}/{args.model_type}_{device}/"
    args.model_name = f"{args.model_type}_h{args.h}_n{args.N_dec}_d{args.d_model}_HD"

    os.makedirs(args.model_path, exist_ok=True)

    main(args)
