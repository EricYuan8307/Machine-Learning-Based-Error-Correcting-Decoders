import torch.nn as nn
import torch

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.hidden = nn.Linear(7, 16, bias=True)  # input layer: 7 neurons to hidden layer: 7 neurons
        self.softmax = nn.Softmax(dim=1)       # Softmax activation function
        self.output = nn.Linear(16, 4, bias=True)  # hidden layer: 7 neurons to output layer: 16 neurons
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.softmax(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


nr_codewords = int(1e5)
seed = torch.tensor([941682])
print("Seed:", seed, flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Set seed
torch.manual_seed(seed)

# Load model parameters
loaded_model = torch.load('Result/Model/Hamming7_4/MLNN_cpu/MLNN_hiddenlayer16.pth')

# ANN Model
model_ann = ANN()
model_ann.load_state_dict(loaded_model)
# model_ann.to(device)
model_ann.eval()

# SNRs (dB)
SNRs = torch.arange(-5, 7.5, 0.5)

# Generator Matrix
G = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 1]])

# Number of codewords
multi = torch.zeros(len(SNRs))
ann_ber = torch.zeros(len(SNRs))

for idx_SNR, SNR in enumerate(SNRs):
    SNRlin = 10 ** (SNR / 10)

    errors_ann = 0

    while (errors_ann < 1000):
        multi[idx_SNR] += 1

        # Generation of the information sequence
        bits_info = torch.randint(0, 2, (nr_codewords, 4))

        # Encode
        codewords = torch.matmul(bits_info, G) % 2

        # BPSK modulation
        s = 2 * codewords - 1

        # AWGN Channel
        # Noise generation
        noise = torch.randn(s.shape)

        # Scale the noise power
        n = torch.sqrt(1 / (2 * SNRlin)) * noise

        # Add AWGN
        y = s + n

        # Apply model
        # y = y.to(device)

        with torch.no_grad():
            predictions_ann = model_ann(y)

        # Calculate the bit errors
        predicted_ann = predictions_ann > 0.5

        errors_ann += (predicted_ann.detach().cpu() != bits_info).sum().sum()

    ann_ber[idx_SNR] = errors_ann / (multi[idx_SNR] * nr_codewords * 4)
    print("SNR: ", SNR.item(), " ANN BER: ", ann_ber[idx_SNR].item(), flush=True)

print("ANN BER: ", ann_ber, flush=True)