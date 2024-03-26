import torch

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder_nonfully
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary

from generating import all_codebook_NonML, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders

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

def SLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, device):
    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)
    SLNN_Matrix = SLNN_D2B_matrix(bits, device)

    encoder = PCC_encoders(encoder_matrix)
    convertor = DecimaltoBinary(SLNN_Matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    # use SLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    SLNN_result = model(noised_signal)
    SLNN_decimal = torch.argmax(SLNN_result, dim=2)

    SLNN_binary = convertor(SLNN_decimal)


    return SLNN_binary, bits_info, practical_snr

def estimation(num, method, bits, encoded, SNR_opt_NN, SLNN_hidden_size, model_pth, mask, edge_delete, order, device):
    # Single-label Neural Network:
    output_size = torch.pow(torch.tensor(2), bits)

    model = SingleLabelNNDecoder_nonfully(encoded, SLNN_hidden_size, output_size, mask).to(device)
    SLNN_final, bits_info, snr_measure = SLNNDecoder(num, method, bits, encoded, SNR_opt_NN, model, model_pth, device)

    BLER_SLNN, error_num_SLNN = calculate_bler(SLNN_final, bits_info) # BER calculation

    if error_num_SLNN < 100:
        num += 1000000
        print(f"the code number is {num}")

    else:
        print(f"SLNN edge deleted{edge_delete}, order{order}: When SNR is {snr_measure} and signal number is {num}, error number is {error_num_SLNN} and BLER is {BLER_SLNN}")

    return BLER_SLNN


def main():
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.cuda.is_available()
                    else torch.device("cpu")))
    # device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters for SLNN neuron=7
    num = int(1e7)
    bits = 4
    encoded = 7
    encoding_method = "Hamming"
    SLNN_hidden_size = 7
    edge_delete = 43
    parameter = "output.weight"
    order = torch.arange(0, 112, 1)
    mask = torch.tensor([[0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1],
                                 [0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0, 0]], dtype=torch.float, device=device) # To delete 42 edges


    SNR_opt_NN = torch.tensor(8, dtype=torch.int, device=device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    # load_pth = f"Result/Model/SLNN_edgedeleted{edge_delete}_{parameter}_{device}/SLNN7_edgedeleted{edge_delete}_{device}.pth"  # The model untrained
    # result_all = estimation(num, bits, encoded, SNR_opt_NN, SLNN_hidden_size, load_pth, mask, edge_delete, device)

    for i in range(len(order)):
        # mask = Mask43(order[i], device)
        load_pth = f"Result/Model/SLNN_edgedeleted{edge_delete}_{parameter}_{device}/SLNN7_edgedeleted{edge_delete}_order{order[i]}.pth" # The model untrained
        # load_pth = f"Result/Model/SLNN_edgedeleted{edge_delete}_trained_{parameter}_{device}_BER8/SLNN_edgedeleted{edge_delete}_order{order[i]}.pth" # The model trained
        result_all = estimation(num, encoding_method, bits, encoded, SNR_opt_NN, SLNN_hidden_size, load_pth, mask, edge_delete, order[i], device)
    # directory_path = "Result/BLER"
    #
    # # Create the directory if it doesn't exist
    # if not os.path.exists(directory_path):
    #     os.makedirs(directory_path)
    #
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # csv_filename = f"BLER_result_{current_time}.csv"
    # full_csv_path = os.path.join(directory_path, csv_filename)
    # np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()