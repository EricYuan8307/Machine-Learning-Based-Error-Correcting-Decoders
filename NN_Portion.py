import torch
import numpy as np
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import SingleLabelNNDecoder1
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_bler, calculate_ber
from Transmit.NoiseMeasure import NoiseMeasure
from Decode.Converter import DecimaltoBinary

from generating import all_codebook_NonML, SLNN_D2B_matrix
from Encode.Encoder import PCC_encoders

def SLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
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

    # Process in batches
    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    SLNN_binary_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        SLNN_result_batch = model(noised_signal_batch)
        SLNN_decimal_batch = torch.argmax(SLNN_result_batch, dim=2)
        SLNN_binary_batch = convertor(SLNN_decimal_batch)

        SLNN_binary_batches.append(SLNN_binary_batch)

        if i % 500 == 0 and i > 0:
            print(f"NN Decoder Batch: Processed batch: {i}/{num_batches}")

    # Concatenate the results
    SLNN_binary = torch.cat(SLNN_binary_batches, dim=0)


    return SLNN_binary, bits_info, practical_snr

def estimation_SLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, edge_deleted, model_pth, result, batch_size, device):
    N = num

    # Single-label Neural Network:
    condition_met = False
    iteration_count = 0  # Initialize iteration counter
    max_iterations = 50

    while not condition_met and iteration_count < max_iterations:
        iteration_count += 1

        if NN_type == "SLNN":
            output_size = torch.pow(torch.tensor(2), bits)
            model = SingleLabelNNDecoder1(encoded, NN_hidden_size, output_size).to(device)
            NN_final, bits_info, snr_measure = SLNNDecoder(N, method, bits, encoded, SNR_opt_NN, model, model_pth, batch_size, device)

        if metric == "BLER":
            error_rate, error_num = calculate_bler(NN_final, bits_info)
        elif metric == "BER":
            error_rate, error_num = calculate_ber(NN_final, bits_info)  # BER calculation

        if error_num < 100:
            N += int(1e7)
            print(f"the code number is {N}")

        else:
            print(
                f"{NN_type}, hiddenlayer{NN_hidden_size} - edgedeleted{edge_deleted}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
            result[0, edge_deleted.to(int)] = error_rate
            condition_met = True

    return result

def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    metrics = ["BLER"]
    num = int(2e7)
    bits = 10
    encoded = 26
    encoding_method = "Parity"
    NN_type = "SLNN"
    batch_size = int(1e4)
    SLNN_hidden_size = 26
    edge_delete_range = torch.arange(0, 26*26+1, 1)
    SNR_opt_NN = torch.tensor(7, dtype=torch.int, device=device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for SLNN article

    result_save = np.zeros((1, len(edge_delete_range)))

    for metric in metrics:
        for i in range(0, len(edge_delete_range)):
            model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{SLNN_hidden_size}_deleted_{device}/{NN_type}_deleted{edge_delete_range[i]}.pth"
            if not os.path.exists(model_pth):
                continue
            edge_delete = edge_delete_range[i]
            result_all = estimation_SLNN1(num, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, SLNN_hidden_size, edge_delete, model_pth, result_save, batch_size, device)
            directory_path = f"Result/{encoding_method}{encoded}_{bits}_deleted/{metric}/"

            # Create the directory if it doesn't exist
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            csv_filename = f"{NN_type}_{SLNN_hidden_size}_deleted.csv"
            full_csv_path = os.path.join(directory_path, csv_filename)
            np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()