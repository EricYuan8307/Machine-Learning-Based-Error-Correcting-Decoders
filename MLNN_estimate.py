import torch
import numpy as np
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.NNDecoder import MultiLabelNNDecoder_N
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure_MLNN
from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders
from Decode.Converter import MLNN_decision


def MLNNDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure_MLNN(noised_signal, modulated_signal, bits, encoded)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    # Splitting the noised_signal into batches and processing each batch
    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    MLNN_final_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        MLNN_result_batch = model(noised_signal_batch)
        MLNN_final_batches.append(MLNN_result_batch)

        if i % batch_size == 0 and i > 0:
            print(f"NN Decoder Batch: Processed {i} batches out of {num_batches}")

    # Concatenate the processed batches
    MLNN_final = torch.cat(MLNN_final_batches, dim=0)

    return MLNN_final, bits_info, practical_snr

def estimation_MLNN1(num, method, bits, encoded, NN_type, metric, SNR_opt_NN, NN_hidden_size, model_pth, result, batch_size, device):
    N = num

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            model = MultiLabelNNDecoder_N(encoded, NN_hidden_size, bits).to(device)
            NN_result, bits_info, snr_measure = MLNNDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)
            NN_final = MLNN_decision(NN_result, device)

            if metric == "BLER":
                error_rate, error_num = calculate_bler(NN_final, bits_info)
            elif metric == "BER":
                error_rate, error_num = calculate_ber(NN_final, bits_info) # BER calculation

            if error_num < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"{NN_type}, hiddenlayer{NN_hidden_size}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
                result[0, i] = error_rate
                condition_met = True

    return result

def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    metric = "BER"
    nr_codeword = int(1e7)
    bits = 4
    encoded = 7
    encoding_method = "Hamming"
    NN_type = "MLNN"
    batch_size = int(1e4)
    MLNN_hidden_size = 16

    SNR_opt_NN = torch.arange(0, 7.5, 0.5).to(device)
    result_save = np.zeros((1, len(SNR_opt_NN)))

    # For trained and deleted model
    model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_hiddenlayer{MLNN_hidden_size}.pth"
    result_NN = estimation_MLNN1(nr_codeword, encoding_method, bits, encoded, NN_type, metric, SNR_opt_NN, MLNN_hidden_size, model_pth, result_save, batch_size, device)
    directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    csv_filename = f"{metric}_{NN_type}_hiddenlayer{MLNN_hidden_size}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_NN, delimiter=', ')



if __name__ == "__main__":
    main()