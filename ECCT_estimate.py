import torch
import numpy as np
import os

from Encode.Generator import generator_ECCT
from Encode.Modulator import bpsk_modulator
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK
from generating import all_codebook_NonML
from Encode.Encoder import PCC_encoders
from Decode.Converter import MLNN_decision
from Transformer.Model import ECC_Transformer
from Codebook.CodebookMatrix import ParitycheckMatrix

# Calculate the Error number and BLER
def ECCTDecoder(nr_codeword, method, bits, encoded, snr_dB, model, model_pth, batch_size, device):
    encoder_matrix, _ = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)

    bits_info = generator_ECCT(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    # use MLNN model:
    model.eval()
    model.load_state_dict(torch.load(model_pth))

    # Splitting the noised_signal into batches and processing each batch
    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    ECCT_final_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        ECCT_result_batch = model(noised_signal_batch)
        ECCT_final_batches.append(ECCT_result_batch)

        if i % batch_size == 0 and i > 0:
            print(f"ECCT Batch: Processed {i} batches out of {num_batches}")

    # Concatenate the processed batches
    ECCT_final = torch.cat(ECCT_final_batches, dim=0)
    ECCT_final = MLNN_decision(torch.sign(ECCT_final*torch.sign(noised_signal)), device)

    return ECCT_final, bits_info, practical_snr

def estimation_ECCT(num, method, bits, encoded, n_head, d_model, n_dec, NN_type, metric, SNR_opt_NN, model_pth, result, batch_size, dropout, device):
    N = num
    H = ParitycheckMatrix(encoded, bits, method, device).squeeze(0)

    # Single-label Neural Network:
    for i in range(len(SNR_opt_NN)):
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1
            model = ECC_Transformer(n_head, d_model, encoded, H, n_dec, dropout, device).to(device)
            ECCT_final, bits_info, snr_measure = ECCTDecoder(N, method, bits, encoded, SNR_opt_NN[i], model, model_pth, batch_size, device)

            if metric == "BLER":
                error_rate, error_num = calculate_bler(ECCT_final, bits_info)
            elif metric == "BER":
                error_rate, error_num = calculate_ber(ECCT_final, bits_info) # BER calculation

            if error_num < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"{NN_type}, Head Num:{n_head}, Encoding Dim:{d_model}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num} and {metric} is {error_rate}")
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
    metrics = ["BER"] # ["BER", "BLER"]
    nr_codeword = int(1e7)
    bits = 4
    encoded = 7
    encoding_method = "Hamming"  # "Hamming", "Parity", "BCH"
    NN_type = "ECCT"
    batch_size = int(1e4)
    d_model = 16
    n_head = 8
    n_dec = 6
    dropout = 0

    SNR_opt_NN = torch.arange(0, 8.5, 0.5).to(device)
    SNR_opt_NN = SNR_opt_NN + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))
    result_save = np.zeros((1, len(SNR_opt_NN)))

    # trained and deleted model
    for metric in metrics:
        model_pth = f"Result/Model/{encoding_method}{encoded}_{bits}/{NN_type}_{device}/{NN_type}_h{n_head}_d{d_model}.pth"
        result_NN = estimation_ECCT(nr_codeword, encoding_method, bits, encoded, n_head, d_model, n_dec, NN_type, metric,
                                SNR_opt_NN, model_pth, result_save, batch_size, dropout, device)
        directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        csv_filename = f"{metric}_{NN_type}_h{n_head}_d{d_model}.csv"
        full_csv_path = os.path.join(directory_path, csv_filename)
        np.savetxt(full_csv_path, result_NN, delimiter=', ')

if __name__ == "__main__":
    main()