import torch
import numpy as np
import time
import os

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.HardDecision import hard_decision
from Decode.LDPC_BP import LDPCBeliefPropagation
from Decode.likelihood import llr
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber, calculate_bler
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK

from generating import all_codebook_NonML, all_codebook_HDML, all_codebook_SDML
from Encode.Encoder import PCC_encoders
from Decode.MaximumLikelihood import SoftDecisionML, HardDecisionML
from Decode.Decoder import PCC_decoder
from Codebook.CodebookMatrix import ParitycheckMatrix

# Calculate the Error number and BER
def UncodedBPSK(nr_codeword, bits, snr_dB, device):
    bits_info = generator(nr_codeword, bits, device)
    modulated_signal = bpsk_modulator(bits_info)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    BPSK_final = hard_decision(noised_signal, device)

    practical_snr = NoiseMeasure_BPSK(noised_signal, modulated_signal)

    return BPSK_final, bits_info, practical_snr

def BeliefPropagation(nr_codeword, method, bits, encoded, snr_dB, iter, H, device):
    iter_start_time = time.time()

    encoder_matrix, decoder_matrix = all_codebook_NonML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    ldpc_bp = LDPCBeliefPropagation(H, device)
    decoder = PCC_decoder(decoder_matrix)

    bits_info = generator(nr_codeword, bits, device)  # Code Generator
    encoded_codeword = encoder(bits_info)  # Hamming(7,4) Encoder
    modulated_signal = bpsk_modulator(encoded_codeword)  # Modulate signal
    noised_signal = AWGN(modulated_signal, snr_dB, device)  # Add Noise

    llr_output = llr(noised_signal, snr_dB)  # LLR
    BP_result = torch.zeros(llr_output.shape, device=device)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    for k in range(llr_output.shape[0]):
        start_time = time.time()

        BP = ldpc_bp(llr_output[k], iter)  # LDPC
        BP_result[k] = BP
        end_time = time.time()

        if k % 10000 == 0 and k > 0:
            elapsed_time = end_time - start_time
            print(f"Processed {k} iterations in {elapsed_time * 10000} seconds")

    iter_end_time = time.time()
    print(f"For {practical_snr}SNR, the Belief Propagation spend {iter_end_time - iter_start_time} seconds.")

    LDPC_HD = hard_decision(BP_result, device)  # Hard Decision
    LDPC_final = decoder(LDPC_HD)  # Decoder

    return LDPC_final, bits_info, practical_snr

def HardDecisionMLP(nr_codeword, method, bits, encoded, snr_dB, batch_size,device):
    encoder_matrix, decoder_matrix, HardDecisionMLMatrix = all_codebook_HDML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    HD_MaximumLikelihood = HardDecisionML(HardDecisionMLMatrix)
    decoder = PCC_decoder(decoder_matrix)

    # ML:
    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    HD_ML_batches = []

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        HD_signal_batch = hard_decision(noised_signal_batch, device)
        HD_ML_batch = HD_MaximumLikelihood(HD_signal_batch)
        HD_ML_batches.append(HD_ML_batch)

        if i % 500 == 0 and i > 0:
            print(f"NN Decoder Batch: Processed batch: {i}/{num_batches}")

    # Concatenating the processed batches
    HD_ML = torch.cat(HD_ML_batches, dim=0)
    HDML_final = decoder(HD_ML)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    return HDML_final, bits_info, practical_snr

def SoftDecisionMLP(nr_codeword, method, bits, encoded, snr_dB, batch_size, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook_SDML(method, bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    SD_MaximumLikelihood = SoftDecisionML(SoftDecisionMLMatrix)
    decoder = PCC_decoder(decoder_matrix)

    # ML:
    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    num_batches = int(np.ceil(noised_signal.shape[0] / batch_size))
    SD_ML_batches = []
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, noised_signal.shape[0])
        noised_signal_batch = noised_signal[batch_start:batch_end]

        SD_ML_batch = SD_MaximumLikelihood(noised_signal_batch)
        SD_ML_batches.append(SD_ML_batch)

        if i % 500 == 0 and i > 0:
            print(f"NN Decoder Batch: Processed batch: {i}/{num_batches}")

    # Concatenating the processed batches
    SD_ML = torch.cat(SD_ML_batches, dim=0)
    HD_final = hard_decision(SD_ML, device)
    SDML_final = decoder(HD_final)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    return SDML_final, bits_info, practical_snr


def estimation_HDML(num, method, bits, encoded, SNR_opt_ML, metric, result, batch_size, device):
    N = num

    # Hard-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            HDML_final, bits_info, snr_measure = HardDecisionMLP(N, method, bits, encoded, snr_dB, batch_size, device)

            if metric == "BER":
                error_rate_HDML, error_num_HDML = calculate_ber(HDML_final, bits_info)
            elif metric == "BLER":
                error_rate_HDML, error_num_HDML = calculate_bler(HDML_final, bits_info)
            else:
                print(f"{metric} is not either BER or BLER")

            if error_num_HDML < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(
                    f"HD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_HDML} and {metric} is {error_rate_HDML}")
                result[0, i] = error_rate_HDML
                condition_met = True

    return result

def estimation_BPSK(num, bits, SNR_opt_BPSK, metric, result, device):
    N = num

    # De-Encoder, BPSK only
    for i in range(len(SNR_opt_BPSK)):
        snr_dB =SNR_opt_BPSK[i]
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            BPSK_final, bits_info, snr_measure = UncodedBPSK(N, bits, snr_dB, device)

            if metric == "BER":
                error_rate_BPSK, error_num_BPSK = calculate_ber(BPSK_final, bits_info)
            elif metric == "BLER":
                error_rate_BPSK, error_num_BPSK = calculate_bler(BPSK_final, bits_info)
            else:
                print(f"{metric} is not either BER or BLER")

            if error_num_BPSK < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"BPSK: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BPSK} and {metric} is {error_rate_BPSK}")
                result[0, i] = error_rate_BPSK
                condition_met = True

    return result

def estimation_BP(num, method, bits, encoded, SNR_opt_BP, iter, H, metric, result, device):
    N = num

    # Belief Propagation
    for i in range(len(SNR_opt_BP)):
        snr_dB = SNR_opt_BP[i]
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            BP_final, bits_info, snr_measure = BeliefPropagation(N, method, bits, encoded, snr_dB, iter, H, device)

            if metric == "BER":
                error_rate_BP, error_num_BP = calculate_ber(BP_final, bits_info)
            elif metric == "BLER":
                error_rate_BP, error_num_BP = calculate_bler(BP_final, bits_info)
            else:
                print(f"{metric} is not either BER or BLER")

            if error_num_BP < 100:
                N += int(1e6)
                print(f"the code number is {N}")

            else:
                print(f"{method} iter{iter}: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BP} and {metric} is {error_rate_BP}")
                result[0, i] = error_rate_BP
                condition_met = True

    return result

def estimation_SDML(num, method, bits, encoded, SNR_opt_ML, metric, result, batch_size, device):
    N = num

    # Soft-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]
        condition_met = False
        iteration_count = 0  # Initialize iteration counter
        max_iterations = 50

        while not condition_met and iteration_count < max_iterations:
            iteration_count += 1

            SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, method, bits, encoded, snr_dB, batch_size, device)

            if metric == "BER":
                error_rate_SDML, error_num_SDML = calculate_ber(SDML_final, bits_info)
            elif metric == "BLER":
                error_rate_SDML, error_num_SDML = calculate_bler(SDML_final, bits_info)
            else:
                print(f"{metric} is not either BER or BLER")

            if error_num_SDML < 100:
                N += int(1e7)
                print(f"the code number is {N}")

            else:
                print(f"SD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_SDML} and {metric} is {error_rate_SDML}")
                result[0, i] = error_rate_SDML
                condition_met = True

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(1e4)
    bits = 4
    encoded = 7
    encoding_method = "Hamming" # "Hamming", "Parity", "BCH", "Golay", "LDPC"
    metrics = ["BER"] # BER or BLER
    batch_size = int(1e4)

    # iter = 10 # BP
    H = ParitycheckMatrix(encoded, bits, encoding_method, device)

    SNR_opt_BPSK = torch.arange(0, 8.5, 0.5)
    SNR_opt_BP = torch.arange(0, 8.5, 0.5)
    SNR_opt_BP = SNR_opt_BP + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))
    SNR_opt_ML = torch.arange(0, 7.5, 0.5)
    SNR_opt_ML = SNR_opt_ML + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float))

    result_save_BPSK = np.zeros((1, len(SNR_opt_BPSK)))
    result_save_SDML = np.zeros((1, len(SNR_opt_ML)))
    result_save_HDML = np.zeros((1, len(SNR_opt_ML)))
    result_save_BP = np.zeros((1, len(SNR_opt_BP)))

    for metric in metrics:
        # result_BPSK = estimation_BPSK(num, bits, SNR_opt_BPSK, metric, result_save_BPSK, device)
        result_SDML = estimation_SDML(num, encoding_method, bits, encoded, SNR_opt_ML, metric, result_save_SDML, batch_size, device)
        # result_HDML = estimation_HDML(num, encoding_method, bits, encoded, SNR_opt_ML, metric, result_save_HDML, batch_size, device)
        # result_BP = estimation_BP(num, encoding_method, bits, encoded, SNR_opt_BP, iter, H, metric, result_save_BP, device)

        result_all = np.vstack([
            # result_BPSK,
            result_SDML,
            # result_HDML,
            # result_BP
        ])


        directory_path = f"Result/{encoding_method}{encoded}_{bits}/{metric}"

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        csv_filename = f"{metric}_{encoding_method}{encoded}_{bits}.csv"
        full_csv_path = os.path.join(directory_path, csv_filename)
        np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()