import torch
import numpy as np
import time
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Decode.HardDecision import hard_decision
from Decode.LDPC_BP import LDPCBeliefPropagation
from Decode.likelihood import llr
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber
from Decode.MaximumLikelihood import HardDecisionML74
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK

from generating import all_codebook
from Encode.Encoder import PCC_encoders
from Decode.MaximumLikelihood import SoftDecisionML
from Decode.Decoder import PCC_decoder

# Calculate the Error number and BER
def UncodedBPSK(nr_codeword, bits, snr_dB, device):
    bits_info = generator(nr_codeword, bits, device)
    modulated_signal = bpsk_modulator(bits_info)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    BPSK_final = hard_decision(noised_signal, device)

    practical_snr = NoiseMeasure_BPSK(noised_signal, modulated_signal)

    return BPSK_final, bits_info, practical_snr

def HardDecisionMLP(nr_codeword, bits, encoded, snr_dB, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    HD_MaximumLikelihood = HardDecisionML74(device)
    decoder = PCC_decoder(decoder_matrix)

    # ML:
    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    HD_signal = hard_decision(noised_signal, device)
    HD_ML = HD_MaximumLikelihood(HD_signal)
    HDML_final = decoder(HD_ML)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    return HDML_final, bits_info, practical_snr

def BeliefPropagation(nr_codeword, bits, encoded, snr_dB, iter, device):
    iter_start_time = time.time()

    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    ldpc_bp = LDPCBeliefPropagation(device)
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

def SoftDecisionMLP(nr_codeword, bits, encoded, snr_dB, device):
    encoder_matrix, decoder_matrix, SoftDecisionMLMatrix = all_codebook(bits, encoded, device)

    encoder = PCC_encoders(encoder_matrix)
    SD_MaximumLikelihood = SoftDecisionML(SoftDecisionMLMatrix)
    decoder = PCC_decoder(decoder_matrix)

    # ML:
    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    SD_ML = SD_MaximumLikelihood(noised_signal)
    HD_final = hard_decision(SD_ML, device)
    SDML_final = decoder(HD_final)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal, bits, encoded)

    return SDML_final, bits_info, practical_snr


def estimation_HDML(num, bits, encoded, SNR_opt_ML, result, device):
    N = num

    # Hard-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        for _ in range(10):
            HDML_final, bits_info, snr_measure = HardDecisionMLP(N, bits, encoded, snr_dB, device)

            BER_HDML, error_num_HDML = calculate_ber(HDML_final, bits_info)
            if error_num_HDML < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(
                    f"HD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_HDML} and BER is {BER_HDML}")
                result[0, i] = BER_HDML
                break

    return result

def estimation_BPSK(num, bits, SNR_opt_BPSK, result, device):
    N = num

    # De-Encoder, BPSK only
    for i in range(len(SNR_opt_BPSK)):
        snr_dB =SNR_opt_BPSK[i]

        for _ in range(10):
            BPSK_final, bits_info, snr_measure = UncodedBPSK(N, bits, snr_dB, device)

            BER_BPSK, error_num_BPSK= calculate_ber(BPSK_final, bits_info)
            if error_num_BPSK < 100:
                N += 2000000
                print(f"the code number is {N}")

            else:
                print(f"BPSK: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_BPSK} and BER is {BER_BPSK}")
                result[0, i] = BER_BPSK
                break

    return result

def estimation_BP(num, bits, encoded, SNR_opt_BP, iter, result, device):
    N = num

    # Belief Propagation
    for i in range(len(SNR_opt_BP)):
        snr_dB = SNR_opt_BP[i]

        for _ in range(10):
            LDPC_final, bits_info, snr_measure = BeliefPropagation(N, bits, encoded, snr_dB, iter, device)

            BER_LDPC, error_num_LDPC = calculate_ber(LDPC_final, bits_info) # BER calculation

            if error_num_LDPC < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(f"LDPC: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_LDPC} and BER is {BER_LDPC}")
                result[0, i] = BER_LDPC
                break

    return result

def estimation_SDML(num, bits, encoded, SNR_opt_ML, result, device):
    N = num

    # Soft-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        # BER
        for _ in range(10):
            SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, bits, encoded, snr_dB, device)

            BER_SDML, error_num_SDML = calculate_ber(SDML_final, bits_info)
            if error_num_SDML < 100:
                N += 1000000
                print(f"the code number is {N}")

            else:
                print(f"SD-ML: When SNR is {snr_measure} and signal number is {N}, error number is {error_num_SDML} and BER is {BER_SDML}")
                result[0, i] = BER_SDML
                break

    return result


def main():
    # device = (torch.device("mps") if torch.backends.mps.is_available()
    #           else (torch.device("cuda") if torch.backends.cuda.is_available()
    #                 else torch.device("cpu")))
    device = torch.device("cpu")
    # device = torch.device("cuda")

    # Hyperparameters
    num = int(7)
    iter = 5
    bits = 4
    encoded = 7
    SNR_opt_BPSK = torch.arange(0, 8.5, 0.5)
    SNR_opt_BP = torch.arange(0, 9, 0.5)

    SNR_opt_ML = torch.arange(0, 8.5, 0.5)
    SNR_opt_ML = SNR_opt_ML + 10 * torch.log10(torch.tensor(bits / encoded, dtype=torch.float)) # for MLNN article

    result_save = np.zeros((1, len(SNR_opt_BPSK)))
    # result_BPSK = estimation_BPSK(num, bits, SNR_opt_BPSK, result_save, device)
    result_SDML = estimation_SDML(num, bits, encoded, SNR_opt_ML, result_save, device)

    result_HDML = estimation_HDML(num, bits, encoded, SNR_opt_ML, result_save, device)
    result_BP = estimation_BP(num, bits, encoded, SNR_opt_BP, iter, result_save, device)

    result_all = np.vstack([
        # result_BPSK,
        result_SDML,
        result_HDML,
        result_BP
    ])


    directory_path = "Result/BER"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"BER_result_{current_time}.csv"
    full_csv_path = os.path.join(directory_path, csv_filename)
    np.savetxt(full_csv_path, result_all, delimiter=', ')


if __name__ == "__main__":
    main()