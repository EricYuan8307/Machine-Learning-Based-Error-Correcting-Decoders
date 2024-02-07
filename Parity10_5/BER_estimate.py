import torch
import numpy as np
import os
from datetime import datetime

from Encode.Generator import generator
from Encode.Modulator import bpsk_modulator
from Encode.Encoder import Parity10_5_encoder
from Decode.HardDecision import hard_decision
from Transmit.noise import AWGN
from Metric.ErrorRate import calculate_ber
from Decode.Decoder import Parity10_5decoder
from Decode.MaximumLikelihood import SoftDecisionML10_5
from Transmit.NoiseMeasure import NoiseMeasure, NoiseMeasure_BPSK


# Calculate the Error number and BER
def UncodedBPSK(nr_codeword, bits, snr_dB, device):
    bits_info = generator(nr_codeword, bits, device)
    modulated_signal = bpsk_modulator(bits_info)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    BPSK_final = hard_decision(noised_signal, device)

    practical_snr = NoiseMeasure_BPSK(noised_signal, modulated_signal)

    return BPSK_final, bits_info, practical_snr

def SoftDecisionMLP(nr_codeword, bits, snr_dB, device):
    encoder = Parity10_5_encoder(device)
    SD_MaximumLikelihood = SoftDecisionML10_5(device)
    decoder = Parity10_5decoder(device)

    # ML:
    bits_info = generator(nr_codeword, bits, device)
    encoded_codeword = encoder(bits_info)
    modulated_signal = bpsk_modulator(encoded_codeword)
    noised_signal = AWGN(modulated_signal, snr_dB, device)

    SD_ML = SD_MaximumLikelihood(noised_signal)
    HD_final = hard_decision(SD_ML, device)
    SDML_final = decoder(HD_final)

    practical_snr = NoiseMeasure(noised_signal, modulated_signal)

    return SDML_final, bits_info, practical_snr


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

def estimation_SDML(num, bits, SNR_opt_ML, result, device):
    N = num

    # Soft-Decision Maximum Likelihood
    for i in range(len(SNR_opt_ML)):
        snr_dB = SNR_opt_ML[i]

        # BER
        for _ in range(10):
            SDML_final, bits_info, snr_measure = SoftDecisionMLP(N, bits, snr_dB, device)

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
    num = int(1e6)
    bits = 5
    SNR_opt_BPSK = torch.arange(0, 8.5, 0.5)

    SNR_opt_ML = torch.arange(0, 8.5, 0.5)
    SNR_opt_ML = SNR_opt_ML + 10 * torch.log10(torch.tensor(bits / 10, dtype=torch.float)) # for MLNN article

    result_save = np.zeros((1, len(SNR_opt_BPSK)))
    result_BPSK = estimation_BPSK(num, bits, SNR_opt_BPSK, result_save, device)
    result_SDML = estimation_SDML(num, bits, SNR_opt_ML, result_save, device)

    result_all = np.vstack([result_BPSK,
                            result_SDML,
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