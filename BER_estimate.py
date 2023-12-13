import matplotlib.pyplot as plt
import torch
from Encoder.BPSK import bpsk_modulator
from Encoder.Hamming74 import hamming_encoder
from Decoder.likelihood import llr
from Decoder.LDPC_BP import LDPCBeliefPropagation
from Decoder.HardDecision import hard_decision_cutter
from Estimation.BitErrorRate import calculate_ber

# Code Generation
def generator(nr_codewords):
    bits = torch.randint(0, 2, size=(nr_codewords, 1, 4), dtype=torch.int)

    return bits

def main():
    bits_info = generator(num) # Code Generator

    encoder = hamming_encoder()
    encoded_codeword = encoder(bits_info) # Hamming(7,4) Encoder

    modulator = bpsk_modulator()
    modulated_noise_signal = modulator(encoded_codeword.to(mps_device), snr_dB) # Modulate the signal
    llr_output = llr(modulated_noise_signal, snr_dB) #LLR

    ldpc_bp = LDPCBeliefPropagation(llr_output.to(mps_device))
    LDPC_result = ldpc_bp(iter) # LDPC
    final_result = hard_decision_cutter(LDPC_result) #Hard Decision

    bits_info = bits_info.to(mps_device)
    ber = calculate_ber(final_result, bits_info) # BER calculation
    print(ber)

if __name__ == "__main__":
    mps_device = (torch.device("mps") if torch.backends.mps.is_available()
                  else (torch.device("cuda") if torch.backends.cuda.is_available()
                        else torch.device("cpu")))

    num = 1000000
    snr_dB = 15  # Signal-to-noise ratio in dB
    iter = 20 # LDPC Belief Propagation iteration time

    main()