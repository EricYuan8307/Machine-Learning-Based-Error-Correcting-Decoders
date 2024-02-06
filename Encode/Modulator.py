# BPSK Modulator and Add Noise After Modulator
def bpsk_modulator(codeword):
    codeword = 2 * codeword - 1

    return codeword