import torch

from Encode.Encoder import PCC_encoders
from CodebookMatrix import coderMatrix
from Encode.Modulator import bpsk_modulator



def all_codebook(original, encoded, device):
    codebook_formatted = [list(map(int, format(i, f'0{original}b'))) for i in range(2 ** original)]
    bits_info_G = torch.tensor(codebook_formatted, dtype=torch.float, device=device) # SLNN_DecimaltoBinary

    # select encoder matrix and decoder matrix
    matrix = coderMatrix(device)
    encoder_matrix, decoder_matrix = matrix(encoded, original)

    encoder = PCC_encoders(encoder_matrix)
    encoded_info_G = encoder(bits_info_G)

    SDMLMatrix = bpsk_modulator(encoded_info_G)


    return encoder_matrix, decoder_matrix, SDMLMatrix

def SLNN_D2B_matrix(original, device):
    codebook_formatted = [list(map(int, format(i, f'0{original}b'))) for i in range(2 ** original)]
    SLNN_DecimaltoBinary = torch.tensor(codebook_formatted, dtype=torch.float, device=device)

    return SLNN_DecimaltoBinary