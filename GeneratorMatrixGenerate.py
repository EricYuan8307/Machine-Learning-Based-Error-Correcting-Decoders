import numpy as np
import torch

def generaterMatrixGenerate(info_bits, Encoder_bits):
    """
    Generate a random generator matrix G and a corresponding decoder matrix D
    :param info_bits: the number of information bits
    :param Encoder_bits: the number of encoded bits
    :return: generator matrix G and decoder matrix D
    """

    r = Encoder_bits - info_bits

    # Construct G
    G_identity = np.eye(info_bits, dtype=int)
    G_parity = np.random.randint(0, 2, size=(info_bits, r))
    G = np.hstack((G_identity, G_parity)).T

    # Construct Decoder Matrix
    D_identity = np.eye(info_bits, dtype=int)
    D_zeros = np.zeros((info_bits, r), dtype=int)
    D = np.hstack((D_identity, D_zeros))

    return G, D


def ParityCheckMatirxGenerate(G):
    """
    :param G: The generator matrix
    :return: Parity check matrix H in systematic form
    """

    r = G.shape[1] - G.shape[0]

    # Extracting P from G
    P = G[:, G.shape[0]:]  # G is [I|P], so P starts from the n th column

    # Constructing H = [P^T|I]
    P_T = P.T  # Transpose of P
    I_n = torch.eye(r, dtype=torch.int)  # 13x13 Identity matrix

    # Concatenating P^T and I to form H
    H = torch.cat((P_T, I_n), dim=1)

    return H

# info= 24
# encoded = 49
#
# G, D = generaterMatrixGenerate(info, encoded)
# np.savetxt("G.csv", G, delimiter=',', fmt='%d')
# np.savetxt("D.csv", D, delimiter=',', fmt='%d')