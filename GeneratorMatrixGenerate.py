import numpy as np

def generaterMatrixGenerate(info_bits, Encoder_bits):
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

info= 24
encoded = 49

G, D = generaterMatrixGenerate(info, encoded)
np.savetxt("G.csv", G, delimiter=',', fmt='%d')
np.savetxt("D.csv", D, delimiter=',', fmt='%d')