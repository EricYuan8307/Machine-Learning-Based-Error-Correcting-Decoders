import numpy as np

def generaterMatrixGenerate(info_bits, Encoder_bits):
    r = Encoder_bits - info_bits

    # Construct G
    G_identity = np.eye(info_bits, dtype=int)
    G_parity = np.random.randint(0, 2, size=(info_bits, r))
    G = np.hstack((G_identity, G_parity))

    return G