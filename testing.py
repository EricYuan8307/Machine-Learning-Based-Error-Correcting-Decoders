import torch
import random

n=7
k=4
sigma = [0]
d_model = 32

def bin_to_sign(x):
    return 2 * x - 1

def sign_to_bin(x):
    return (x + 1) / 2

generator_matrix = torch.tensor([[1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]], dtype=torch.float)

pc_matrix = torch.tensor([[1, 0, 1, 0, 1, 0, 1],
                                 [0, 1, 1, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 1, 1, 1]], dtype=torch.float)  # Hamming(7,4) BP

m = torch.randint(0, 2, (5, 1, k)).to(torch.float)
x = torch.matmul(m, generator_matrix) % 2
z = torch.randn(n) * random.choice(sigma)
y = bin_to_sign(x) + z
magnitude = torch.abs(y)
binary = sign_to_bin(torch.sign(y))
syndrome = torch.matmul(binary, pc_matrix.T) % 2
syndrome1 = bin_to_sign(syndrome)

emb = torch.cat([magnitude, syndrome1], 2)

src_embed = torch.nn.Parameter(torch.empty((n + pc_matrix.size(0), d_model)))

emb = torch.matmul(emb, src_embed)
print(emb.shape)