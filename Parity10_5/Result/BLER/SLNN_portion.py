import matplotlib.pyplot as plt
import torch

SLNN_hidden_size = torch.arange(0, 101, 1)

BLER_SLNN_100 = [ ]

plt.semilogy(SLNN_hidden_size, BLER_SLNN_100, marker='.', label='N=7', color='blue', linestyle='--')

plt.xlabel('Number of Hidden Layer Neurons', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('Parity(10,5) BLER VS number of nodes for Single-label neural decoders', fontsize=20)
plt.legend(['N=7'], loc='upper right')


plt.show()



SLNN_hidden_size = torch.arange(0, 41, 1)

BLER_SLNN_100 = [ ]

plt.figure(figsize=(20, 10))
plt.semilogy(SLNN_hidden_size, BLER_SLNN_100, marker='.', label='N = 7', color='blue', linestyle='--')

plt.xlabel('Number of Hidden Layer Neurons', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('Parity(10,5) BLER VS number of nodes for Single-label neural decoders', fontsize=20)
plt.legend(['N=7'], loc='upper right')


plt.show()