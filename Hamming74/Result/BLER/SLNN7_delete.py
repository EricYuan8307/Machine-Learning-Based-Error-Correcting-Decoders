import matplotlib.pyplot as plt
import torch

SLNN_hidden_size = [0, 9, 14, 19, 24, 29]
SLNN_N7 = SLNN_N7_tensor = torch.full((6,), 1.66e-05)

output_weight = [

    ]

hidden_weight = [

    ]

hidden_output_weight = [

    ]

plt.figure(figsize=(16, 9))
plt.semilogy(SLNN_hidden_size, hidden_weight, marker='.', label='modified, hidden weight', color='red', linestyle='--')
plt.semilogy(SLNN_hidden_size, SLNN_N7, label='SLNN N=7, SNR=8', color='black')

plt.xlabel('Number of edges Deleted', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('SLNN, N=7, Edge Delete Performance(Trained)', fontsize=20)
plt.legend([
    'modified, hidden weight',
    'SLNN N=7, SNR=8',
], loc='upper left')

plt.show()
