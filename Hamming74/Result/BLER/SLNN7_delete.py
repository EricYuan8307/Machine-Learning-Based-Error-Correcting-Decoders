import matplotlib.pyplot as plt
import torch

SLNN_hidden_size = [9, 14, 19, 24, 29, 34, 39, 40, 41, 42, 43]
SLNN_N7 = SLNN_N7_tensor = torch.full((11,), 1.66e-05)

hidden_weight = [
    1.73e-05, 1.51e-05, 1.61e-05, 1.585e-05, 1.535e-05, 1.275e-05, 1.27e-05, 0.00030795, 0.00449175, 0.0043446, 0.00431055
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
