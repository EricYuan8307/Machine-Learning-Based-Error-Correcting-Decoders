import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_BPSK = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
             0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
BLER_SDML = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
             0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_5 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_6 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_8 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_9 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(20, 10))
# plt.semilogy(SNR, article_SDML, marker='x', label='Soft-decision ML, Article')
# plt.semilogy(SNR, article_BPSK, marker='+', label='BPSK, Uncoded, Article')
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")

plt.semilogy(SNR, BLER_SLNN_5, marker='x', label='Single-label Neural network N=5', color='blue', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_6, marker='D', label='Single-label Neural network N=6', color='orange', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_7, marker='o', label='Single-label Neural network N=7', color='green', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_8, marker='v', label='Single-label Neural network N=8', color='red', linestyle='--')
plt.semilogy(SNR, BLER_SLNN_9, marker='<', label='Single-label Neural network N=9', color='purple', linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('BLER Estimation', fontsize=20)
# plt.legend(['SDML', 'BPSK'], loc='lower left')
# plt.legend(['Soft-decision ML, Article', 'BPSK, Uncoded, Article', 'Soft-decision ML', 'BPSK, Uncoded','Single-label Neural network'], loc='lower left')
plt.legend(['Soft-decision ML', 'BPSK, Uncoded',
            'Single-label Neural network N=5', 'Single-label Neural network N=6', 'Single-label Neural network N=7', 'Single-label Neural network N=8', 'Single-label Neural network N=9', ], loc='lower left')


plt.show()