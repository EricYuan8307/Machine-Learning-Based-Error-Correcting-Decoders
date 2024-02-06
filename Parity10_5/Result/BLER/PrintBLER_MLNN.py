import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_BPSK = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
             0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
BLER_SDML = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
             0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
BLER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
                 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
BLER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
                   0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
BLER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
                     0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(20, 10))
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BLER_SLNN_7, label='Single-label Neural network N=7', color='blue')


plt.semilogy(SNR, BLER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
plt.semilogy(SNR, BLER_MLNN_50_50, marker='D', label='N1=50, N2=50', color = "orange", linestyle='--')
plt.semilogy(SNR, BLER_MLNN_100_100, marker='D', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('BLER Estimation', fontsize=20)
plt.legend(['Soft-decision ML', 'BPSK, Uncoded', 'Single-label Neural network N=7',
            # 'N=100, Article', 'N1=50, N2=50, Article','N1=100, N2=100, Article',
            'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')


plt.show()