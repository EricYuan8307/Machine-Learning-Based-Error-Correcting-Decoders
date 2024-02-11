import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_BPSK = [0.3481, 0.3036, 0.2534, 0.2186, 0.1763, 0.1376, 0.105, 0.087, # 0.0~3.5
             0.0584, 0.0484, 0.0291, 0.0191, 0.0126, 0.00697, 0.00384527, 0.00197, 0.0009473] # 4.0~8.0
BLER_SDML = [0.222, 0.1811, 0.1446, 0.1187, 0.0867, 0.0599, 0.0394, 0.0266, # 0~3.5
             0.0164, 0.0101, 0.00594, 0.0030357, 0.001507, 0.00067, 0.000277, 0.000102, 3.68771e-05] # 4.0~8.0

BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
BLER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
BLER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
BLER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

plt.figure(figsize=(20, 10))
plt.semilogy(SNR, BLER_SDML,  label='Soft-decision ML', color = "black")
plt.semilogy(SNR, BLER_BPSK,  label='BPSK, Uncoded', color = "green")
# plt.semilogy(SNR, BLER_SLNN_7, label='Single-label Neural network N=7', color='blue')
#
#
# plt.semilogy(SNR, BLER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
# plt.semilogy(SNR, BLER_MLNN_50_50, marker='D', label='N1=50, N2=50', color = "orange", linestyle='--')
# plt.semilogy(SNR, BLER_MLNN_100_100, marker='D', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BLER', fontsize=20)
plt.title('Parity(10,5) BLER Estimation', fontsize=20)
plt.legend(['Soft-decision ML', 'BPSK, Uncoded'
               # , 'Single-label Neural network N=7',
            # 'N=100, Article', 'N1=50, N2=50, Article','N1=100, N2=100, Article',
            # 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'
            ], loc='lower left')


plt.show()