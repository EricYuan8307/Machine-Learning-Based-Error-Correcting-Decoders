import matplotlib.pyplot as plt

# Data Storage BLER:
BLER_BPSK = [0.3381, 0.2956, 0.2518, 0.211, 0.1841, 0.1377, 0.1135, 0.0818, # 0.0~3.5
             0.0621, 0.0447, 0.0296, 0.0194, 0.0118, 0.007, 0.00384527, 0.00191, 0.0009473] # 4.0~8.0
BLER_SDML = [0.1845, 0.1471, 0.11, 0.0869, 0.0579, 0.0367, 0.024, 0.0159, # 0~3.5
             0.008506, 0.004556, 0.002075, 0.000919, 0.000384, 0.0001455, 4.019933e-05, 1.109878e-05, 0.0] # 4.0~8.0

# BLER_SLNN_7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0
# BLER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4.0~8.0

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