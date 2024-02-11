import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

article_BPSK = [0.078, 0.066, 0.055, 0.045, 0.039, 0.030, 0.022, 0.018, # 0~3.5
                0.013, 0.0088, 0.0059, 0.0039, 0.0024, 0.0014, 0.00079, 0.0004, 0.00018] # 4~8.0
article_SDML = [0.099, 0.072, 0.059, 0.04, 0.022, 0.015, 0.0075, 0.0041, # 0~3.5
                0.0021, 0.0008, 0.0003, 0.0001, 2.1e-5, 7.1e-6, 1.9e-6, 3.5e-7, 5.2e-8] # 4~8.0
article_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0


BER_uncoded_BPSK = [0.0785931, 0.0670444, 0.0563007, 0.0463861, 0.037532, 0.0297584, 0.0228147, 0.017205, # 0~3.5
                    0.0124542, 0.0088088, 0.0059633, 0.0038867, 0.0023655, 0.0014013, 0.0007842, 0.0003949, 0.0001866] # 4~8.0

BER_SDML = [0.11825, 0.0976, 0.08251, 0.06465, 0.04962, 0.03668, 0.02662, 0.01955, # 0~3.5
            0.01257, 0.00895, 0.00565, 0.00337, 0.00187, 0.00127, 0.00061, 0.000387, 0.000215]# 4~8.0

BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# 4~8.0

BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# 4~8.0

BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]# 4~8.0


plt.figure(figsize=(20, 10))
plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")
# plt.semilogy(SNR, article_BPSK, marker='.', label='BPSK, Uncoded, Article', color = "green")
plt.semilogy(SNR, article_SDML, marker='x', label='Soft-decision ML, Article', color = "black")
# plt.semilogy(SNR, article_MLNN_100, marker='D', label='N=100, Article', color = "pink", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
# plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

plt.xlabel('SNR', fontsize=20)
plt.ylabel('BER', fontsize=20)
plt.title('Multi-label Neural Network BER Estimation', fontsize=20)
plt.legend(['BPSK, Uncoded',
            'Soft-decision ML',
            # 'BPSK, Uncoded, Article',
            'Soft-decision ML, Article'], loc='lower left')
# plt.legend(['BPSK, Uncoded', 'Soft-decision ML', 'N=100, Article', 'N=100', 'N1=50, N2=50', 'N1=100, N2=100'], loc='lower left')

plt.show()