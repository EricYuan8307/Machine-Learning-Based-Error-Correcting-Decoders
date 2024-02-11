import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK = [0.07860764, 0.0670996, 0.05626964, 0.04642562, 0.0374909, 0.02964128, 0.02288382, 0.0171741, # 0~3.5
                0.0125199, 0.0087939, 0.0059538, 0.00385776, 0.00238986, 0.00140726, 0.00077454, 0.0003984, 0.0001925]# 4.0~8.0

BER_SDML = [0.0848702, 0.0664146, 0.050346, 0.0365868, 0.0257976, 0.0170588, 0.010735, 0.006256, # 0~3.5
        0.0035448, 0.0017802, 0.000826, 0.0003926, 0.0001478, 4.6e-05, 1.51e-05, 4.56e-06, 0.0]# 4~8.0


# BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0


def nnplot():
    plt.figure(figsize=(20, 10))
    plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
    plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")
    # plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('BER', fontsize=20)
    plt.title('Parity(16,5) Multi-label Neural Network BER Estimation', fontsize=20)
    plt.legend([
        'BPSK, Uncoded',
        'Soft-decision ML',
        # 'N=100, Article',
        # 'N=100',
        # 'N1=50, N2=50',
        # 'N1=100, N2=100'
    ], loc='lower left')
    # Display the Plot
    plt.show()

def main():
    # originalplot()
    nnplot()

if __name__ == '__main__':
    main()