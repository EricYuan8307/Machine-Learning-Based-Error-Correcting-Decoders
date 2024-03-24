import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

BER_uncoded_BPSK = [0.07736, 0.06564, 0.05604, 0.04576, 0.03804, 0.03134, 0.02232, 0.0173, # 0~3.5
                0.01336, 0.00856, 0.0061, 0.00368, 0.0025, 0.001376, 0.00076358, 0.00038925, 0.00018886] # 4.0~8.0

BER_SDML = [0.08216, 0.06296, 0.0492, 0.03682, 0.02502, 0.01782, 0.01056, 0.00646, # 0~3.5
        0.0037, 0.001874, 0.00089, 0.0003893, 0.0001503, 5.0099e-05, 1.57214e-05, 4.590818e-06, 0] # 4.0~8.0


# BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
#         0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
#         0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, # 0~3.5
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0] # 4~8.0


def nnplot():
    plt.figure(figsize=(16, 9))
    plt.semilogy(SNR, BER_uncoded_BPSK, label='BPSK, Uncoded', color = "green")
    plt.semilogy(SNR, BER_SDML, label='Soft-decision ML', color = "black")
    # plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('BER', fontsize=20)
    plt.title('Parity(10,5) Multi-label Neural Network BER Estimation', fontsize=20)
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