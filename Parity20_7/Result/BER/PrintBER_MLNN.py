import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8.0]

article_BPSK = [0.08, 0.067, 0.055, 0.046, 0.038, 0.0305, 0.023, 0.017, # 0~3.5
                0.013, 0.009, 0.006, 0.004, 0.0025, 0.0015, 0.0007, 0.0004, 0.0002]# 4.0~8.0
article_SDML = [0.12, 0.1, 0.08, 0.068, 0.05, 0.031, 0.021, 0.015, # 0~3.5
                0.0078, 0.0039, 0.002, 0.00098, 3.7e-04, 1.5e-04, 4e-05, 1.1e-05, 3.5e-06]# 4.0~8.0

BER_uncoded_BPSK = [0.07860764, 0.0670996, 0.05626964, 0.04642562, 0.0374909, 0.02964128, 0.02288382, 0.0171741, # 0~3.5
                0.0125199, 0.0087939, 0.0059538, 0.00385776, 0.00238986, 0.00140726, 0.00077454, 0.0003984, 0.0001925]# 4.0~8.0

BER_SDML = [0.09382857, 0.07122857, 0.0566142857, 0.0414, 0.02938571, 0.0188142857, 0.0119857142857, 0.007942857, # 0~3.5
        0.00427142857, 0.0026142857, 0.00126623377, 0.00062857143, 0.00031948, 0.000133766, 4.49477e-05, 2.41217799e-05, 0.0]# 4~8.0

# BER_SDML_R105 = [0.12203142857142857, 0.09985, 0.08040285714285714, 0.06210142857142857, 0.04662714285714286, 0.032632857142857144, 0.022555714285714287, 0.014808571428571428, # 0~3.5
#         0.008988571428571429, 0.005237142857142857, 0.003092857142857143, 0.00154, 0.0, 0.0, 0.0, 0.0, 0.0]# 4~8.0


# BER_MLNN_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_50_50 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0
#
# BER_MLNN_100_100 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # 0~3.5
#                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4~8.0


def nnplot():
    plt.figure(figsize=(16, 9))
    plt.semilogy(SNR, article_BPSK, label='BPSK, Uncoded, Article', color = "green")
    plt.semilogy(SNR, article_SDML, label='Soft-decision ML, Article', color = "black")

    plt.semilogy(SNR, BER_uncoded_BPSK, marker='.', label='BPSK, Uncoded', color = "green")
    plt.semilogy(SNR, BER_SDML, marker='x', label='Soft-decision ML', color = "black")
    # plt.semilogy(SNR, BER_SDML_R105, label='Soft-decision ML', color="yellow")
    # plt.semilogy(SNR, BER_MLNN_100, marker='D', label='N=100', color = "pink", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_50_50, marker='o', label='N1=50, N2=50', color = "orange", linestyle='--')
    # plt.semilogy(SNR, BER_MLNN_100_100, marker='v', label='N1=100, N2=100', color = "red", linestyle='--')

    plt.xlabel('SNR', fontsize=20)
    plt.ylabel('BER', fontsize=20)
    plt.title('Parity(20,7) Multi-label Neural Network BER Estimation', fontsize=20)
    plt.legend([
        'BPSK, Uncoded, Article',
        'Soft-decision ML, Article',
        'BPSK, Uncoded',
        'Soft-decision ML',
        # 'N=100, Article',
        # 'N=100',
        # 'N1=50, N2=50',
        # 'N1=100, N2=100'
        # "BER_SDML_R105",
    ], loc='lower left')
    # Display the Plot
    plt.show()

def main():
    # originalplot()
    nnplot()

if __name__ == '__main__':
    main()