import matplotlib.pyplot as plt

# Data Storage Reference:
SNR = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]



# def originalplot():
#     plt.figure(figsize=(16, 9))
#     plt.semilogy(SNR, ref_HDML, marker='*', label='ref_HDML')
#     plt.semilogy(SNR, ref_SDML, marker='x', label='ref_SDML')
#     plt.semilogy(SNR, ref_BP, marker='+', label='ref_BP')
#     plt.semilogy(SNR, uncoded_BPSK, marker='.', label='uncoded_BPSK')
#     plt.semilogy(SNR, HDML, marker='*', label='HDML')
#     plt.semilogy(SNR, SDML, marker='x', label='SDML')
#     plt.semilogy(SNR, BP, marker='+', label='BP')
#
#     plt.xlabel('SNR', fontsize=20)
#     plt.ylabel('BER', fontsize=20)
#     plt.title('BER Estimation', fontsize=20)
#     plt.legend(['ref_HDML', 'ref_SDML', 'ref_BP', 'uncoded_BPSK', 'HDML', 'SDML', "BP"], loc='lower left')
#
#     # Display the Plot
#     plt.show()
#
# def main():
#     originalplot()
#
# if __name__ == '__main__':
#     main()