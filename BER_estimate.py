import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))


# Code Generation
def generator(nr_codewords):
    bits = torch.randint(0, 2, size=(nr_codewords, 1, 4), dtype=torch.int)

    return bits

# Hamming(7,4) Encoder
class hamming_encode(torch.nn.Module):
    def __init__(self):
        """
        Use Hamming(7,4) to encode the data.

        Args:
            data: data received from the Hamming(7,4) encoder(Tensor)
            generator matrix: generate the parity code

        Returns:
            encoded data: 4 bits original info with 3 parity code.
        """
        super(hamming_encode, self).__init__()

        # Define the generator matrix for Hamming(7,4)
        self.generator_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ], dtype=torch.int)

    def forward(self, input_data):
        # Ensure input_data has shape (batch_size)
        # assert input_data.size(0) == self.generator_matrix.shape[1], "Input data must have same generator matrix row number bits."

        # Perform matrix multiplication to encode the data
        # result_tensor = (self.generator_matrix @ input_data.squeeze(1).mT).unsqueeze(1).T % 2
        result_tensor = torch.matmul(input_data, self.generator_matrix.t())

        return result_tensor

# BPSK Modulator and Add Noise After Modulator
class bpsk_modulator(torch.nn.Module):
    def __init__(self):
        """
        Use BPSK to compress the data, which is easily to transmit.

        Args:
            codeword: data received from the Hamming(7,4) encoder(Tensor)

        Returns:
            data: Tensor contain all data modulated and add noise
        """
        super(bpsk_modulator, self).__init__()

    def forward(self, codeword, snr_dB):

        # data = torch.tensor(data, dtype=float)
        data = codeword.to(dtype=torch.float).to(mps_device)

        # for i in range(data.shape[0]):
        bits = data
        bits = 2 * bits - 1

        # Add Gaussian noise to the signal
        noise_power = torch.tensor(10**(snr_dB / 10)).to(mps_device)
        noise = torch.sqrt(1/(2*noise_power)) * torch.randn(bits.shape).to(mps_device)
        noised_signal = bits + noise
        # noised_signal = bits
        data = noised_signal

        return data

# Log-Likelihood Ratio
def llr(signal, snr):
    """
    Calculate Log Likelihood Ratio (LLR) for a simple binary symmetric channel.

    Args:
        signal (torch.Tensor): Received signal from BPSK.
        noise_std (float): Standard deviation of the noise.

    Returns:
        llr: Log Likelihood Ratio (LLR) values.
    """

    # Assuming Binary Phase Shift Keying (BPSK) modulation
    noise_std = torch.sqrt(torch.tensor(10**(snr / 10))).to(mps_device)

    # Calculate the LLR
    llr = 2 * signal * noise_std

    # return llr_values, llr
    return llr


class LDPCBeliefPropagation(torch.nn.Module):
    def __init__(self, H, llr):
        """
        LDPC Belief Propagation.

        Args:
            H: Low density parity code for building tanner graph.
            llr: Log Likelihood Ratio (LLR) values. Only for 7-bit codeword.

        Returns:
            estimated_bits: the output result from belief propagation.
        """

        super(LDPCBeliefPropagation, self).__init__()
        self.llr = llr
        self.H = H
        self.num_check_nodes, self.num_variable_nodes = H.shape
        self.channel = llr.shape[2]

        # Initialize messages
        self.messages_v_to_c = torch.ones((self.num_variable_nodes, self.num_check_nodes, self.channel),
                                          dtype=torch.float).to(mps_device)
        self.messages_c_to_v = torch.zeros((self.num_check_nodes, self.num_variable_nodes, self.channel),
                                           dtype=torch.float).to(mps_device)

    def forward(self, max_iter):
        start_time = time.time()
        for iteration in range(max_iter):

            # Variable to check node messages
            for i in range(self.num_variable_nodes):
                for j in range(self.num_check_nodes):
                    # Compute messages from variable to check nodes
                    connected_checks = self.H[j, :] == 1
                    product = torch.prod(torch.tanh(0.5 * self.messages_v_to_c[connected_checks, j]), dim=0,
                                         keepdim=True)
                    self.messages_v_to_c[i, j] = torch.sign(self.llr[j]) * product

            # Check to variable node messages
            for i in range(self.num_check_nodes):
                for j in range(self.num_variable_nodes):
                    # Compute messages from check to variable nodes
                    connected_vars = self.H[:, j] == 1
                    sum_msgs = torch.sum(self.messages_c_to_v[connected_vars, i]) - self.messages_v_to_c[j, i]
                    self.messages_c_to_v[i, j] = 2 * torch.atan(torch.exp(0.5 * sum_msgs))

        # Calculate the final estimated bits and only take first four bits
        estimated_bits = torch.sign(self.llr) * torch.prod(torch.tanh(0.5 * self.messages_c_to_v))
        tensor_1 = torch.tensor(1, device=mps_device)
        tensor_0 = torch.tensor(0, device=mps_device)
        estimated_bits = torch.where(estimated_bits > 0, tensor_1, tensor_0)
        estimated_bits = estimated_bits[:, :, 0:4]

        end_time = time.time()
        elapsed_time = end_time - start_time

        return estimated_bits, elapsed_time

# Calculate the BER compared to output of original
def calculate_ber(compare_bits, origin_bits):
    # Ensure that both tensors have the same shape
    assert compare_bits.shape == origin_bits.shape, "Shapes of transmitted and received bits must be the same."

    # Calculate the bit errors
    errors = (compare_bits != origin_bits).sum().item()

    # Calculate the Bit Error Rate (BER)
    ber = errors / compare_bits.numel()

    return errors, ber


# Code Generator
num = 1000000
bits_info = generator(num)

# Generation Encoded Data with 3 parity bits
encoder = hamming_encode()
encoded_codeword = encoder(bits_info)

# Signal-to-noise ratio in dB
snr_dB = 15

# Modulate the signal
modulator = bpsk_modulator()
modulated_noise_signal = modulator(encoded_codeword.to(mps_device), snr_dB)

# Log-Likelihood Calculation
llr_output = llr(modulated_noise_signal, snr_dB)

# LDPC Belief Propagation
H = torch.tensor([ [1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 0, 1],], device=mps_device)
iter = 20
ldpc_bp = LDPCBeliefPropagation(H, llr_output.to(mps_device))

final_result, time = ldpc_bp(iter)

# print(final_result)
# print(f"The Entire LDPC Belief propagation runs {time} seconds")

# Count error number and BER:
bits_info = bits_info.to(mps_device) # bits_info: original signal
decoded_bits = final_result #output from Maximum Likelihood

error_num, BER = calculate_ber(decoded_bits, bits_info)
print(BER)