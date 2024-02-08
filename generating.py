# Generate a codebook for 10-bit binary codewords

# Total number of bits for each codeword
num_bits = 10

# Generate all possible codewords in the new format
codebook_formatted = [list(map(int, format(i, f'0{num_bits}b'))) for i in range(2**num_bits)]

# for i in range(2**num_bits):
#     m = list(map(int, format(i, f'0{num_bits}b')))
#     print(m)

# print(codebook_formatted)
