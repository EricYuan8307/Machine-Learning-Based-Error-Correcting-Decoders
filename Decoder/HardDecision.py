import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

def hard_decision_BPSK(estimated_bits):
    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(estimated_bits > 0, tensor_1, tensor_0)
    # estimated_bits = estimated_bits[:, :, 0:4]

    return estimated_bits

def hard_decision_cutter(bitcodeword):
    r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)

    result = torch.matmul(bitcodeword, r.T)
    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(result > 0, tensor_1, tensor_0)

    return estimated_bits

# input0 = torch.randn([10000000, 1, 7]).to(mps_device)
# result = reverse_matrix(input0)
# input1 = hard_decision_cutter(result)
#
#
#
# print(input1)