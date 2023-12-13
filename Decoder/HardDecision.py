import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

def hard_decision_BPSK(estimated_bits):
    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(estimated_bits > 0, tensor_1, tensor_0)

    return estimated_bits


def hard_decision_cutter(bitcodeword):
    r = torch.tensor([[0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 1], ],device=mps_device, dtype=torch.float)


    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(bitcodeword > 0, tensor_1, tensor_0)
    result = torch.matmul(estimated_bits.to(torch.float), r.T).to(torch.int)

    return result

# # input0 = torch.randn([10000000, 1, 7]).to(mps_device)
# input0 = torch.randint(0, 2, size = [1, 1, 7],dtype=int).to(mps_device)
#
# input1 = hard_decision_cutter(input0)
# #
# #
# #
# print("input0",input0)
# print(input1)