import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

def hard_decision(bitcodeword):
    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(bitcodeword > 0, tensor_1, tensor_0)

    return estimated_bits

# # input0 = torch.randn([10000000, 1, 7]).to(mps_device)
# input0 = torch.randint(0, 2, size = [1, 1, 7],dtype=int).to(mps_device)
#
# input1 = hard_decision_cutter(input0)
# #
# #
# #
# print("input0",input0)
# print(input1)