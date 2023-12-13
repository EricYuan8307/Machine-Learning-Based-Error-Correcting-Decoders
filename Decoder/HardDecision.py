import torch

mps_device = (torch.device("mps") if torch.backends.mps.is_available()
              else (torch.device("cuda") if torch.backends.cuda.is_available()
                    else torch.device("cpu")))

def hard_decision_cutter(estimated_bits):
    tensor_1 = torch.tensor(1, device=mps_device)
    tensor_0 = torch.tensor(0, device=mps_device)
    estimated_bits = torch.where(estimated_bits > 0, tensor_1, tensor_0)
    estimated_bits = estimated_bits[:, :, 0:4]

    return estimated_bits