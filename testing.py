import torch
import torch.nn.functional as F

model_pth = "Result/Model/BCH63_51/ECCT_cuda/ECCT_h8_d128.pth"

weight_place = "out_fc.weight"

# models = torch.load(path)

model = torch.load(model_pth)
weight = model[weight_place]

print(weight.shape)