import torch
num = 26
model_pth = "Result/MLNN_hiddenlayer16_peter.pth"
model = torch.load(model_pth)


print(model)