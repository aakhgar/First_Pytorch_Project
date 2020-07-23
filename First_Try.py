
import torch

def activation(x):
    return 1/(1+torch.exp(-x))

torch.manual_seed(7)

features = torch.randn((1,5))

print(features)

weights = torch.randn_like(features).view((5,1))

print(weights)

bias = torch.randn((1,1))

y = activation(torch.mm(features, weights) + bias)

print(y)

    