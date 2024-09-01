import torch
from torch_geometric.nn.dense.linear import Parameter
from torch_geometric.nn.inits import constant
a = torch.Tensor(16, 4, 64)

b =  Parameter(torch.Tensor(1,4,64))
constant(b, 1.)


print(a.shape)
print(b.shape)
print(a*b.shape)