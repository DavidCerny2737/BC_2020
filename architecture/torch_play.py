import torch

a = torch.randn((2, 1, 2))
b = torch.randn((2, 1, 2))
c = (a, b)
print(a)
print(b)
d = torch.cat((c[0], c[1]), 2)
print(d)
print(d.shape)
d = torch.transpose(d, 0, 1).contiguous()
print(d.shape)
e = d.view(d.shape[0], -1)
print(e.shape)