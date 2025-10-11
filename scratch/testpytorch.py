
import torch

x = torch.arange(12, dtype=torch.float32)

x

x.numel()

X=x.reshape(3,4)
print(X)

torch.zeros((2,3,4))

torch.ones((2,3,4))

print(torch.randn(3,4))