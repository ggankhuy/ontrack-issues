import torch
t1=torch.randn(10)
t2=torch.randn(10)
t3=torch.matmul(t1, t2)
print(t1.shape)
print(t2.shape)
print(t3.shape)
