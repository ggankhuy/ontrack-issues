# m n k: 2304, 8, 16394

import torch
import numpy as np

cuda = torch.device('cuda')

m = 2304
n = 8
k = 16384

t1 = torch.randn([m,n], dtype=torch.bfloat16, device=cuda)
t2 = torch.randn([n,k], dtype=torch.bfloat16, device=cuda)
print("t1, t2: ", t1.shape, t2.shape)
t3 = torch.mm(t1, t2) 
#print(np.array(t3).shape)
print(t3.shape)
