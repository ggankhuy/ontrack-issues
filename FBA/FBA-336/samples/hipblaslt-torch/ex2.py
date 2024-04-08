#!/usr/bin/python3
import torch

'''
workingdims without T.
t1=torch.randn([2304, 8, 16384], dtype=torch.float32)
t2=torch.randn([2304, 16384, 8], dtype=torch.float32)
'''

t1=torch.randn([16384, 8, 2304], dtype=torch.float32)
t2=torch.randn([2304, 16384, 8], dtype=torch.float32)

print(t1.shape)
print(t2.shape)
print("after T:")
print(t1.T.shape)
print(t2.shape)

t3=torch.matmul(t1.T, t2)
print(t3.shape)

'''
torch.Size([2304, 8, 16384])
torch.Size([2304, 16384, 8])
after T:
/root/extdir/gg/git/fba/FBA/FBA-336/samples/hipblaslt-torch/./ex2.py:9: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3675.)
  print(t1.T.shape)
torch.Size([16384, 8, 2304])
torch.Size([2304, 16384, 8])
Traceback (most recent call last):
'''

# working dims (printout)
#torch.Size([2304, 8, 16384])
#torch.Size([2304, 16384, 8])

#torch.Size([2304, 8, 8])

# transpose_mat1 1 
# transpose_mat2 0 
# m 2304 n 8 k 16384
# mat1_ld 16384 mat2_ld 16384 result_ld 2304 
# abcType 14 computeType 2 scaleType 0
