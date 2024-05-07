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

# transpose_mat1 1 
# transpose_mat2 0 
# m 2304 n 8 k 16384
# mat1_ld 16384 mat2_ld 16384 result_ld 2304 
# abcType 14 computeType 2 scaleType 0
