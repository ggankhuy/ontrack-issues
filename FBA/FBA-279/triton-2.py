from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_xdwang/ji/cjivs4encceskkc525nwivnfuv5dk53uzdmkdwyfmwj5vg7cuqdv.py
# Source Nodes: [input_2], Original ATen: [aten.native_dropout]
# input_2 => gt, mul, mul_1
triton_poi_fused_native_dropout_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 10
    tmp6 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.2
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.25
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_xdwang/6w/c6wfhlykuebg6quufewn3d654fewmicbhkso22zv5ww62ck2w6up.py
# Source Nodes: [input_24], Original ATen: [aten.native_dropout]
# input_24 => gt_11, mul_22, mul_23
triton_poi_fused_native_dropout_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_dropout_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, load_seed_offset, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x1 = xindex % 10
    tmp6 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp7 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = 0.1
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp5 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(in_out_ptr0 + (x0), tmp11, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25 = args
    args.clear()
    assert_size_stride(primals_1, (10, 10), (10, 1))
    assert_size_stride(primals_2, (10, ), (1, ))
    assert_size_stride(primals_3, (10, 10), (10, 1))
    assert_size_stride(primals_4, (10, ), (1, ))
    assert_size_stride(primals_5, (10, 10), (10, 1))
    assert_size_stride(primals_6, (10, ), (1, ))
    assert_size_stride(primals_7, (10, 10), (10, 1))
    assert_size_stride(primals_8, (10, ), (1, ))
    assert_size_stride(primals_9, (10, 10), (10, 1))
    assert_size_stride(primals_10, (10, ), (1, ))
    assert_size_stride(primals_11, (10, 10), (10, 1))
    assert_size_stride(primals_12, (10, ), (1, ))
    assert_size_stride(primals_13, (10, 10), (10, 1))
    assert_size_stride(primals_14, (10, ), (1, ))
    assert_size_stride(primals_15, (10, 10), (10, 1))
    assert_size_stride(primals_16, (10, ), (1, ))
    assert_size_stride(primals_17, (10, 10), (10, 1))
    assert_size_stride(primals_18, (10, ), (1, ))
    assert_size_stride(primals_19, (10, 10), (10, 1))
    assert_size_stride(primals_20, (10, ), (1, ))
    assert_size_stride(primals_21, (10, 10), (10, 1))
    assert_size_stride(primals_22, (10, ), (1, ))
    assert_size_stride(primals_23, (10, 10), (10, 1))
    assert_size_stride(primals_24, (10, ), (1, ))
    assert_size_stride(primals_25, (10, 10), (10, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(primals_25, reinterpret_tensor(primals_1, (10, 10), (1, 10), 0), out=buf0)
        del primals_1
        buf1 = empty_strided((12, ), (1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [12], out=buf1)
        buf3 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf4 = buf0; del buf0  # reuse
        # Source Nodes: [input_2], Original ATen: [aten.native_dropout]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_native_dropout_0.run(buf4, buf1, primals_2, buf3, 0, 100, grid=grid(100), stream=stream0)
        del primals_2
        buf5 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf4, reinterpret_tensor(primals_3, (10, 10), (1, 10), 0), out=buf5)
        buf7 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf8 = buf5; del buf5  # reuse
        # Source Nodes: [input_4], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf8, buf1, primals_4, buf7, 1, 100, grid=grid(100), stream=stream0)
        del primals_4
        buf9 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(primals_5, (10, 10), (1, 10), 0), out=buf9)
        buf11 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf12 = buf9; del buf9  # reuse
        # Source Nodes: [input_6], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf12, buf1, primals_6, buf11, 2, 100, grid=grid(100), stream=stream0)
        del primals_6
        buf13 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf12, reinterpret_tensor(primals_7, (10, 10), (1, 10), 0), out=buf13)
        buf15 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf16 = buf13; del buf13  # reuse
        # Source Nodes: [input_8], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf16, buf1, primals_8, buf15, 3, 100, grid=grid(100), stream=stream0)
        del primals_8
        buf17 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf16, reinterpret_tensor(primals_9, (10, 10), (1, 10), 0), out=buf17)
        buf19 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf20 = buf17; del buf17  # reuse
        # Source Nodes: [input_10], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf20, buf1, primals_10, buf19, 4, 100, grid=grid(100), stream=stream0)
        del primals_10
        buf21 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf20, reinterpret_tensor(primals_11, (10, 10), (1, 10), 0), out=buf21)
        buf23 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf24 = buf21; del buf21  # reuse
        # Source Nodes: [input_12], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf24, buf1, primals_12, buf23, 5, 100, grid=grid(100), stream=stream0)
        del primals_12
        buf25 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf24, reinterpret_tensor(primals_13, (10, 10), (1, 10), 0), out=buf25)
        buf27 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf28 = buf25; del buf25  # reuse
        # Source Nodes: [input_14], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf28, buf1, primals_14, buf27, 6, 100, grid=grid(100), stream=stream0)
        del primals_14
        buf29 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf28, reinterpret_tensor(primals_15, (10, 10), (1, 10), 0), out=buf29)
        buf31 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf32 = buf29; del buf29  # reuse
        # Source Nodes: [input_16], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf32, buf1, primals_16, buf31, 7, 100, grid=grid(100), stream=stream0)
        del primals_16
        buf33 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf32, reinterpret_tensor(primals_17, (10, 10), (1, 10), 0), out=buf33)
        buf35 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf36 = buf33; del buf33  # reuse
        # Source Nodes: [input_18], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf36, buf1, primals_18, buf35, 8, 100, grid=grid(100), stream=stream0)
        del primals_18
        buf37 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf36, reinterpret_tensor(primals_19, (10, 10), (1, 10), 0), out=buf37)
        buf39 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf40 = buf37; del buf37  # reuse
        # Source Nodes: [input_20], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf40, buf1, primals_20, buf39, 9, 100, grid=grid(100), stream=stream0)
        del primals_20
        buf41 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf40, reinterpret_tensor(primals_21, (10, 10), (1, 10), 0), out=buf41)
        buf43 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf44 = buf41; del buf41  # reuse
        # Source Nodes: [input_22], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_0.run(buf44, buf1, primals_22, buf43, 10, 100, grid=grid(100), stream=stream0)
        del primals_22
        buf45 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf44, reinterpret_tensor(primals_23, (10, 10), (1, 10), 0), out=buf45)
        buf47 = empty_strided((10, 10), (10, 1), device='cuda', dtype=torch.bool)
        buf48 = buf45; del buf45  # reuse
        # Source Nodes: [input_24], Original ATen: [aten.native_dropout]
        triton_poi_fused_native_dropout_1.run(buf48, buf1, primals_24, buf47, 11, 100, grid=grid(100), stream=stream0)
        del buf1
        del primals_24
        return (buf48, primals_25, buf3, buf4, buf7, buf8, buf11, buf12, buf15, buf16, buf19, buf20, buf23, buf24, buf27, buf28, buf31, buf32, buf35, buf36, buf39, buf40, buf43, buf44, buf47, reinterpret_tensor(primals_23, (10, 10), (10, 1), 0), reinterpret_tensor(primals_21, (10, 10), (10, 1), 0), reinterpret_tensor(primals_19, (10, 10), (10, 1), 0), reinterpret_tensor(primals_17, (10, 10), (10, 1), 0), reinterpret_tensor(primals_15, (10, 10), (10, 1), 0), reinterpret_tensor(primals_13, (10, 10), (10, 1), 0), reinterpret_tensor(primals_11, (10, 10), (10, 1), 0), reinterpret_tensor(primals_9, (10, 10), (10, 1), 0), reinterpret_tensor(primals_7, (10, 10), (10, 1), 0), reinterpret_tensor(primals_5, (10, 10), (10, 1), 0), reinterpret_tensor(primals_3, (10, 10), (10, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((10, 10), (10, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
