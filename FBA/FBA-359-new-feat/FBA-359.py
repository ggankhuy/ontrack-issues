import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--enable-tuning", action="store_true")
parser.add_argument("--enable-profiler", action="store_true")
parser.add_argument("--enable-cudagraph", action="store_true")
parser.add_argument("--disable-hipblaslt", action="store_true")
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)
args = parser.parse_args()

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
if not args.disable_hipblaslt:
    os.environ["DISABLE_ADDMM_HIP_LT"] = "0"
if args.enable_tuning:
    print("Enabled tuning")
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"


'''
shapes = [
    <fill in the shape here> 
]
'''
dtype_size_mapping = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}

dtype = torch.bfloat16
do_profile = args.enable_profiler
enable_cudagraph = args.enable_cudagraph
dtype_size = dtype_size_mapping[dtype]

results = {}

shapes = [
[8, 8192]
]

for n, k in shapes:
    print("n/k: ", n, "/", k)
    m = args.batch_size
    print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
    inp = torch.randn((m, k), dtype=dtype, device="cuda")
    weights = torch.randn((n, k), dtype=dtype, device="cuda")

    if enable_cudagraph:
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        # this may be needed for tuning
        ref = inp @ weights.T
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.graph(g, stream=s):
            for _ in range(100):
                ref = inp @ weights.T

    ## Run warmup
    if not enable_cudagraph:
        for _ in range(20):
            # ref = F.linear(inp, weights)
            ref = inp @ weights.T

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    if do_profile:
        torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        torch_profiler.start()

    start_event.record()
    if enable_cudagraph:
        g.replay()
    else:
        for _ in range(100):
            ref = inp @ weights.T

    stop_event.record()
    torch.cuda.synchronize()
    if do_profile:
        torch_profiler.stop()
        torch_profiler.export_chrome_trace(f"{m}_{n}_{k}.json")
    elapsed = start_event.elapsed_time(stop_event)
    ms = elapsed / 100
    us = ms * 1000

    def compute_FC_flops(m, n, k):
        flops = m * n * k * 2
        return flops

    def compute_total_bytes(m, n, k):
        return (
            dtype_size * m * n
            + dtype_size * n * k
            + dtype_size_mapping[torch.float32] * m * k
        )

    flops = compute_FC_flops(m, n, k) / (ms / 1e3)
    bw = compute_total_bytes(m, n, k) / (ms / 1e3)
    print(
        "Avg time: {} us, Achieved {:.2f} TFLOPS, {:.2f} GB/s\n".format(
            us, flops / 1e12, bw / 1e9
        )
    )
    results[f"{m}x{n}x{k}-{dtype}"] = [us, flops / 1e12, bw / 1e9]

for config, result in results.items():
    out_str = f"{config}"
    for i in result:
        out_str += f",{i}"

    print(out_str)
