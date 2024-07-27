	
import argparse
import multiprocessing as mp
import os
import tempfile
import uuid

import torch
from torch.distributed.launcher.api import elastic_launch, LaunchConfig


parser = argparse.ArgumentParser()
parser.add_argument("--enable-profiler", action="store_true", default=True)
parser.add_argument("--enable-cudagraph", action="store_true", default=True)
parser.add_argument(
    "--message_size", type=int, default=65536
)  # default 64K message size
parser.add_argument("--iter", type=int, default=10000)

args = parser.parse_args()

enable_cudagraph = args.enable_cudagraph
do_profile = args.enable_profiler
N = args.message_size // 2  # number of elements using bf16
ITER = args.iter

torch.ops.load_library("//gen_ai/llm_inference/fb/llm/csrc:llama_cpp") 
## AMD side: you may change this line and below to call msccl and the oneshot 
## kernel that Cen shared with you to do the microbenchmark



def run_all_reduce(path, results):
    rank = int(os.environ["LOCAL_RANK"])
    W = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.ops.llama_cpp.nccl_init(rank, W, os.path.join(path, "rdvz"))

    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        init_method=f"file://{os.path.join(path, 'gloo_rdvz')}",
        world_size=W,
        rank=rank,
    )

    buffer = torch.ops.llama_cpp.car_tensor()
    barrier = torch.ops.llama_cpp.car_tensor()
    barrier.zero_()

    buffer_handle = torch.ops.llama_cpp.car_ipc_handle(buffer)
    all_buffer_handles = [torch.empty_like(buffer_handle) for _ in range(W)]
    torch.distributed.all_gather(all_buffer_handles, buffer_handle)

    barrier_handle = torch.ops.llama_cpp.car_ipc_handle(barrier)
    all_barrier_handles = [torch.empty_like(barrier_handle) for _ in range(W)]
    torch.distributed.all_gather(all_barrier_handles, barrier_handle)

    torch.ops.llama_cpp.car_init(
        rank, W, barrier, all_barrier_handles, buffer, all_buffer_handles
    )
    torch.cuda.synchronize()
    torch.distributed.barrier()

    print(f"===== Running {N} on rank {rank}")
    y = torch.zeros(size=(N,), dtype=torch.bfloat16, device="cuda")
    y_allreduce = torch.empty_like(y)
    if enable_cudagraph:
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.graph(g, stream=s):
            for _ in range(ITER):
                torch.ops.llama_cpp.one_shot_car_allreduce(y_allreduce, y) ## AMD side: you may change this line and below to call msccl and the oneshot kernel that Cen shared with you to do the microbenchmark

    ## Run warmup
    for _ in range(20):
        torch.ops.llama_cpp.one_shot_car_allreduce(y_allreduce, y)

    ## Actual benchmark
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
        for _ in range(ITER):
            torch.ops.llama_cpp.one_shot_car_allreduce(y_allreduce, y)

    stop_event.record()
    torch.cuda.synchronize()
    if do_profile:
        torch_profiler.stop()
        torch_profiler.export_chrome_trace(
            f"one_shot_car_allreduce_msg_size{args.message_size}_rank{rank}.json"
        )
        elapsed = start_event.elapsed_time(stop_event)
        ms = elapsed / ITER
        us = ms * 1000

        print(f"Avg time: {us} us\n")
        results[f"{rank}"] = us


def invoke_main():
    manager = mp.Manager()
    results = manager.dict()

    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as path:
        lc = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=torch.cuda.device_count(),
            run_id=str(uuid.uuid4()),
            rdzv_backend="c10d",
            rdzv_endpoint=os.path.join(tmpdir, "rdzv"),
            rdzv_configs={"store_type": "file"},
            start_method="spawn",
            monitor_interval=1,
            max_restarts=0,
        )
        elastic_launch(config=lc, entrypoint=run_all_reduce)(path, results)

    print("====== Done running all reduce ====== \n ")
    print(results)


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
