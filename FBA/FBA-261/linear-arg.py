import argparse
import time

import torch
import torch._inductor
import torch.nn as nn

print(torch.backends.cudnn.version())
torch.backends.cuda.matmul.allow_tf32 = True


parser = argparse.ArgumentParser(description="FC Example")
parser.add_argument(
    "--batch_size",
    type=int,
    default=1024,
)

parser.add_argument(
    "--input_size",
    type=int,
    default=1024,
)

parser.add_argument(
    "--output_size",
    type=int,
    default=1024,
)

parser.add_argument(
    "--num_iter",
    type=int,
    default=100,
    help="Number of iterations",
)

parser.add_argument(
    "--warmup_iter",
    type=int,
    default=10,
    help="Number of iterations",
)

parser.add_argument(
    "--dtype",
    default="float",
    choices=["float", "float16", "bfloat16"],
    help="data type",
)

parser.add_argument(
    "--transpose",
    default=False,
    action="store_true",
)

parser.add_argument("--is_training", action="store_true", help="training or inference")

parser.add_argument("--pt2", action="store_true", help="apply torch.compile")

parser.add_argument(
    "--triton_matmul",
    action="store_true",
    help="Only work with PT2 flag: whether to turn on triton matmul or not (cublas)",
)

args, _ = parser.parse_known_args()

batch_size = args.batch_size
input_size = args.input_size
output_size = args.output_size
dtype = args.dtype

is_training = False

total_time = 0.0

dtype = torch.float

dtype_mapping = {
    "float": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

dtype = dtype_mapping[args.dtype]

device = torch.device("cuda:0")
torch.cuda.set_device(device)


def run_linear(
    batch_size: int, input_size: int, output_size: int, dtype, device: torch.device
):
    print(
        "\n* Run Linear {} x {} x {}, dtype = {}".format(
            batch_size, input_size, output_size, dtype
        )
    )
    raw_model = nn.Linear(input_size, output_size, dtype=dtype, device=device)

    @torch.compile(backend="inductor", fullgraph=True)
    def func(input):
        return raw_model(input)

    if args.pt2 or args.triton_matmul:
        print("PT2 compile")
        model = func
    else:
        model = raw_model

    if args.triton_matmul:
        torch._inductor.config.global_cache_dir = None
        torch._inductor.config.max_autotune_gemm = True

    input_data = torch.randn(
        batch_size, input_size, device=device, dtype=dtype, requires_grad=True
    )

    if args.transpose:
        input_data = input_data.t()

    for _ in range(args.warmup_iter):
        # warmup for both fwd and bwd to avoid compilation during benchmarking time
        output = model(input_data)
        output.backward(output)

    torch.cuda.synchronize()
    t1 = time.time()
    if args.is_training:
        for _ in range(args.num_iter):
            output = model(input_data)
            output.backward(output)
    else:
        with torch.no_grad():
            for _ in range(args.num_iter):
                output = model(input_data)

    torch.cuda.synchronize()
    t2 = time.time()
    total_time = t2 - t1

    print(
        "Finished {} {} x {} x {}, {} iterations in {:.2f} us/iter".format(
            "training" if args.is_training else "inference",
            batch_size,
            input_size,
            output_size,
            args.num_iter,
            total_time / args.num_iter * 1e6,
        )
    )

    def compute_FC_flops():
        flops = input_size * output_size * args.num_iter * batch_size
        if args.is_training:
            flops *= 6
        else:
            flops *= 2
        return flops

    flops = compute_FC_flops() / total_time
    print("Achieved {:.2f} TFLOPS".format(flops / 1e12))

run_linear(batch_size, input_size, output_size , dtype, device)
