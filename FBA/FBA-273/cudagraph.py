#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import time

import torch


def main(enable_cudagraph) -> int:
    N, D_in, H, D_out = 10, 10, 10, 10
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, H),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(H, D_out),
        torch.nn.Dropout(p=0.1),
    ).cuda()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Placeholders used for capture
    static_input = torch.randn(N, D_in, device="cuda")
    static_target = torch.randn(N, D_out, device="cuda")

    # warmup
    # Uses static_input and static_target here for convenience,
    # but in a real setting, because the warmup includes optimizer.step()
    # you must use a few batches of real data.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
            loss.backward()
            optimizer.step()
    torch.cuda.current_stream().wait_stream(s)

    # capture
    if enable_cudagraph:
        g = torch.cuda.CUDAGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g, stream=s):
            static_y_pred = model(static_input)
            static_loss = loss_fn(static_y_pred, static_target)
            static_loss.backward()
            optimizer.step()

    real_inputs = [torch.rand_like(static_input) for _ in range(1000)]
    real_targets = [torch.rand_like(static_target) for _ in range(1000)]

    torch.cuda.synchronize()
    t1 = time.time()
    for data, target in zip(real_inputs, real_targets):
        if enable_cudagraph:
            # Fills the graph's input memory with new data to compute on
            static_input.copy_(data)
            static_target.copy_(target)
            # replay() includes forward, backward, and step.
            # You don't even need to call optimizer.zero_grad() between iterations
            # because the captured backward refills static .grad tensors in place.
            g.replay()
            # Params have been updated. static_y_pred, static_loss, and .grad
            # attributes hold values from computing on this iteration's data.
        else:
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(data)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()

    t2 = time.time()
    total_time = t2 - t1
    print(f"cudagraph: {enable_cudagraph}, time: {total_time}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cudagraph example")
    parser.add_argument("--enable_cudagraph", default=False, action="store_true")
    args, _ = parser.parse_known_args()
    main(args.enable_cudagraph)

