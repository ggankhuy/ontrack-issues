## Compile-only command
mkdir log
hipcc -v -x hip -I/opt/rocm/include -I/opt/rocm-static//include -O3 -c -fgpu-rdc -o onerank.o onerank.cpp 2>&1 | tee log/FBA-396.workaround.compile.log


## Link command
hipcc -v /opt/rocm-static//lib/librccl.a -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -lrocm_smi64 -fgpu-rdc onerank.o -o onerank.out 2>&1 | tee log/FBA-396.workaround.link.log
