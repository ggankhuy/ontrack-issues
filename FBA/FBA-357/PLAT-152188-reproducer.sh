mpirun --allow-run-as-root -map-by ppr:8:node --bind-to numa -mca pml ob1 -mca btl \
^openib -mca btl_tcp_if_include ens14np0 -x NCCL_DEBUG=WARN -x \
LD_PRELOAD=/root/extdir/gg/git/rccl/build/librccl.so \
:$LD_PRELOAD -x NPKIT_DUMP_DIR=npkit_dump/ \
/root/extdir/gg/git/rccl-tests/build/all_reduce_perf -b 16K -e 1M -f 4 -g 1 -n 10 -w 0 -c 0
