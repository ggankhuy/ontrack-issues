#LD_LIBRARY_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/build/rdc_libs/bootstrap/:/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/build/rdc_libs/rdc/ ./a.out --device 0 --mode=range --htod --iters=50000 --start=100000000 --end=100000000 --increment=1 2>&1 | tee run.log
#when debian is installed all libraries will be in  /opt/rocm/lib/rdc/
LD_LIBRARY_PATH=/opt/rocm/lib/rdc ./a.out --device 0 --mode=range --htod --iters=50000 --start=100000000 --end=100000000 --increment=1 2>&1 | tee run.log

