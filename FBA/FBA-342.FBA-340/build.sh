set -x
# for rdc installed through rpm
#LD_LIBRARY_PATH=/opt/rocm/lib/ hipcc -I/opt/rocm/include/rdc/ -lrdc -lrdc_bootstrap rdc-sample.cpp

# for rdc built from source.
RDC_REPO_ROOT=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/

hipcc \
/$RDC_REPO_ROOT/build/rdc_libs/bootstrap/librdc_bootstrap.so \
$RDC_REPO_ROOT/build/rdc_libs/rdc/librdc.so -I$RDC_REPO_ROOT/include/ rdc-sample.cpp
hipcc \
/$RDC_REPO_ROOT/build/rdc_libs/bootstrap/librdc_bootstrap.so \
$RDC_REPO_ROOT/build/rdc_libs/rdc/librdc.so -I$RDC_REPO_ROOT/include/ rdc-sample.cpp -ggdb -o a.dbg.out
