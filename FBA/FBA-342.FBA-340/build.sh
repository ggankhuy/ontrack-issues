set -x
# for rdc installed through rpm
#LD_LIBRARY_PATH=/opt/rocm/lib/ hipcc -I/opt/rocm/include/rdc/ -lrdc -lrdc_bootstrap rdc-sample.cpp

# for rdc built from source.
RDC_REPO_ROOT=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/

#echo -e "$RDC_REPO_ROOT/build/rdc_libs/bootstrap/" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
#echo -e "$RDC_REPO_ROOT/build/rdc_libs/rdc/" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
#ldconfig

# not working
#export LD_LIBRARY_PATH=$RDC_REPO_ROOT/build/rdc_libs/bootstrap/:$RDC_REPO_ROOT/build/rdc_libs/rdc/
#export LD_LIBRARY_PATH=$RDC_REPO_ROOT/build/rdc_libs/bootstrap/
#echo $LD_LIBRARY_PATH

hipcc \
/$RDC_REPO_ROOT/build/rdc_libs/bootstrap/librdc_bootstrap.so \
$RDC_REPO_ROOT/build/rdc_libs/rdc/librdc.so -I$RDC_REPO_ROOT/include/ rdc-sample.cpp
hipcc \
/$RDC_REPO_ROOT/build/rdc_libs/bootstrap/librdc_bootstrap.so \
$RDC_REPO_ROOT/build/rdc_libs/rdc/librdc.so -I$RDC_REPO_ROOT/include/ rdc-sample.cpp -ggdb -o a.dbg.out
#hipcc -L/$RDC_REPO_ROOT/build/rdc_libs/bootstrap/:$RDC_REPO_ROOT/build/rdc_libs/rdc/ -I$RDC_REPO_ROOT/include/ -lrdc -lrdc_bootstrap rdc-sample.cpp


#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/include/rdc/rdc.h

#root@localhost:~/extdir/gg/git/fba/FBA/FBA-342.FBA-340# find ~/extdir/gg/git/codelab-scripts/ -name "librdc_bootstrap.so"
#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/build/rdc_libs/bootstrap/librdc_bootstrap.so
#root@localhost:~/extdir/gg/git/fba/FBA/FBA-342.FBA-340# find ~/extdir/gg/git/codelab-scripts/ -name "librdc.so"
#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/build/rdc_libs/rdc/librdc.so
#root@localhost:~/extdir/gg/git/fba/FBA/FBA-342.FBA-340# find ~/extdir/gg/git/codelab-scripts/ -name "librdc_rvs.so"
#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc/build/rdc_libs/rdc_modules/rdc_rvs/librdc_rvs.so
