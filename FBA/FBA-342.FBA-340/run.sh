RDC_REPO_ROOT=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc.ga/
LD_LIBRARY_PATH=$RDC_REPO_ROOT/ras_prebuild/:$RDC_REPO_ROOT/build/rdc_libs/rdc_modules/rdc_rvs:$RDC_REPO_ROOT/build/rdc_libs/rdc_modules/rdc_rocr/ ./a.out

#root@localhost:~/extdir/gg/git/fba/FBA/FBA-342.FBA-340# find ~/extdir/gg/git/codelab-scripts/ -name librdc_ras.so
#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc.ga/ras_prebuild/librdc_ras.so
#root@localhost:~/extdir/gg/git/fba/FBA/FBA-342.FBA-340# find ~/extdir/gg/git/codelab-scripts/ -name librdc_rvs.so
#/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/rdc.ga/build/rdc_libs/rdc_modules/rdc_rvs/librdc_rvs.so
