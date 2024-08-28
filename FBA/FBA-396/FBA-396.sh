set -x 
for i in rccl rccl-tests ; do
    if [[ ! -z $i ]] ; then
        git clone https://github.com/ROCm/$i.git
    fi
done
mkdir log
OPT_STATIC=/opt/rocm-static
pushd rccl
rm -rf build
./install.sh -i --static  --prefix=$OPT_STATIC 2>&1 | tee ../log/install.rccl.static.log
ret=$?
echo ret: $ret
[[ $ret -eq 0 ]] || exit 1
popd

pushd rccl-tests
./install.sh --rccl_home=$OPT_STATIC 2>&1 | tee ../log/install.rccl-tests.static.log
popd

MINI_TEST=onerank.cpp


