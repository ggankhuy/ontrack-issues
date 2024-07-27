set -x 
for i in rccl rccl-test ; do
    if [[ ! -z rccl ]] ; then
        git clone https://github.com/ROCm/$i
    fi
done
