for j in {1..1000} ; do
    for i in {1..48}; do 
        /root/rccl/tools/TransferBench/TransferBench example-1.cfg  &
        /root/rccl/tools/TransferBench/TransferBench example-2.cfg  &
    done
done
