DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=log/$DATE
mkdir -p $LOG_FOLDER

NODE_IPS=(\
    10.7.53.18 10.7.54.186 10.7.53.77 10.7.53.76 \
    10.7.97.225 10.7.97.178 10.7.54.216 10.7.54.210 
    10.7.54.221 10.7.53.55 10.7.53.53 10.7.54.220 \
    10.7.53.54 10.7.53.49 10.7.54.217 10.7.54.222)
NODE_USERS=(\
    AMD amd AMD AMD\
    AMD AMD AMD AMD\
    AMD AMD AMD AMD\
    AMD amd AMD AMD)
NODE_PWS=(\
    "AMD123!" "amd123!" "AMD123!" "AMD123!"\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!"\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!"\
    "AMD123!" "amd123!" "AMD123!" "AMD123!")

counter=0
for i in ${NODE_IPS[@]} ; do
    echo "--------------------------------------------------------------------"
    echo "NODE_IPS/USER/PWS: $i, ${NODE_USERS[$counter]}, ${NODE_PWS[$counter]}"
    echo sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo cat /opt/rocm-*/.info/version"    
    sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo cat /opt/rocm-*/.info/version"    
#    sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "tree -f /opt | grep librccl -i"    

    mkdir -p /log/$i
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "mkdir -p /log/SMCA-21/"    
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "pushd /root/gg/git/rccl-tests/build/; NCCL_DEBUG=INFO LD_LIBRARY_PATH=/home/AMD/rccl/build/:/opt/rocm/lib NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_NET_GDR_LEVEL=3 NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_2:14 sudo ./all_reduce_perf -b 8 -e 16G -f 2 -g 8 -c 0 2>&1 | tee /log/SMCA-21/all_reduce_perf.log ; dmesg | sudo tee /log/SMCA-21/all_reduce_perf.dmesg.log"    
    sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no root@$i:/log/SMCA-21 /log/$i/

    counter=$((counter+1))
    exit 0
done

