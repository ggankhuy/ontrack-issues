ATE=`date +%Y%m%d-%H-%M-%S`
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

#    if [[ $counter -eq 0 ]] ; then
#        counter=$((counter+1))
#        echo "bypassing 1st node."
#        continue
#    fi

#   get rocm version.

#   echo sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo cat /opt/rocm-*/.info/version"    

#   get rccl version. 

#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo cat /opt/rocm-*/.info/version"    
#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "tree -f /opt | grep librccl -i"    

#   run rccl.

#   mkdir -p /log/$i
#   sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "mkdir -p /log/SMCA-21/"    
#   sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "pushd /root/gg/git/rccl-tests/build/; NCCL_DEBUG=INFO LD_LIBRARY_PATH=/home/AMD/rccl/build/:/opt/rocm/lib NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_NET_GDR_LEVEL=3 NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_2:14 sudo ./all_reduce_perf -b 8 -e 16G -f 2 -g 8 -c 0 2>&1 | tee /log/SMCA-21/all_reduce_perf.log ; dmesg | sudo tee /log/SMCA-21/all_reduce_perf.dmesg.log"    
#   sshpass -p amd1234 scp -C -v -r -o StrictHostKeyChecking=no root@$i:/log/SMCA-21 /log/$i/

#   copy files to each node.

#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "mkdir ~/transit"
#   sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/KT.SMCA-21/ ${NODE_USERS[$counter]}@$i:~/transit/

#   install rccl/rccl-devel, checkout rccl-test

#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo yum install rccl-devel -y ; cd ; git checkout https://github.com/ROCmSoftwarePlatform/rccl-tests.git rccl-tests ; cd rccl-tests ; ./install.sh"

#   run rccl-tests:/all_reduce_perf

#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "cd ~/rccl-tests/build; NCCL_DEBUG=INFO LD_LIBRARY_PATH=/home/AMD/rccl/build/:/opt/rocm/lib NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_NET_GDR_LEVEL=3 NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_4:1,mlx5_6:1,mlx5_8:1,mlx5_10:1,mlx5_12:1,mlx5_2:14 ./all_reduce_perf -b 8 -e 16G -f 2 -g 8 -c 0 2>&1 | sudo tee /log/SMCA-21/all_reduce_perf.log"
#   sshpass -p amd1234 scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@$i:/log/SMCA-21 /log/$i/

#   copy perftest to each node.

#   echo sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest ${NODE_USERS[$counter]}@$i:/home/AMD/
#   sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest ${NODE_USERS[$counter]}@$i:/home/AMD/
#   sshpass -p amd1234 scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest root@$i:~
#   echo sshpass -p amd1234 scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest root@$i:~

#   copy perftest. working (root)
#   sshpass -p amd1234 scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest root@$i:~

#   copy perftest. working (amd)    
#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo rm -rf ~/perftest"
#   echo sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest ${NODE_USERS[$counter]}@$i:~
#   sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/perftest ${NODE_USERS[$counter]}@$i:~
#   counter=$((counter+1))

#   copy KT files. working

#    sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo rm -rf ~/KT.SMCA-21"
#    sshpass -p ${NODE_PWS[$counter]} scp -C -r -o StrictHostKeyChecking=no -o ConnectTimeout=10 /home/master/transit/KT.SMCA-21 ${NODE_USERS[$counter]}@$i:~/

#   run disacs.

    sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo chmod 755 ~/KT.SMCA-21/dis_acs.sh ; sudo ~/KT.SMCA-21/dis_acs.sh"
    RET_HOSTNAME=`sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "hostname"`
    RET_HOSTNAME=`echo $RET_HOSTNAME | cut -d '.' -f1 | cut -d '-' -f2`
    echo $RET_HOSTNAME: $RET_HOSTNAME
    sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo ~/KT.SMCA-21//netconfig/ctr-ubbsmc-net-scripts.tar/ctr-ubbsmc-net-scripts/roce_iprouting.$RET_HOSTNAME.sh"

#   run netconfig script.
#   run recreate script.

#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "pushd ~/perftest ; sudo dmesg --clear ; sudo nohup ./recreate_test.sh &"

#   ping test
#   ping -c 4 -W 5 $i 

#   set root password.
#   sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "echo amd1234 | sudo passwd root --stdin"
#   sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "ls -l /opt"
#   sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "sudo reboot"
#   check No. of threads running.
    counter=$((counter+1))
done

