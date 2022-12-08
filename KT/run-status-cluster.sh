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

#    if [[ $counter -eq 0 ]] ; then
#        counter=$((counter+1))
#        echo "bypassing 1st node."
#        continue
#    fi

#   get rocm version.
#   echo sshpass -p ${NODE_PWS[$counter]} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 ${NODE_USERS[$counter]}@$i "sudo cat /opt/rocm-*/.info/version"    

#   check No. of threads running.
    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "pgrep ib_write | wc -l ; netstat -in ; dmesg" | tee $LOG_FOLDER/perftest.stat.all.$DATE.$i.log &
#    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "pgrep ib_write | wc -l" | tee $LOG_FOLDER/perftest.no.of.threads.$DATE.$i.log
#    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "netstat -in" | tee $LOG_FOLDER/netstat.$DATE.$i.log
#    sshpass -p amd1234 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$i "dmesg" | tee $LOG_FOLDER/dmesg.$DATE.$i.log
    counter=$((counter+1))
done

