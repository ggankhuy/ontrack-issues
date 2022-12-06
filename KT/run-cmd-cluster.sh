NODE_IPS=(\
    10.7.53.18 10.6.189.140 10.7.53.77 10.7.53.76 \
    10.7.97.225 10.7.97.178 10.7.54.216 10.7.54.210 
    10.7.54.221 10.7.53.55 10.7.53.53 10.7.54.220 \
    10.7.53.54 10.7.53.49 10.7.54.217 10.7.54.222)
NODE_USERS=(\
    AMD AMD AMD AMD\
    AMD AMD AMD AMD\
    AMD AMD AMD AMD\
    AMD AMD AMD AMD)
NODE_PWS=(\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!"\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!"\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!"\
    "AMD123!" "AMD123!" "AMD123!" "AMD123!")

counter=0
for i in ${NODE_IPS[@]} ; do
    echo "NODE_IPS/USER/PWS: $i, ${NODE_USERS[$counter]}, ${NODE_PWS[$counter]}"
    counter=$((coutner+1))
done

#RESP=$((sshpass -p amd1234 ssh -o StrictHostKeyChecking=no root@$CONFIG_NODE_IP "virsh list --all | grep -i gpu | wc -l"))
