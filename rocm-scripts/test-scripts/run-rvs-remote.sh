vmIp=10.6.189.83
pw=val
user=root
for i in {1..10} ; do
    echo "loop $i..."
    ping -c 4 $vmIp

    if [[ $? -ne 0 ]] ; then
        echo "Unable to ping $vmIp "
        exit 1
    else
        echo "ping ok..."
    fi
    sshpass -p $pw ssh -o StrictHostKeyChecking=no $user@$vmIp 'cd /root/gg/ad-hoc-scripts/rocm-scripts; ./run-rvs.sh'
    sshpass -p $pw ssh -o StrictHostKeyChecking=no $user@$vmIp 'reboot'
    sleep 300
done
