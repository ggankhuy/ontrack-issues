# not working
PW='\#3paleHorse5\#'
USER=master

# working ok.
PW=amd1234
USER=root

DATE=`date +%Y%m%d-%H-%M-%S`
OUTPUT_DIR=fba-212-$DATE
CONFIG_IP_GUEST=10.216.64.100
CONFIG_PORT_GUEST=30115
LOOP_COUNT=3
SINGLE_BAR="-----------------------------"
DOUBLE_BAR="============================="
function wait_host_up() {
    sleep 3
    for i in {1..60} ; do
        echo -ne "."
        ping -c 1 -W 3 $CONFIG_IP_GUEST
        if  [[ $? -eq 0 ]] ; then
            echo "Host $CONFIG_IP_GUEST is back online."
            return 
        fi
    done
    echo "Host is not unreachable after reboot or not reachable at all."
    exit 1
}

#                "sudo reboot" \
#                "sudo dmesg | sudo tee fba-212/dmesg.after.reboot.'$loop'.log" \
#                "/usr/local/bin/run_kfdtest.sh 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
#                "/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
#                "sudo /opt/rocm-5.0.1/rvs/rvs -c /opt/rocm-5.0.1/rvs/conf/gst_single.conf 2>&1 | sudo tee fba-212/$loop.rvs.gst.test.log" \

wait_host_up
for loop in {1..10} ; do
    echo $DOUBLE_BAR
    echo "Current loop: $loop........"

    for cmd in  "sudo mkdir fba-212" \
                "sudo dmesg --clear"  "sudo modprobe amdgpu" \
                "sudo dmesg | sudo tee fba-212/dmesg.after.modprobe.amdgpu.'$loop'.log" \
                "sudo dmesg --clear" \
                "/usr/local/bin/kfdtest --gtest_filter=-*LargestSysBuffero* 2>&1 | sudo tee fba-212/$loop.kfdtest.log " \
                "sudo dmesg | sudo tee fba-212/dmesg.after.workload.'$loop'.log" \
                "sudo dmesg --clear" "sudo rmmod amdgpu" "sudo dmesg | sudo tee fba-212/dmesg.after.rmmod.'$loop'.log" \
        ; do
        echo "$SINGLE_BAR"
        echo --- $cmd ---
        sshpass -p $PW ssh -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST $cmd
        ret=$?
        sleep 3
        
        #if [[ $ret != 0 && $ret != 1 ]] ; then 
        #    echo "code: $ret. Error executing $cmd on remote $CONFIG_IP_GUEST with credentials: $USER/$PW"
        #    exit 1
        #fi
        sleep 1

        if [[ "$cmd" == "sudo reboot" ]] ; then
            echo "Reboot issued. Waiting for system to become online"
            wait_host_up
        else
            echo "not a reboot cmd"
        fi
    done
done

sudo sshpass -p $PW scp  -p $CONFIG_PORT_GUEST -o StrictHostKeyChecking=no -r $USER@$CONFIG_IP_GUEST:~/fba-212 $OUTPUT_DIR
