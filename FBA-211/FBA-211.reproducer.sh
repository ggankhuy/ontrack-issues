DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=./log/$DATE
mkdir -p $LOG_FOLDER

for i in {1..10} ; do 
    echo "========================="
    echo "loop $i..."

    dmesg --clear
    modprobe amdgpu
    echo "dmesg after modprobe amdgpu..." 
    dmesg | tee $LOG_FOLDER/dmesg.modprobe.amdgpu.$i.log

    ./gpuburn-hip 2>&1 | tee  $LOG_FOLDER/gpuburn-hip.$i.log
    dmesg | tee $LOG_FOLDER/dmesg.after.apprun.$i.log
    dmesg --clear
    
    sleep 5
    modprobe -r amdgpu 

    r=$?  
    echo "r: $r /modprobe -r amdgpu/" 
    
    if [[ $r -ne 0 ]] ; then
        echo "modprobe -r amdgpu returned non-zero. Exiting prematurely."
        exit 1
    fi
    echo "dmesg after modprobe -r amdgpu..." 
    dmesg | tee $LOG_FOLDER/dmesg.modprobe-r.amdgpu.$i.log
done
