# Run this when kernel-devel package for $KERNEL_VERSION is installed and module $VERSION  is copied to /usr/src.

VERSION=amdgpu/5.16.9.22.20-1423825.el8
#KERNEL_VERSION=5.12.0-0_fbk5_zion_rc1_5697_g2c723fb88626
KERNEL_VERSION=5.11.0

#VERSION=amdgpu/5.13
#KERNEL_VERSION=5.13
#KERNEL_VERSION=5.13.0-000.el8.x86_64

#dkms remove  $VERSION -k $(uname -r)/x86_64
#sudo dkms add $VERSION
#dkms build $VERSION -k $(uname -r)/x86_64 --kernelsourcedir=/usr/src/kernels/$KERNEL_VERSION
#dkms install  $VERSION -k $(uname -r)/x86_64

LOG_FOLDER=./log/
mkdir -p $LOG_FOLDER
LOG_FILE_BASE=$LOG_FOLDER/dkms.amkdgpu

for action in remove add build install 
    do dkms $action $VERSION  -k $KERNEL_VERSION/x86_64 2>&1 | tee $LOG_FILE_BASE.$action.log
done
