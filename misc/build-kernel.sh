CONFIG_REMOVE_EXISTING_KERNELS=0
CONFIG_BUILD_BINRPM=0
    
KERNEL_VERSION=5.11

if [[ $CONFIG_REMOVE_EXISTING_KERNELS -eq 1 ]] ; then
    echo "Will remove all existing $KERNEL_VERSION kernels!.. ctrl + c to discontinue..."
    #sleep 5
    KERNELS=`rpm -q kernel | grep $KERNEL_VERSION`
    #KERNELS=`ls -l | grep kernel`

    for i in $KERNELS; do
        echo "Removing $i..."
        yum remove $i -y
        ls -l /boot/initram*$KERNEL_VERSION*
        rm -rf /boot/initram*$KERNEL_VERSION*
    done
else    
    echo "Will not remove existing $KERNEL_VERSION kernels..."
fi

if [[ `cat .config | grep CONFIG_LOCALVERSION | grep fbk`  ]] ; then
    echo "building fbk kernel..."
    RPM_FOLDER=/home/nonroot/gg/rpm/rock-kernel/config.meta
else
    echo "building rock kernel..."
    RPM_FOLDER=/home/nonroot/gg/rpm/rock-kernel/config.amd
fi

mkdir $RPM_FOLDER/1/ -p
mkdir $RPM_FOLDER/2/ -p
rm -rf $RPM_FOLDER/1/*
rm -rf $RPM_FOLDER/2/*

if [[ $CONFIG_BUILD_BINRPM -eq 1 ]] ; then
    echo "Will build binrpm ..."
    rm -rf /root/rpmbuild/ ; 
    make clean ; yes "" | make -j`nproc` binrpm-pkg
    mv /root/rpmbuild/RPMS/x86_64/kernel-*  $RPM_FOLDER/1
    ls -l $RPM_FOLDER/1
else
    echo "Will NOT build binrpm ..."
fi

rm -rf /root/rpmbuild/ ; 
make clean ; make -j`nproc` rpm-pkg
mv /root/rpmbuild/RPMS/x86_64/*  $RPM_FOLDER/2
tree -f $RPM_FOLDER/
echo "Build files stored at:  $RPM_FOLDER/"

exit 0
pushd $RPM_FOLDER/2
yum install  ./kernel*.rpm -y
#static commands: 
#yum install -y kernel-5.11.0-13.x86_64.rpm OK...complaints about dkms not able to file header/src file.
#yum install -y kernel-devel-5.11.0-13.x86_64.rpm OK...
#yum install -y kernel-headers-5.11.0-13.x86_64.rpm

rpm -q kernel

KERNEL_STRING=""
pushd /usr/src/
dkms install amdgpu/5.16.9.22.20-1455861.el8/ -k $KERNEL_STRING/x86_64


