CONFIG_DMESG_DIR=/log/dmesg/
sudo mkdir -p $CONFIG_DMESG_DIR/
DATE=`date +%Y%m%d-%H-%M-%S`
dmesg | sudo tee $CONFIG_DMESG_DIR/dmesg-$DATE-nokmod.log
sudo dmesg --clear
sudo modprobe amdgpu
dmesg | sudo tee $CONFIG_DMESG_DIR/dmesg-$DATE-kmod.log



