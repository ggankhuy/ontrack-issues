# tested on centos 8 stream ok, needs testing on ubuntu if needed, likely break.
# may need to test as nonroot user.

source lib.sh
yum update -y
yum install -y wget sudo nano tree git dstat
install_python

counter=0
LOG_FOLDER=./log
mkdir -p $LOG_FOLDER
SINGLE_BAR='----'
yum groupinstall "Development Tools" -y | tee $LOG_FOLDER/setup.development.tools.log
pip3 install --upgrade pip 2>&1 | tee $LOG_FOLDER/pip.install.upgrade.log
for i in "python3 -m pip3 install --upgrade pip" "yum update -y"
do
    echo $SINGLE_BAR | tee $LOG_FOLDER/setup.generic.$counter.log
    echo "DBG: executing '$i'..." | tee -a $LOG_FOLDER/setup.generic.$counter.log
    echo $SINGLE_BAR  | tee -a $LOG_FOLDER/setup.generic.$counter.log
    $i 2>&1 | tee -a $LOG_FOLDER/setup.generic.$counter.log
    counter=$((counter+1))
done 

for i in zlib kernel-devel python3-devel
do
    echo $SINGLE_BAR | tee $LOG_FOLDER/setup.yum.$counter.log
    echo "DBG: executing '$i'..." | tee -a $LOG_FOLDER/setup.yum.$counter.log
    echo $SINGLE_BAR  | tee -a $LOG_FOLDER/setup.yum.$counter.log
    yum install -y $i 2>&1 | tee -a $LOG_FOLDER/setup.yum.$counter.log
    counter=$((counter+1))
done 


counter=0
for i in neuralnet matplotlib pandas sklearn Pillow scipy==1.2.0 cloudpickle mlxtend
do
    echo $SINGLE_BAR | tee $LOG_FOLDER/setup.pip3.$counter.log
    echo "DBG: executing '$i'..." | tee -a $LOG_FOLDER/setup.pip3.$counter.log
    echo $SINGLE_BAR  | tee -a $LOG_FOLDER/setup.pip3.$counter.log
    pip3 install $i 2>&1 | tee -a $LOG_FOLDER/setup.pip3.$counter.log
    counter=$((counter+1))
done

#For 4.5.2
#pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2 2>&1  | tee $LOG_FOLDER/torch.torchvision.log

#For 5.1.1
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.1.1

# For 5.2
#pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.2/

