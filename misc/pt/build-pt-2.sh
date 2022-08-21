yum update

for i in wget build-essential git cmake gpg-agent kmod python python3 python3-pip python3-opencv python  
do 
    yum install -y $i 
done

yum install rocm-dev rocm-utils rocm-libs rccl -y
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
python3 -m pip install -r requirements.txt
python3 tools/amd_build/build_amd.py
USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=$(nproc) python3 setup.py install --user
