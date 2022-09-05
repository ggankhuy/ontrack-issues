source lib.sh

# Upgrading python.
echo "Upgrading python..."
install_python 

# make sure following rocm components are installed.

yum install -y rocblas-devel rocrand-devel hipblas-devel rocfft-devel miopen-hip-devel hipfft-devel hipsparse-devel rocprim-devel hipcub-devel rocthrust-devel rccl-devel 

# building pytorch
echo "Building pytorch..."

git clone --recursive https://github.com/pytorch/pytorch
pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install --upgrade distlib

cd pytorch
git checkout v1.12.1

python3 -m pip install -r requirements.txt
python3 tools/amd_build/build_amd.py
mkdir build ; cd build ; cmake .. ; make -j 64 
cd ..
USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=$(nproc) python3 setup.py install --user
python3 -c "import torch ; cuda = torch.device('cuda')"

