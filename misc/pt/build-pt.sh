source lib.sh

# install conda.
echo "Installing conda..."
#install_conda

# Upgrading python.
echo "Upgrading python..."
install_python 
exit 0

$EXEC_PATH_CONDA install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
cd ..

# building pytorch
echo "Building pytorch..."
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive --jobs 0

pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install --upgrade distlib

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python3 tools/amd_build/build_amd.py 2>&1  | tee build-pytorch.log
#PYTORCH_ROCM_ARCH=gfx908  .jenkins/pytorch/build.sh
USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=$(nproc) python3 -v setup.py install --user

#python setup.py install

