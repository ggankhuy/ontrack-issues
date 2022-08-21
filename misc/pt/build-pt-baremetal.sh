function install_python() {
    #Python defs.

    PYTHON_VER_MAJOR=3.9
    PYTHON_VER_MINOR=10
    PYTHON_VER=$PYTHON_VER_MAJOR.$PYTHON_VER_MINOR
    PYTHON_FULL_NAME=Python-$PYTHON_VER
    PYTHON_TAR=$PYTHON_FULL_NAME.tgz

    sudo yum -y install epel-release
    sudo yum update -y
    sudo yum groupinstall "Development Tools" -y

    sudo yum install openssl-devel libffi-devel bzip2-devel -y
    wget -nc https://www.python.org/ftp/python/$PYTHON_VER/$PYTHON_TAR
    tar -xvf $PYTHON_TAR
    cd $PYTHON_FULL_NAME

    if [[ $? -ne 0 ]] ; then
        echo "Can not cd into $PYTHON_VER directory..."
        exit 1
    fi
    ./configure --enable-optimizations
    sudo make -j`nproc` install

    echo "Testing the installation..."
    python$PYTHON_VER_MAJOR --version
    if [[ $? -ne 0 ]] ; then
        echo "Unable to find 3.9"
    fi
    PATH_PYTHON_U=`which python$PYTHON_VER_MAJOR`
    echo "new path: $PATH_PYTHON_U"
    rm -rf /usr/bin/python
    echo ln -s $PATH_PYTHON_U /usr/bin/python
    ln -s $PATH_PYTHON_U /usr/bin/python
    rm -rf /usr/bin/python3
    ln -s /usr/bin/python /usr/bin/python3
    cd ..
}

# Upgrading python.
echo "Upgrading python..."
install_python 

# make sure following rocm components are installed.

yum install rocm-dev rocm-utils rocm-libs rccl -y

# building pytorch
echo "Building pytorch..."

git clone --recursive https://github.com/pytorch/pytorch

pip3 install --upgrade setuptools
pip3 install --upgrade pip
pip3 install --upgrade distlib

cd pytorch

python3 -m pip install -r requirements.txt
python3 tools/amd_build/build_amd.py
cmake . ; make -j 64
USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 MAX_JOBS=$(nproc) python3 setup.py install --user

python3 -c "import torch ; cuda = torch.device('cuda')"

