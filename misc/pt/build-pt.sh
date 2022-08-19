function install_conda() {
    
    # Anaconda defs.

    FILE_ANACONDA=Anaconda3-2021.11-Linux-x86_64.sh
    PREFIX_ANACONDA=/Anaconda3
    CONFIG_UPGRADE_ANACONDA=1
    EXEC_PATH_CONDA=$PREFIX_ANACONDA/bin/conda

    wget -nc https://repo.anaconda.com/archive/$FILE_ANACONDA
    ret=$? ; echo $ret.
    if [[ $ret -ne 0 ]] ; then
        echo "code: $ret. Download failure for $FILE_ANACONDA, check the url."
        exit 1
    fi
    chmod  777 $FILE_ANACONDA
    mkdir -p $PREFIX_ANACONDA
    if [[ $CONFIG_UPGRADE_ANACONDA=="1" ]] ; then
        echo "Installing/upgrading regardless of existing installation..."
        ./$FILE_ANACONDA  -u -b -p $PREFIX_ANACONDA
    else
        echo "Installing only if it is not installed in $PREFIX_ANACONDA location..."
        ./$FILE_ANACONDA  -b -p $PREFIX_ANACONDA
    fi    


    #if [[ `cat ~/.bashrc | grep PATH | grep $PREFIX_ANACONDA` ]] ; then
    #    echo "Inserting path onto bashrc in case reboot next time. "
    #    echo "export PATH=$PATH:$PREFIX_ANACONDA/bin" >> ~/.bashrc
    #else
    #    echo "conda path is already defined in bashrc."
    #fi
    ln -s $EXEC_PATH_CONDA /usr/bin/conda
    echo "Testing the installation..."
    $EXEC_PATH_CONDA

    if [[ $? -ne 0 ]] ; then
        echo "Can not find $EXEC_PATH_CONDA."
        exit 1
    fi  
}

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
}

# install conda.
echo "Installing conda..."
install_conda

# Upgrading python.
echo "Upgrading python..."
install_python 

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

