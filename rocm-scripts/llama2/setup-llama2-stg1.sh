# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 
SUDO=sudo
MINICONDA_SRC_DIR=/home/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/home/miniconda3/bin/conda
/home/miniconda3/envs

# if user is empty, then assuming it is root, this could be a problem
# if cases is not. 

if [[ -z $USER ]] ; then
    USER=root
    CONDA_ENV_PATH=$MINICONDA_SRC_DIR/envs
else
    CONDA_ENV_PATH=$HOME/.conda/envs
fi

CONDA_ENV_NAME="llama2-$USER-new"
BASHRC=~/.bashrc

if [[ $SUDO_USER ]] ; then
    echo "Do not run as sudo user. Instead all sudo required commands are issued directly from this script."
    exit 1
fi

if [[ ! -d $MINICONDA_SRC_DIR ]] ; then
    $SUDO mkdir -p $MINICONDA_SRC_DIR
    $SUDO wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
    $SUDO chmod 755 ./miniconda.sh
    $SUDO bash ./miniconda.sh -b -u -p /$MINICONDA_SRC_DIR
    $SUDO rm -rf ./miniconda.sh
else
    echo "$MINICONDA_SRC_DIR exists. Assuming installed, bypassing installation..."
fi


if [[ -z `cat $BASHRC | egrep "export.*$MINICONDA_SRC_DIR/bin"` ]] ; then
    echo "export PATH=$PATH:/$MINICONDA_SRC_DIR/bin" | sudo tee -a $BASHRC
fi

# This is done so that to isolated environments are created on /home partition than / partition.
# By default Centos 9 stream would create only 70B / and huge space on /home.

ln -s $MINICONDA_SRC_DIR /$HOME/

if [[ -z `cat $BASHRC | egrep "export CONDA_ENV_NAME"` ]] ; then
    echo  "export CONDA_ENV_NAME=$CONDA_ENV_NAME" | sudo tee -a $BASHRC
fi

$CONDA create python==3.9 --name $CONDA_ENV_NAME  -y
$CONDA init
echo "conda init" >> $BASHRC
echo "conda activate $CONDA_ENV_NAME" >> $BASHRC
echo "Conda envs created..."
$CONDA info --env
