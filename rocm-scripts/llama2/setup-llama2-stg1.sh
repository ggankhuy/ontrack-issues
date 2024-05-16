# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 
#SUDO=sudo
MINICONDA_SRC_DIR=/home/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/home/miniconda3/bin/conda

if [[ -z $SUDO_USER ]] ; then
    CURR_USER=$USER
else
    CURR_USER=$SUDO_USER
fi

if [[ ! -d $MINICONDA_SRC_DIR ]] ; then
    mkdir -p $MINICONDA_SRC_DIR
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
    chmod 755 ./miniconda.sh
    bash ./miniconda.sh -b -u -p /$MINICONDA_SRC_DIR
    rm -rf ./miniconda.sh
else
    echo "$MINICONDA_SRC_DIR exists. Assuming installed, bypassing installation..."
fi

if [[  -z $SUDO_USER ]] ; then
    BASHRC=~/.bashrc
else
    BASHRC=/home/$SUDO_USER/.bashrc
fi
if [[ -z `cat $BASHRC | egrep "export.*$MINICONDA_SRC_DIR/bin"` ]] ; then
    echo "export PATH=$PATH:/$MINICONDA_SRC_DIR/bin" | sudo tee -a $BASHRC
fi

# This is done so that to isolated environments are created on /home partition than / partition.
# By default Centos 9 stream would create only 70B / and huge space on /home.

ln -s $MINICONDA_SRC_DIR /$HOME/

CONDA_ENV_NAME="llama2-nonroot"

$CONDA create --name  $CONDA_ENV_NAME python==3.9 -y
if [[  -z $SUDO_USER ]] ; then
    $CONDA init
else
    runuser -l nonroot -c "$CONDA init"
fi
echo "conda activate $CONDA_ENV_NAME" >> $BASHRC
echo "Conda envs created..."
$CONDA info --env
