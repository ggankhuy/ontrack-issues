# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 

SUDO=sudo
MINICONDA_SRC_DIR=/home/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/home/miniconda3/bin/conda
BASHRC=~/.bashrc

if [[ $SUDO_USER ]] ; then
    echo "Do not run as sudo user. Instead all sudo required commands are issued directly from this script."
    exit 1
fi

LOG_DIR=./log
for i in gfortran libomp; do 
    $SUDO yum install $i -y ; 
done
SOFT_LINK=1

if [[ ! -f $LLAMA_PREREQ_PKGS.tar ]] ; then 
    echo "$LLAMA_PREREQ_PKGS.tar does not exist." 
    exit 1
fi
$SUDO tar -xvf  $LLAMA_PREREQ_PKGS.tar
pushd $LLAMA_PREREQ_PKGS
    for i in *tar ; do 
        dirname=`echo $i | awk '{print $1}' FS=. `
        $SUDO mkdir $dirname ; pushd $dirname
        $SUDO ln -s ../$i .
        $SUDO tar -xvf ./$i 
        pip3 install ./*.whl
        popd
    done
popd

conda install mkl-service -y
pip3 install mkl 

$SUDO tar -xf $LLAMA_PREREQ_PKGS.tar
pwd
ls -l 

pushd $LLAMA_PREREQ_PKGS
$SUDO mkdir $LOG_DIR
$SUDO bash install.sh 2>&1 | tee $LOG_DIR/install.log
popd

$SUDO git clone https://bitbucket.org/icl/magma.git
pushd magma

BASHRC_EXPORT=./export.md
ROCM_PATH=/opt/rocm-6.2.0-13611

if [[ ! -d $ROCM_PATH ]] ; then
    echo "ROCM_PATH: $ROCM_PATH does not exist! Can not continue." ; exit 1 
fi

ls -l $BASHRC
if [[ -z `cat $BASHRC | grep "export.*MAGMA_HOME"` ]] ; then
    echo "export MAGMA_HOME=$PWD" | tee -a $BASHRC | tee -a $BASHRC_EXPORT
    export MAGMA_HOME=$PWD
fi

if [[  -z `cat $BASHRC | grep "export.*MKLROOT"` ]] ; then
    echo "export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME" | tee -a $BASHRC | tee -a $BASHRC_EXPORT
    export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME
fi

if [[ -z `cat $BASHRC | grep "export.*ROCM_PATH"` ]] ; then
    export ROCM_PATH=$ROCM_PATH
    echo "export ROCM_PATH=$ROCM_PATH" | tee -a $BASHRC | tee -a $BASHRC_EXPORT
fi

$SUDO cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" | $SUDO tee -a make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" | $SUDO tee -a make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" | $SUDO tee -a make.inc
# build MAGMA
$SUDO make -f make.gen.hipMAGMA -j
$SUDO HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j 2>&1 | $SUDO tee $LOG_DIR/make.magma.log
popd

pushd $LLAMA_PREREQ_PKGS

if [[ $SOFT_LINK == 1 ]] ; then
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        $SUDO ln -s \
        $HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
else
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        rm -rf $HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
        cp \
        $HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
fi

$SUDO chmod 755 *sh
echo "Use following cmd to run:"

#/home/nonroot/.conda/envs/llama2-nonroot-2/lib/python3.9/site-packages

#    $HOME/$USER/.conda/envs/$CONDA_ENV_NAME/lib:\
echo '\
    LD_LIBRARY_PATH=$HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:$MAGMA_HOME/lib ./run_llama2_70b.sh'
popd

echo "$HOME/.conda/pkgs/mkl-2023.1.0-h213fc3f_46344/lib" | tee /etc/ld.so.conf.d/mkl.conf
echo "$MAGMA_HOME/lib" | $SUDO tee /etc/ld.so.conf.d/magma.conf
ls -l /etc/ld.so.conf.d/
