# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.

MINICONDA_SRC_DIR=/home/miniconda3
MINICONDA_DIR=/$HOME/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2.tar
mkdir -p $MINICONDA_SRC_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
chmod 755 ./miniconda.sh
bash ./miniconda.sh -b -u -p /$MINICONDA_SRC_DIR
rm -rf ./miniconda.sh
echo "export PATH=$PATH:/$MINICONDA_SRC_DIR/bin" >> ~/.bashrc
ln -s $MINICONDA_SRC_DIR /$HOME/


CONDA_ENV_NAME="llama2"
conda create --name  $CONDA_ENV_NAME python==3.9
conda activate $CONDA_ENV_NAME

conda install mkl-service
pip3 install mkl

tar -xf $LLAMA_PREREQ_PKGS
pushd $LLAMA_PREREQ_PKGS
mkdir log
bash install.sh 2>&1 | tee log/install.log
popd

git clone https://bitbucket.org/icl/magma.git
pushd magma
export MAGMA_HOME=$PWD
export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME
export ROCM_PATH=/opt/rocm-6.2.0-13873
popd

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" >> make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" >> make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" >> make.inc
# build MAGMA
make -f make.gen.hipMAGMA -j
HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j


ln -s \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_intel_lp64.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_intel_lp64.so.1
ln -s \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_gnu_thread.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_gnu_thread.so.1

ln -s $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_core.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_core.so.1 

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
    $HOME/miniconda3/envs/$CONDA_ENV_NAME/lib:\
    $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:\
    $MAGMA_HOME/lib"




