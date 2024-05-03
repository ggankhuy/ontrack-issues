# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 
MINICONDA_SRC_DIR=/home/miniconda3
MINICONDA_DIR=/$HOME/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/$HOME/miniconda3/bin/conda
for i in gfortran libomp; do 
    yum install $i -y ; 
done
CONDA_ENV_NAME="llama2"
SOFT_LINK=0

if [[ ! -f $LLAMA_PREREQ_PKGS.tar ]] ; then 
    echo "$LLAMA_PREREQ_PKGS.tar does not exist." 
    exit 1
fi
tar -xvf  $LLAMA_PREREQ_PKGS.tar
pushd $LLAMA_PREREQ_PKGS
    for i in *tar ; do 
        dirname=`echo $i | awk '{print $1}' FS=. `
        mkdir $dirname ; pushd $dirname
        ln -s ../$i .
        tar -xvf ./$i 
        pip3 install ./*.whl
        popd
    done
popd

conda install mkl-service -y
pip3 install mkl 

tar -xf $LLAMA_PREREQ_PKGS.tar
pwd
ls -l 

pushd $LLAMA_PREREQ_PKGS
mkdir log
bash install.sh 2>&1 | sudo tee log/install.log
popd

git clone https://bitbucket.org/icl/magma.git
pushd magma

BASHRC=~/.bashrc
BASHRC_EXPORT=./export.md
ROCM_PATH=/opt/rocm-6.2.0-13611

ls -l $BASHRC
if [[ -z `cat $BASHRC | grep "export.*MAGMA_HOME"` ]] ; then
    echo "export MAGMA_HOME=$PWD" | sudo tee -a $BASHRC | sudo tee -a $BASHRC_EXPORT
    export MAGMA_HOME=$PWD
fi

if [[  -z `cat $BASHRC | grep "export.*MKLROOT"` ]] ; then
    echo "export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME" |  sudo tee -a $BASHRC | sudo tee -a $BASHRC_EXPORT
    export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME
fi

if [[ -z `cat $BASHRC | grep "export.*ROCM_PATH"` ]] ; then
    export ROCM_PATH=$ROCM_PATH
    echo "export ROCM_PATH=$ROCM_PATH" |  sudo tee -a $BASHRC | sudo tee -a $BASHRC_EXPORT
fi

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" >> make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" >> make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" >> make.inc
# build MAGMA
make -f make.gen.hipMAGMA -j
HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j 2>&1 | tee make.magma.log
popd

pushd $LLAMA_PREREQ_PKGS

if [[ $SOFT_LINK == 1 ]] ; then
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        ln -s \
        $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
else
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        rm -rf $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
        cp \
        $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
fi

chmod 755 *sh
echo "Use following cmd to run:"
echo 'LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV_NAME/lib:$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:$MAGMA_HOME/lib ./run_llama2_70b.sh'
popd

echo "$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib" | tee /etc/ld.so.conf.d/mkl.conf
echo "$MAGMA_HOME/lib" | tee /etc/ld.so.conf.d/magma.conf
ls -l /etc/ld.so.conf.d/
