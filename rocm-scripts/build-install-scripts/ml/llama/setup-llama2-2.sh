# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 
MINICONDA_SRC_DIR=/home/miniconda3
MINICONDA_DIR=/$HOME/miniconda3
LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/$HOME/miniconda3/bin/conda

CONDA_ENV_NAME="llama2"

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
bash install.sh 2>&1 | tee log/install.log
popd

git clone https://bitbucket.org/icl/magma.git
pushd magma

if [[ -z `cat ~/.bashrc | grep "export.*MAGMA_HOME"` ]] ; then
    export MAGMA_HOME=$PWD | tee -a ~/.bashrc
fi

if [[  -z `cat ~/.bashrc | grep "export.*MKLROOT"` ]] ; then
    export MKLROOT=$HOME/miniconda3/envs/$CONDA_ENV_NAME | tee -a ~/.bashrc
fi

if [[ -z `cat ~/.bashrc | grep "export.*ROCM_PATH"` ]] ; then
    export ROCM_PATH=/opt/rocm-6.2.0-13611 | tee -a ~/.bashrc
fi

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" >> make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" >> make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" >> make.inc
# build MAGMA
make -f make.gen.hipMAGMA -j
HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j
popd

pushd $LLAMA_PREREQ_PKGS
ln -s \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_intel_lp64.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_intel_lp64.so.1
ln -s \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_gnu_thread.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_gnu_thread.so.1

ln -s $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_core.so.2 \
$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/libmkl_core.so.1 

if [[ -z `cat ~/.bashrc | grep "export.*LD_LIBRARY_PATH.*mkl.*$MAGMA_HOME"` ]] ; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
        $HOME/miniconda3/envs/$CONDA_ENV_NAME/lib:\
        $HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:\
        $MAGMA_HOME/lib" | tee -a ~/.bashrc
fi
echo $LD_LIBRARY_PATH
chmod 755 *sh
#LD_LIBRARY_PATH=$HOME/miniconda3/envs/$CONDA_ENV_NAME/lib:$HOME/miniconda3/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:$MAGMA_HOME/lib ./run_llama2_70b.sh 
popd
