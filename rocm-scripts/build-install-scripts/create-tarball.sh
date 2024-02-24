SINGLE_LINE=----------------------------------------------
rocm_build_no=$1
amdgpu_build_no=$2
skip_download=$3

if [[ -z $rocm_build_no ]] ; then
    clear
    echo $SINGLE_LINE
    echo "Need rocm build No: usage: $0 rocm_build_no amdgpu_build_no: example: $0 13435 1720129"
    echo $SINGLE_LINE
    exit 1
fi

if [[ -z $amdgpu_build_no ]] ; then
    clear
    echo $SINGLE_LINE
    echo "Need amdgpu build No: usage: $0 rocm_build_no amdgpu_build_no: example: $0 13435 1720129"
    echo $SINGLE_LINE
    exit 1
fi

ROCM_DW_DIR=rocm-$rocm_build_no
AMDGPU_DW_DIR=amdgpu-$amdgpu_build_no

mkdir $ROCM_DW_DIR $AMDGPU_DW_DIR

if [[ -z $skip_download ]] ; then
    wget -r  -nv -np -A "*.rpm" -P $ROCM_DW_DIR http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-dkms-no-npi-hipclang-$rocm_build_no
    wget --mirror -L -np -nH -c -nv --cut-dirs=6 -A "*.rpm" -P $AMDGPU_DW_DIR  http://mkmartifactory.amd.com/artifactory/amdgpu-rpm-local/rhel/9.2/builds/$amdgpu_build_no/x86_64
fi

tar -cvf $ROCM_DW_DIR.tar.gz $ROCM_DW_DIR
tar -cvf $AMDGPU_DW_DIR.tar.gz $AMDGPU_DW_DIR
