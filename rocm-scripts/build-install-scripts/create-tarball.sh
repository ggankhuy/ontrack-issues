set -x 
SINGLE_LINE=---------------------------------------------------------------------------------------------
DEBUG=0

function usage()
{
    echo "examples: $0 --rocm=13435 --amdgpu=1720129"
    echo "examples: $0 --release --ver=6.0 --rocm=91 --amdgpu=1704596"
    echo "examples: $0 --rocm=13435 --amdgpu=1720129 --no-download"
}

function error() {
    echo "Error! $error_string"
    exit 1
}

for var in "$@"
do
    if [[ $DEBUG -eq 1 ]] ; then echo var: $var ; fi

    if [[ $var == *"--help="* ]]  ; then
        usage
        exit 0
    fi

    if [[ $var == *"--rocm="* ]]  ; then
        rocm_build_no=`echo $var | cut -d '=' -f2`
    fi

    if [[ $var == *"--amdgpu="* ]]  ; then
        amdgpu_build_no=`echo $var | cut -d '=' -f2`
    fi

    if [[ $var == *"--downloadno"* ]]  ; then
        if [[ $DEBUG -eq 1 ]] ; then echo "Setting no download flag..." ; fi
        skip_download="1"
    fi

    if [[ $var == *"--release"* ]]  ; then
        if [[ $DEBUG -eq 1 ]] ; then eecho "Setting release..." ; fi
        release=1
    fi

    if [[ $var == *"--ver="* ]]  ; then
        if [[ $DEBUG -eq 1 ]] ; then eecho "Setting rocm version..." ; fi
        rocm_ver=`echo $var | cut -d '=' -f2`
    fi
done

if [[ -z $amdgpu_build_no ]] ; then
    clear
    echo $SINGLE_LINE
    echo "Need amdgpu build No."
    usage
    echo $SINGLE_LINE
    exit 1
fi

if [[ -z $rocm_build_no ]] ; then
    clear
    echo $SINGLE_LINE
    echo "Need rocm build No."
    usage
    echo $SINGLE_LINE
    exit 1
fi

if [[ ! -z $release ]] && [[ -z $rocm_ver ]] ; then
    clear
    echo $SINGLE_LINE
    echo "If release is specified, must supply rocm version."
    usage
    echo $SINGLE_LINE
    exit 1
fi

echo "rocm: $rocm_build_no, amdgpu: $amdgpu_build_no, bypass download: $skip_download, release: $release, rocm version: $rocm_ver"

http_amdgpu=http://mkmartifactory.amd.com/artifactory/amdgpu-rpm-local/rhel/9.2/builds/$amdgpu_build_no/x86_64/

if [[ $release -eq 1 ]] ; then
    http_rocm="http://compute-cdn.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-rel-$rocm_ver-$rocm_build_no/"
else
    http_rocm="http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-dkms-no-npi-hipclang-$rocm_build_no/"
fi

ROCM_DW_DIR=rocm
AMDGPU_DW_DIR=amdgpu

mkdir $ROCM_DW_DIR $AMDGPU_DW_DIR

if [[ -z $skip_download ]] ; then
    wget -r  -nv -np -A "*.rpm" -P $ROCM_DW_DIR $http_rocm
    if [[ $? -ne 0 ]] ; then error "Unable to download rocm packages. Check the URL." ; fi
    wget --mirror -L -np -nH -c -nv --cut-dirs=6 -A "*.rpm" -P $AMDGPU_DW_DIR $http_amdgpu
    if [[ $? -ne 0 ]] ; then error "Unable to download amdgpu packages. Check the URL." ; fi
fi

output_filename=$ROCM_DW_DIR-$rocm_build_no-$AMDGPU_DW_DIR-$amdgpu_build_no.tar.gz
tar -cvf $output_filename.tar.gz $ROCM_DW_DIR $AMDGPU_DW_DIR rocm-local-install.sh
if [[ $? -ne 0 ]] ; then error "Failed to create tar archive." ; fi

tar -tf $output_filename.tar.gz | sudo tee $output_filename.log
echo "content of tar: $output_filename.log"
