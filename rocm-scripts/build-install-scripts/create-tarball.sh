set -x 
SINGLE_LINE=---------------------------------------------------------------------------------------------
DEBUG=0

function usage()
{
    echo "examples: $0 --rocm=13435 --amdgpu=1720129"
    echo "examples: $0 --release --ver=6.0 --rocm=91 --amdgpu=1704596"
    echo "examples: $0 --rocm=13435 --amdgpu=1720129 --downloadno"
    echo "examples: $0 --rocm=13435 --amdgpu=1720129 --install"
    echo "examples: $0 --rocm=13435 --amdgpu=1720129 --src"
}

function error() {
    echo "Error! $error_string"
    exit 1
}

src=0

for var in "$@"
do
    if [[ $DEBUG -eq 1 ]] ; then echo var: $var ; fi

    case "$var" in 
        --help)
            usage
            exit 0
            ;;
        --src)
            src=1
            ;;
        --rocm=*)
            rocm_build_no=`echo $var | cut -d '=' -f2`
            ;;
        --amdgpu=*)
            amdgpu_build_no=`echo $var | cut -d '=' -f2`
            ;;
        --downloadno)
            if [[ $DEBUG -eq 1 ]] ; then echo "Setting no download flag..." ; fi
            skip_download="1"
            ;;
        --install)
            if [[ $DEBUG -eq 1 ]] ; then echo "Will continue rocm install after creating tarball..." ; fi
            continue_install="1"
            ;;
        --release)
            if [[ $DEBUG -eq 1 ]] ; then eecho "Setting release..." ; fi
            release=1
            ;;
        --ver=*)
            if [[ $DEBUG -eq 1 ]] ; then eecho "Setting rocm version..." ; fi
            rocm_ver=`echo $var | cut -d '=' -f2`
            ;;
        *)
            echo "Error: unknown parameter: $var" ; exit 1
            ;;
    esac
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

yum update -y ; yum install cmake git tree nano wget g++ python3-pip sudo -y

echo "rocm: $rocm_build_no, amdgpu: $amdgpu_build_no, bypass download: $skip_download, release: $release, rocm version: $rocm_ver"

http_amdgpu=http://mkmartifactory.amd.com/artifactory/amdgpu-rpm-local/rhel/9.2/builds/$amdgpu_build_no/x86_64/

if [[ $release -eq 1 ]] ; then
    http_rocm="http://compute-cdn.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-rel-$rocm_ver-$rocm_build_no/"
    http_release_page="http://rocm-ci.amd.com/view/Release-6.1/job/compute-rocm-rel-$rocm_ver/$rocm_build_no/"
else
    http_rocm="http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-dkms-no-npi-hipclang-$rocm_build_no/"
    http_release_page="http://rocm-ci.amd.com/job/compute-rocm-dkms-no-npi-hipclang/$rocm_build_no/"
fi

ROCM_DW_DIR=rocm
AMDGPU_DW_DIR=amdgpu
SRC_DIR=""
mkdir $ROCM_DW_DIR $AMDGPU_DW_DIR

if [[ -z $skip_download ]] ; then
    wget -r  -nv -np -A "*.rpm" -P $ROCM_DW_DIR $http_rocm
    if [[ $? -ne 0 ]] ; then error "Unable to download rocm packages. Check the URL." ; fi
    wget --mirror -L -np -nH -c -nv --cut-dirs=6 -A "*.rpm" -P $AMDGPU_DW_DIR $http_amdgpu
    if [[ $? -ne 0 ]] ; then error "Unable to download amdgpu packages. Check the URL." ; fi
fi

if [[ $src == 1 ]] ; then
    wget $http_release_page/artifact/manifest.xml
    if [[ -f manifest.xml ]] ; then
        revid=`cat manifest.xml  | grep amd-smi | grep -o 'revision=.*' | awk {'print $1'} | awk -F "\"" {'print $2'}`
        branch=`cat manifest.xml  | grep amd-smi | grep -o 'upstream=.*' | awk {'print $1'} | awk -F "\"" {'print $2'}`
    else
        echo "Error: Unable to wget manifest.xml" ; exit 1
    fi

    if [[ -z $revid ]] || [[ -z $branch ]] ; then
        echo "Error: Unable to extract revid/commit ID and/or branch name from manifest.xml"
        echo "Revid: $revid, branch: $branch"
        exit 1
    fi

    mkdir -p src/
    pushd src/
    git clone "ssh://ggankhuy@gerrit-git.amd.com:29418/SYS-MGMT/ec/amd-smi"
    if [[ -d amd-smi ]] ; then
        pushd amd-smi
        git checkout $branch
        result=`git log | grep $revid`
        if [[ -z $result ]] ; then
            echo "Error: Unable to locate revid: $revid"; exit 1
        else
            git checkout $revid
        fi
        popd
    else
        echo "Error: Unable to checkout amd-smi repository from gerrit!" ; exit 1
    fi
    popd
    SRC_DIR=src
fi

output_filename=$ROCM_DW_DIR-$rocm_build_no-$AMDGPU_DW_DIR-$amdgpu_build_no.tar.gz
tar -cvf $output_filename $ROCM_DW_DIR $AMDGPU_DW_DIR $SRC_DIR rocm-local-install-2.sh README
if [[ $? -ne 0 ]] ; then error "Failed to create tar archive." ; fi

tar -tf $output_filename | sudo tee $output_filename.log
echo "content of tar: $output_filename.log"

if [[ ! -z $continue_install ]] ; then
    ./rocm-local-install-internal.sh --no-dkms
    rm -rf $output_filename $output_filename.log
fi
