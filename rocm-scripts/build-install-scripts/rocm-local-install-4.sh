set -x
SINGLE_LINE=---------------------------------------------------------------------------------------------
DEBUG=0
function usage()
{
    if [[ -z $DEBUG ]] ; then clear ; fi
    echo $SINGLE_LINE
    echo "examples: $0  # install rocm + amdgpu"
    echo "examples: $0 --no-dkms # install rocm only, no dkms, usually for docker env. "
    echo $SINGLE_LINE
}

function error() {
    echo "Error! $error_string"
    exit 1
}

for var in "$@"
do
    if [[ $DEBUG -eq 1 ]] ; then echo var: $var ; fi

    case "$var" in 
        *--help*)
            usage
            exit 0
            ;;

        *--no-dkms*)
            if [[ $DEBUG -eq 1 ]] ; then echo "Will bypass dkms installation usually for docker env..." ; fi
            nodkms=1
            ;;
        *)
            echo "Error: unknown parameter: $var" ; exit 1
    esac        
done

# needed for building amdgpu, so adding here. Otherwise we will only discovery by looking 
# at make.log of amdgpu build.

yum install dwarves  -y

yum install createrepo -y
yum install epel-release epel-next-release -y 
yum config-manager --set-enabled crb
yum install epel-release epel-next-release -y

amdgpu-install --uninstall -y
amdgpu-install --uninstall -y --rocmrelease=all
sudo yum remove rocm amdgpu -y

for i in  rocm amdgpu; do
    rm -rf /etc/yum.repos.d/*$i*.repo
done

for i in rocm amdgpu; do
    path=`find $i -name repodata | head -1`
    dir=`dirname $path`
    pushd $dir
    pwd

    createrepo --simple-md-filenames .
    echo -ne "" > $i.repo
    echo "[$i]" >> $i.repo
    echo "name=$i repository" >> $i.repo
    echo "baseurl=//`pwd`" >> $i.repo
    echo "enabled=1" >> $i.repo
    echo "gpgcheck=0" >> $i.repo

    cp $i.repo /etc/yum.repos.d/


    popd
done

for i in rocm amdgpu ; do
    if [[ $i == "amdgpu" ]] ; then
        if [[ $nodkms -eq 1 ]] ; then
            echo "Will bypass dkms installation."
            continue
        fi
    fi
    yum install $i -y --nogpgcheck

    if [[ $i == "rocm" ]] ; then
        echo "Creating llvm soft links..."
        sleep 3
        for j in clang ld.lld ; do
            sudo rm -rf /usr/bin/$j
            sudo ln -s /opt/rocm/llvm/bin/$j /usr/bin/
        done
    fi
done
