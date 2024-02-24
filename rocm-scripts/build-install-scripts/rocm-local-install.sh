tarball_name=$1

if [[ -z $tarball_name ]] ; then
    echo "Tarball needs to be specified: usage: $0 <tarball_name>, example: $0 rocm13534.tar.gz"
    exit 1
fi

tar -xvf $tarball_name
