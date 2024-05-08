set -x 

LOG_DIR=log/
BUILD_TIME_LOG=build-time.log
BUILD_LOG=build.log
mkdir $LOG_DIR -p
NINJA_DW_PATH=/usr/local/bin/ninja.gz
NINJA=/usr/local/bin/ninja
wget -qO $NINJA_DW_PATH https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
if [[ ! -f $NINJA ]] ; then 
    gunzip $NINJA_DW_PATH 
else 
    echo "$NINJA already exists, bypassing extract..." 
fi
chmod a+x $NINJA

CONFIG_PATH_NINJA_TRACING=~/extdir/gg/git/ninjatracing/ninjatracing
CONFIG_PATH_NINJA_TRACING_DIRNAME=`dirname $CONFIG_PATH_NINJA_TRACING`

git clone https://github.com/nico/ninjatracing.git $CONFIG_PATH_NINJA_TRACING_DIRNAME
ln -s /usr/bin/python3 /usr/bin/python
ln -s $NINJA `dirname $NINJA`/ninja-build

# One of following softlink worked with cmake generator. 
# Probably some idiot at cmake development team made hard code path such that
# only /usr/bin or /usr/sbin works and nothing even though it is found by $PATH variable!

ln -s $NINJA /usr/bin/ninja
ln -s $NINJA /usr/sbin/ninja
# build rccl manually

echo "start time: " | sudo tee $LOGDIR/$BUILD_TIME_LOG
date | sudo tee -a $LOGDIR/$BUILD_TIME_LOG
pwd

sudo git clone https://github.com/ROCm/rccl
cd rccl
mkdir build ; cd build
CXX=hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -G Ninja ..
BUILD_CMD=$NINJA
NPROC="-j 32"
cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=1 -DCMAKE_PREFIX_PATH=/opt/rocm/ -GNinja ..
if [[ $? != 0 ]] ; then exit 1 ; fi
time $BUILD_CMD $NPROC 2>&1 | sudo tee $LOG_DIR/$BUILD_LOG
if [[ $? != 0 ]] ; then exit 1 ; fi
$CONFIG_PATH_NINJA_TRACING ./.ninja_log | sudo tee $LOGDIR/compile.ninja.trace.json
if [[ $? != 0 ]] ; then exit 1 ; fi

echo "end time: " | sudo tee -a $LOGDIR/$BUILD_TIME_LOG
date | sudo tee -a $LOGDIR/$BUILD_TIME_LOG
