set -x 

LOG_DIR=log/
BUILD_TIME_LOG=build-time.log

mkdir $LOG_DIR -p
wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
gunzip /usr/local/bin/ninja.gz
chmod a+x /usr/local/bin/ninja

CONFIG_PATH_NINJA_TRACING=~/extdir/gg/git/ninjatracing/ninjatracing
CONFIG_PATH_NINJA_TRACING_DIRNAME=`dirname $CONFIG_PATH_NINJA_TRACING`

git clone https://github.com/nico/ninjatracing.git $CONFIG_PATH_NINJA_TRACING_DIRNAME
ln -s /usr/bin/python3 /usr/bin/python

# build rccl manually

echo "start time: " | sudo tee $LOGDIR/step3.time.$SUFFIX.log
date | sudo tee -a $LOGDIR/$BUILD_TIME_LOG
pwd


CXX=hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ ..
BUILD_CMD="ninja"
NPROC="-j 32"
cmake -DCMAKE_BUILD_WITH_INSTALL_RPATH=1 -DCMAKE_PREFIX_PATH=/opt/rocm/ -GNinja ..
time $BUILD_CMD $NPROC 2>&1 | sudo tee $LOGDIR/step2.$BUILD_CMD.time.$SUFFIX.log
$CONFIG_PATH_NINJA_TRACING  ./.ninja_log | sudo tee $LOGDIR/compile.ninja.trace.json

echo "end time: " | sudo tee -a $LOGDIR/step3.time.$SUFFIX.log
date | sudo tee -a $LOGDIR/$BUILD_TIME_LOG
