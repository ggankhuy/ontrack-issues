set -x

# build types full build: will build entire CK, cpp build: will build specific source file, specified in OPTION_BUILD_TYPE_CPP_FILENAME
OPTION_BUILD_TYPE_FULL=1
OPTION_BUILD_TYPE_CPP=2
OPTION_BUILD_TYPE=$OPTION_BUILD_TYPE_CPP
OPTION_BUILD_TYPE_CPP_FILENAME="device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instance"
OPTION_BUILD_TYPE_CPP_DIR="library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm"
OPTION_BUILD_GEN_MAKE=1
OPTION_BUILD_GEN_NINJA=2
OPTION_BUILD_GEN=$OPTION_BUILD_GEN_NINJA
OPTION_GENERATE_REPORT=1
CONFIG_SAVE_BUILD_FILES=1

CONFIG_PATH_NINJA_TRACING=/home/master/gg/git/ninjatracing/ninjatracing
#for i in "" ; do
#for i in "-g2" ; do
for i in "" "-gdwarf-aranges" "-gdwarf-4" "-g2" ; do

    echo '-------------------'
    mkdir build
    cd build
    SUFFIX=$i
    FLAGS=$i
    GPU_TARGET="gfx90a"
    BUILD_TARGET="device_gemm_add_relu_add_xdl_c_shuffle_layernorm_f16_km_kn_mn_mn_mn_instance"
    LOGDIR=../log/savetemp/build-target_cpp_1_$BUILD_TARGET_debug_$SUFFIX/

    if [[ $OPTION_GENERATE_REPORT == 1 ]] ; then
        FLAGS="$FLAGS -v -ftime-report --save-temps"
        SUFFIX="$SUFFIX.report"
    fi
    mkdir $LOGDIR/$SUFFIX -p
    rm -rf ./build/*

    # cmake for entire build...

    case "$OPTION_BUILD_GEN" in 
    "$OPTION_BUILD_GEN_MAKE")
        cmake \
        -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
        -DCMAKE_BUILD_TYPE=Release \
        -DGPU_TARGETS="$GPU_TARGET" \
        -DCMAKE_CXX_FLAGS="$FLAGS" \
        .. 2>&1 | tee $LOGDIR/step1.cmake.log
        ;;
    "$OPTION_BUILD_GEN_NINJA")
        cmake \
        -DBUILD_DEV=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang-17 \
        -DGPU_TARGETS="$GPU_TARGET" \
        -DCMAKE_CXX_FLAGS="$FLAGS" \
        -GNinja \
        -DCMAKE_PREFIX_PATH=/opt/rocm .. 2>&1 | sudo tee $LOGDIR/step1.cmake.log
        ;;
    *)
        echo "Error! OPTION_BUILD_TYPE is unknown: $OPTION_BUILD_TYPE"
        ;;
    esac

    # -DGPU_TARGETS="gfx908;gfx90a"
    # -DCMAKE_CXX_FLAGS="$FLAGS -v" \

    echo "start time: " | sudo tee $LOGDIR/step3.time.$SUFFIX.log
    date | sudo tee -a $LOGDIR/step3.time.$SUFFIX.log
    pwd


    # for cpp target
    pwd

    case "$OPTION_BUILD_GEN" in 
    "$OPTION_BUILD_GEN_MAKE")
        BUILD_CMD="make"
        NPROC="-j 32"
        ;;
    "$OPTION_BUILD_GEN_NINJA")
        BUILD_CMD="ninja"
        NPROC="-j 32"
        ;;
    esac

    TARGET=""
    echo "choosing build type: cpp or full: OPTION_BUILD_TYPE: $OPTION_BUILD_TYPE"
    case "$OPTION_BUILD_TYPE" in 
    $OPTION_BUILD_TYPE_CPP)
        
        echo "Performing cpp build only..."

        case "$OPTION_BUILD_GEN" in 
        "$OPTION_BUILD_GEN_MAKE")
            cd library/src/tensor_operation_instance/gpu/gemm_add_relu_add_layernorm
            time $BUILD_CMD $NPROC 2>&1 | sudo tee ../../../../../$LOGDIR/step2.$BUILD_CMD.time.$SUFFIX.log
            cd ../../../../../../build 
            ;;
        "$OPTION_BUILD_GEN_NINJA")
            time $BUILD_CMD $NPROC $OPTION_BUILD_TYPE_CPP_DIR/all $NPROC 2>&1 | sudo tee $LOGDIR/step2.$BUILD_CMD.time.$SUFFIX.log
            $CONFIG_PATH_NINJA_TRACING  ./.ninja_log | sudo tee $LOGDIR/step5.compile.ninja.trace.json
            ;;
        esac    

        ;;
    $OPTION_BUILD_TYPE_FULL)
        echo "Performing full build..."
        pwd
        time $BUILD_CMD $NPROC 2>&1 | sudo tee $LOGDIR/step2.$BUILD_CMD.time.$SUFFIX.log 
        pwd
    esac

    echo "end time: " | sudo tee -a $LOGDIR/step3.time.$SUFFIX.log
    date | sudo tee -a $LOGDIR/step3.time.$SUFFIX.log
    if [[ $CONFIG_SAVE_BUILD_FILES == 1 ]] ; then
        echo "Saving all build files to log folder..."
        sudo mv ./* $LOGDIR/$SUFFIX/
    fi
    cd ..
done
