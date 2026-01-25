#!/bin/bash

#!/bin/bash

cmake -Bbuild  -DCMAKE_TOOLCHAIN_FILE=../scripts/toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON
cmake --build build --verbose

sshpass -p nvidia scp build/CameraPluginExec build/libCameraPlugin.so data/multi_config.json data/single_config.json nvidia@198.18.42.9:~


