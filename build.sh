#!/bin/bash

#!/bin/bash

cmake -Bbuild  -DCMAKE_TOOLCHAIN_FILE=../scripts/toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON
cmake --build build --verbose

# sshpass -p nvidia scp build/VideoSource Config/test_config.json build/libyuvtorgb.so nvidia@192.168.1.22:~


