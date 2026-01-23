set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(DEFINED ENV{NV_WORKSPACE})
    set(TOOLCHAIN_PREFIX $ENV{NV_WORKSPACE}/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-)
else()
    set(TOOLCHAIN_PREFIX /home/lw/Workspace/ns/pdk/toolchains/aarch64--glibc--stable-2022.03-1/bin/aarch64-buildroot-linux-gnu-)
endif()
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
