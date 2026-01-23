# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

include ../../../make/nvdefs.mk

TARGETS = nvsipl_multicast

NV_BUILD_DOS = 6
NV_BUILD_CARDETECT ?= 0

CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -I. -I../../../include/nvmedia_6x/
CXXFLAGS += -I./config/include -I./core/include -I./modules/include -I./utils/include -I./control_channel/include/
CXXFLAGS += -I./features/low_power_mode/include
CXXFLAGS += -I./services/status_manager/include
CXXFLAGS += -fexceptions -frtti -fPIC $(NV_PLATFORM_CXXFLAGS)
CPPFLAGS := $(NV_PLATFORM_CPPFLAGS) $(NV_PLATFORM_SDK_INC)
CPPFLAGS += -I./platform -frtti -rdynamic -fexceptions
LDFLAGS := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS) -rdynamic
ifeq ($(NV_BUILD_DOS),7)
CPPFLAGS += -DNV_BUILD_DOS7
endif


ifeq ($(NV_PLATFORM_SAFETY),1)
CPPFLAGS += -DNV_IS_SAFETY=1
endif


MULTICAST_CONFIG_SRCS_DIR := ./config/src
MULTICAST_CONFIG_SRCS := $(wildcard $(MULTICAST_CONFIG_SRCS_DIR)/*.cpp)
MULTICAST_CONFIG_OBJS := $(patsubst $(MULTICAST_CONFIG_SRCS_DIR)/%.cpp, $(MULTICAST_CONFIG_SRCS_DIR)/%.o, $(MULTICAST_CONFIG_SRCS))
MULTICAST_CONFIG_DEPS := $(patsubst $(MULTICAST_CONFIG_SRCS_DIR)/%.cpp, $(MULTICAST_CONFIG_SRCS_DIR)/%.d, $(MULTICAST_CONFIG_SRCS))

MULTICAST_CONTROL_CHANNEL_SRCS_DIR := ./control_channel/src
MULTICAST_CONTROL_CHANNEL_SRCS := $(wildcard $(MULTICAST_CONTROL_CHANNEL_SRCS_DIR)/*.cpp)
MULTICAST_CONTROL_CHANNEL_OBJS := $(patsubst $(MULTICAST_CONTROL_CHANNEL_SRCS_DIR)/%.cpp, $(MULTICAST_CONTROL_CHANNEL_SRCS_DIR)/%.o, $(MULTICAST_CONTROL_CHANNEL_SRCS))
MULTICAST_CONTROL_CHANNEL_DEPS := $(patsubst $(MULTICAST_CONTROL_CHANNEL_SRCS_DIR)/%.cpp, $(MULTICAST_CONTROL_CHANNEL_SRCS_DIR)/%.d, $(MULTICAST_CONTROL_CHANNEL_SRCS))

MULTICAST_CORE_SRCS_DIR := ./core/src
MULTICAST_CORE_SRCS := $(wildcard $(MULTICAST_CORE_SRCS_DIR)/*.cpp)
MULTICAST_CORE_OBJS := $(patsubst $(MULTICAST_CORE_SRCS_DIR)/%.cpp, $(MULTICAST_CORE_SRCS_DIR)/%.o, $(MULTICAST_CORE_SRCS))
MULTICAST_CORE_DEPS := $(patsubst $(MULTICAST_CORE_SRCS_DIR)/%.cpp, $(MULTICAST_CORE_SRCS_DIR)/%.d, $(MULTICAST_CORE_SRCS))

MULTICAST_MODULES_SRCS_DIR := ./modules/src
MULTICAST_MODULES_SRCS := $(wildcard $(MULTICAST_MODULES_SRCS_DIR)/*.cpp)

CV_CUDA_OBJS := ./utils/src/cvt2d_kernels.o
CV_CUDA_SRCS := ./utils/src/cvt2d_kernels.cu
CV_CUDA_INCLUDES += -I./utils/include

# Flag set to zero if you don't use vksc module, which avoid installing vksc package.
NV_BUILD_VULKANSC := 1
ifneq ($(NV_BUILD_VULKANSC),1)
FILES_FILTER_OUT := $(MULTICAST_MODULES_SRCS_DIR)/CVulkanSCModule.cpp \
                    $(MULTICAST_MODULES_SRCS_DIR)/CVulkanSCEngine.cpp
MULTICAST_MODULES_SRCS := $(filter-out $(FILES_FILTER_OUT), $(MULTICAST_MODULES_SRCS))
endif

MULTICAST_MODULES_OBJS := $(patsubst $(MULTICAST_MODULES_SRCS_DIR)/%.cpp, $(MULTICAST_MODULES_SRCS_DIR)/%.o, $(MULTICAST_MODULES_SRCS))
MULTICAST_MODULES_DEPS := $(patsubst $(MULTICAST_MODULES_SRCS_DIR)/%.cpp, $(MULTICAST_MODULES_SRCS_DIR)/%.d, $(MULTICAST_MODULES_SRCS))

MULTICAST_UTILS_SRCS_DIR := ./utils/src
MULTICAST_UTILS_SRCS := $(wildcard $(MULTICAST_UTILS_SRCS_DIR)/*.cpp)
MULTICAST_UTILS_OBJS := $(patsubst $(MULTICAST_UTILS_SRCS_DIR)/%.cpp, $(MULTICAST_UTILS_SRCS_DIR)/%.o, $(MULTICAST_UTILS_SRCS))
MULTICAST_UTILS_DEPS := $(patsubst $(MULTICAST_UTILS_SRCS_DIR)/%.cpp, $(MULTICAST_UTILS_SRCS_DIR)/%.d, $(MULTICAST_UTILS_SRCS))

MULTICAST_PVALowPower_SRCS_DIR := ./features/low_power_mode/src
MULTICAST_PVALowPower_SRCS := $(wildcard $(MULTICAST_PVALowPower_SRCS_DIR)/*.cpp)
MULTICAST_PVALowPower_OBJS := $(patsubst $(MULTICAST_PVALowPower_SRCS_DIR)/%.cpp, $(MULTICAST_PVALowPower_SRCS_DIR)/%.o, $(MULTICAST_PVALowPower_SRCS))
MULTICAST_PVALowPower_DEPS := $(patsubst $(MULTICAST_PVALowPower_SRCS_DIR)/%.cpp, $(MULTICAST_PVALowPower_SRCS_DIR)/%.d, $(MULTICAST_PVALowPower_SRCS))


OBJS := main.o
OBJS += $(MULTICAST_CONFIG_OBJS)
OBJS += $(MULTICAST_CONTROL_CHANNEL_OBJS)
OBJS += $(MULTICAST_CORE_OBJS)
OBJS += $(MULTICAST_MODULES_OBJS)
OBJS += $(MULTICAST_UTILS_OBJS)
OBJS += $(MULTICAST_PVALowPower_OBJS)

DEPS := main.d
DEPS += $(MULTICAST_CONFIG_DEPS)
DEPS += $(MULTICAST_CONTROL_CHANNEL_DEPS)
DEPS += $(MULTICAST_CORE_DEPS)
DEPS += $(MULTICAST_MODULES_DEPS)
DEPS += $(MULTICAST_UTILS_DEPS)
DEPS += $(MULTICAST_PVALowPower_DEPS)

MULTICAST_STATUS_MANAGER_SRCS_DIR := ./services/status_manager/src
MULTICAST_STATUS_MANAGER_SRCS := $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/CStatusManagerCommon.cpp \
                                 $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/CStatusManagerClient.cpp
MULTICAST_STATUS_MANAGER_OBJS := $(patsubst $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/%.cpp, $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/%.o, $(MULTICAST_STATUS_MANAGER_SRCS))
MULTICAST_STATUS_MANAGER_DEPS := $(patsubst $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/%.cpp, $(MULTICAST_STATUS_MANAGER_SRCS_DIR)/%.d, $(MULTICAST_STATUS_MANAGER_SRCS))

OBJS += $(MULTICAST_STATUS_MANAGER_OBJS)
DEPS += $(MULTICAST_STATUS_MANAGER_DEPS)

LDLIBS += -lnvsipl
ifeq ($(NV_PLATFORM_SAFETY),0)
LDLIBS += -lnvsipl_query
LDLIBS += -lnvmedia_ide_parser
LDLIBS += -lnvmedia_ide_sci
endif
LDLIBS += -lcam_fsync
LDLIBS += -lnvmedia_iep_sci
LDLIBS += -lnvscistream
LDLIBS += -lnvmedia2d
LDLIBS += -lnvscibuf
LDLIBS += -lnvscisync
LDLIBS += -lnvsciipc
LDLIBS += -lnvscicommon
LDLIBS += -lcuda
LDLIBS += -ltegrawfd
ifeq ($(NV_PLATFORM_OS),QNX)
  CPPFLAGS += -DNVMEDIA_QNX
  LDLIBS += $(NV_PLATFORM_CUDA_LIB)/libcudart_static.a
  LDLIBS += -lnvdtcommon
  LDLIBS += -lsocket
  LDLIBS += -lnvdvms_client
  LDLIBS += -lslog2
else
  LDLIBS += -L$(NV_PLATFORM_CUDA_TOOLKIT)/targets/aarch64-linux/lib/ -lcudart
  LDLIBS += -lpthread
  LDLIBS += -ldl
endif

PCC_TOOL :=
PIPELINE_CACHE_RELY_FILES :=
PIPELINE_CACHE_OUTPUT :=
PIPELINE_CACHE_PATH := ./features/vksc_scenes/data
ifeq ($(NV_BUILD_VULKANSC),1)
CPPFLAGS += -DBUILD_VULKANSC

#Build Vulkan to get json file which will be used to generate pipeline cache.
NV_USE_VULKAN := 0
ifeq ($(NV_USE_VULKAN),0)
	CPPFLAGS += -I$(NV_PLATFORM_DIR)/include/VulkanSC -I$(NV_PLATFORM_DIR)/include/vulkan
	LDLIBS   += -lnvidia-vksc-core

  PIPELINE_CACHE_RELY_FILES := $(wildcard $(PIPELINE_CACHE_PATH)/*.spv $(PIPELINE_CACHE_PATH)/*.json)
  PCC_TOOL += $(NV_PLATFORM_DIR)/vulkansc/pcc/Linux_x86-64/pcc
  PIPELINE_CACHE_OUTPUT := $(PIPELINE_CACHE_PATH)/pipeline_cache.bin
else
	CPPFLAGS += -DVULKAN=1
	CXXFLAGS += -I$(NV_PLATFORM_SDK_INC_DIR)/vulkan
	LDLIBS   += -lvulkan
endif

VULKANSC_SCENES_PATH := features/vksc_scenes
CXXFLAGS += -I$(VULKANSC_SCENES_PATH)/include
VULKANSC_SCENES_SRCS := $(wildcard $(VULKANSC_SCENES_PATH)/src/*.cpp)
VULKANSC_SCENES_OBJS := $(patsubst $(VULKANSC_SCENES_PATH)/src/%.cpp, $(VULKANSC_SCENES_PATH)/src/%.o, $(VULKANSC_SCENES_SRCS))
VULKANSC_SCENES_DEPS := $(patsubst $(VULKANSC_SCENES_PATH)/src/%.cpp, $(VULKANSC_SCENES_PATH)/src/%.d, $(VULKANSC_SCENES_SRCS))
OBJS += $(VULKANSC_SCENES_OBJS)
DEPS += $(VULKANSC_SCENES_DEPS)
endif

ifeq ($(NV_PLATFORM_OS),Linux)
  INFER_PATH := aarch64-linux-gnu
  NVCC := $(NV_PLATFORM_CUDA_TOOLKIT)/bin/nvcc -ccbin ${CC}
else ifeq ($(NV_PLATFORM_OS),QNX)
  INFER_PATH := aarch64-unknown-nto-qnx
  HOST_COMPILER ?= ${QNX_HOST}/usr/bin/aarch64-unknown-nto-qnx7.1.0-g++
  LDLIBS += -L/usr/local/cuda-targets/aarch64-qnx/11.4/targets/aarch64-qnx/lib/
  NVCC := $(NV_PLATFORM_CUDA_TOOLKIT)/bin/nvcc -ccbin ${HOST_COMPILER}
else
  @echo "PLATFORM unsupport!"
endif

ifneq ($(NV_PLATFORM_SAFETY),1)
ifeq ($(NV_BUILD_CARDETECT),1)
  CPPFLAGS += -DBUILD_CARDETECT
  CAR_DETECT_PATH := features/car_detect
  CAR_DETECT_SRCS := $(wildcard $(CAR_DETECT_PATH)/src/*.cpp)
  CAR_DETECT_OBJS := $(patsubst $(CAR_DETECT_PATH)/src/%.cpp, $(CAR_DETECT_PATH)/src/%.o, $(CAR_DETECT_SRCS))
  CAR_DETECT_DEPS := $(patsubst $(CAR_DETECT_PATH)/src/%.cpp, $(CAR_DETECT_PATH)/src/%.d, $(CAR_DETECT_SRCS))
  OBJS += $(CAR_DETECT_OBJS)
  DEPS += $(CAR_DETECT_DEPS)
  CU_OBJS += $(CAR_DETECT_PATH)/src/cuda_kernels.o
  CU_SRCS += $(CAR_DETECT_PATH)/src/cuda_kernels.cu
  CXXFLAGS += -I$(CAR_DETECT_PATH)/include -Wno-deprecated-declarations

  SYSROOT_PATH ?= /
  CXXFLAGS += -I${SYSROOT_PATH}/usr/include/${INFER_PATH}
  LDLIBS += -L${SYSROOT_PATH}/usr/lib/${INFER_PATH}/stubs/
  LDLIBS += -L${SYSROOT_PATH}/usr/lib/${INFER_PATH}/
  LDLIBS += -lnvinfer

ifneq ($(NV_BUILD_DOS),7)
    LDLIBS += -L$(NV_PLATFORM_DIR)/filesystem/targetfs/usr/local/cuda-11.4/targets/aarch64-linux/lib/
    LDLIBS += -lcudla
endif

  CU_INCLUDES += -I$(CAR_DETECT_PATH)/include ${CV_CUDA_INCLUDES}
  TARGET_SIZE := 64
  NVCCFLAGS   := -m${TARGET_SIZE}

endif
endif

ifeq ($(NV_BUILD_DOS),7)
  NVCCFLAGS   += -DNV_BUILD_DOS7 -gencode arch=compute_101,code=sm_101
endif

.PHONY: default
default: $(TARGETS) $(PIPELINE_CACHE_OUTPUT)

ifneq ($(NV_PLATFORM_SAFETY),1)
ifeq ($(NV_BUILD_CARDETECT),1)
$(CU_OBJS): $(CU_SRCS)
	$(NVCC) $(CU_INCLUDES) $(NVCCFLAGS) -o $@ -c $<
endif
endif

ifeq ($(NV_BUILD_VULKANSC),1)
ifeq ($(NV_USE_VULKAN),0)
ifeq ($(NV_BUILD_DOS),7)
  CHIP_TYPE := gb
else
  CHIP_TYPE := ga
endif
$(PIPELINE_CACHE_OUTPUT): $(PIPELINE_CACHE_RELY_FILES)
	$(PCC_TOOL) -chip $(CHIP_TYPE)10b -path $(PIPELINE_CACHE_PATH) -out $@
endif
endif

$(CV_CUDA_OBJS): $(CV_CUDA_SRCS)
	$(NVCC) $(CV_CUDA_INCLUDES) $(NVCCFLAGS) $(NV_PLATFORM_SDK_INC) -o $@ -c $<

$(TARGETS): $(OBJS) $(CU_OBJS) $(CV_CUDA_OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)

STATUS_MANAGER_OBJS := services/status_manager/src/CStatusManagerService.o \
                      services/status_manager/src/CStatusManagerCommon.o utils/src/CLogger.o
STATUS_MANAGER_DEPS := services/status_manager/src/CStatusManagerService.d \
                      services/status_manager/src/CStatusManagerCommon.d utils/src/CLogger.d
ifeq ($(NV_PLATFORM_OS),QNX)
  STATUS_MANAGER_OBJS += services/status_manager/src/CStatusManagerQnxHelper.o
  STATUS_MANAGER_DEPS += services/status_manager/src/CStatusManagerQnxHelper.d
else
  STATUS_MANAGER_OBJS += services/status_manager/src/CStatusManagerLinuxHelper.o
  STATUS_MANAGER_DEPS += services/status_manager/src/CStatusManagerLinuxHelper.d
endif

default: status_manager
status_manager: $(STATUS_MANAGER_OBJS)
	$(LD) $(LDFLAGS) -o $@ $^ $(LDLIBS)
clean clobber:
	rm -rf $(OBJS) $(TARGETS) $(CU_OBJS) $(CV_CUDA_OBJS) $(PIPELINE_CACHE_OUTPUT) $(DEPS) \
         $(STATUS_MANAGER_OBJS) $(STATUS_MANAGER_DEPS) status_manager
