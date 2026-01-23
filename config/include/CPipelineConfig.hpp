/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CPIPELINECONFIG_HPP
#define CPIPELINECONFIG_HPP

/* Pipeline element format: <ModuleType><ModuleId>.
   ModuleId can be used if you want to define multiple elements with the same type.
   If "*" appears before ModuleType, it indicates the module is a stitching module.
*/
static constexpr const char *sPipelineTable[][2] = {
    // {"pipeline-parameter", "pipeline descriptor in string"}
    // Below are pipeline configurations where the camera connection is not fixed and can be specified by "-m" option.
    // {"elem<elemId></elemNum>=<parameters>"}
    // clang-format off
    /* === intra-process pipelines === */
    { "virtual", "VirtualSrc,Enc;"
                 "VirtualSrc,Cuda;" },
    // <cameras>=>SIPL<->NvStreams<->Enc
    { "enc", "SIPL,Enc;" },
    // <cameras>=>SIPL<->NvStreams<->Cuda
    { "cuda", "SIPL,Cuda;" },
    // <cameras>=>SIPL<->NvStreams<->Nvm2d(color conversion)<->NvStreams<->VulkanSC
    { "vulkansc", "SIPL,Nvm2d,VulkanSC=width=1920:height=1080;" },
    // <cameras>=>SIPL<->NvStreams<->Cuda(passthrough)<->NvStreams<->Enc
    { "passthrough", "SIPL,Cuda=passthrough=1,Enc;" },
    // <cameras> =>SIPL<->multicast|<->NvStreams<->Enc
    //                             |<->NvStreams<->Cuda
    { "multicast", "SIPL,Enc;"
                   "SIPL,Cuda;" },
    // <cameras> =>SIPL<->multicast|<->limiter<->NvStreams<->Enc
    //                             |<->limiter<->NvStreams<->Cuda
    { "multicast_limit", "SIPL,Enc=limit=2;"
                         "SIPL,Cuda=limit=2;" },
    // camera#1=>SIPL<->NvStreams<->|
    // ...                          |<=>Nvm2d(stitching)<->NvStreams<->Enc
    // camera#n=>SIPL<->NvStreams<->|
    { "stitch", "SIPL,*Nvm2d,*Enc;" },
    // camera#1=>SIPL<->NvStreams<->|
    // ...                          |<=>Nvm2d(stitching)<->NvStreams<->VirtualDst
    // camera#n=>SIPL<->NvStreams<->|
    { "stitch_virtual", "SIPL,*Nvm2d,*VirtualDst;" },
    // <cameras>=>SIPL(BL,PL)<->multicast|<->NvStreams<->Enc(BL)
    //                                   |<->NvStreams<->Cuda(PL)
    { "multi_elems", "SIPL=elems=siplMulti,Enc=elems=blMulti;"
                     "SIPL=elems=siplMulti,Cuda=elems=plMulti;" },

    /* === inter-process pipelines === */
    // <cameras>=>SIPL<->IpcSrc<->NvStreams
    { "sipl_ipc", "SIPL,IpcSrc;" },
    // NvStreams<->IpcDst<->Enc
    { "ipc_enc", "IpcDst,Enc;" },
    // NvStreams<->IpcDst<->Cuda
    { "ipc_cuda", "IpcDst,Cuda;" },
    // NvStreams<->Nvm2d(color conversion)<->NvStreams<->VulkanSC
    { "ipc_vulkansc", "IpcDst,Nvm2d,VulkanSC=width=1920:height=1080;" },
    // NvStreams<->IpcDst<->Cuda(passthrough)<->NvStreams<->Enc
    { "ipc_passthrough", "IpcDst,Cuda=passthrough=1,Enc;" },
    // NvStreams<->IpcDst<->|
    // ...                  |<=>Nvm2d(stitching)<->NvStreams<->Enc
    // NvStreams<->IpcDst<->|
    { "ipc_stitch", "IpcDst,*Nvm2d,*Enc;" },
    // NvStreams<->IpcDst<->|
    // ...                  |<=>Nvm2d(stitching)<->NvStreams<->VirtualDst
    // NvStreams<->IpcDst<->|
    { "ipc_stitch_virtual", "IpcDst,*Nvm2d,*VirtualDst;" },
    // <cameras>=>SIPL<->multicast|<->IpcSrc0<->NvStreams
    //                            |<->IpcSrc1<->NvStreams
    { "multicast_ipc", "SIPL,IpcSrc0/2;"
                       "SIPL,IpcSrc1/2;" },
    // <cameras>=>SIPL<->multicast|<->limiter<->IpcSrc0<->NvStreams
    //                            |<->limiter<->IpcSrc1<->NvStreams
    { "multicast_ipc_limit", "SIPL,IpcSrc0/2=limit=2;"
                             "SIPL,IpcSrc1/2=limit=2;" },
    // NvStreams<->IpcDst0<->Enc
    // NvStreams<->IpcDst1<->Cuda
    { "ipc_multicast", "IpcDst0/2,Enc;"
                       "IpcDst1/2,Cuda;" },
    // NvStreams<->IpcDst<->Cuda0(passthrough)<->multicast|<->Enc
    //                                                    |<->Cuda1
    { "ipc_passthrough_multicast", "IpcDst,Cuda0=passthrough=1,Enc;"
                                   "IpcDst,Cuda0=passthrough=1,Cuda1;" },

    /* === inter-chip pipelines === */
    // <cameras>=>SIPL<->IpcSrc<->NvStreams(over PCIe)
    { "sipl_c2c", "SIPL,IpcSrc=c2c=1;" },
    // NvStreams(over PCIe)<->IpcDst<->Enc
    { "c2c_enc", "IpcDst=c2c=1,Enc;" },
    // NvStreams(over PCIe)<->IpcDst<->Cuda
    { "c2c_cuda", "IpcDst=c2c=1,Cuda;" },
    // <cameras>=>SIPL<->multicast|<->IpcSrc0<->NvStreams(over PCIe)
    //                            |<->IpcSrc1<->NvStreams(over PCIe)
    { "multicast_c2c", "SIPL,IpcSrc0/2=c2c=1;"
                       "SIPL,IpcSrc1/2=c2c=1;" },
    // <cameras>=>SIPL<->multicast|<->limiter<->IpcSrc0<->NvStreams(over PCIe)
    //                            |<->limiter<->IpcSrc1<->NvStreams(over PCIe)
    { "multicast_c2c_limit", "SIPL,IpcSrc0/2=c2c=1:limit=2;"
                             "SIPL,IpcSrc1/2=c2c=1:limit=2;" },
    // NvStreams(over PCIe)<->IpcDst0<->Enc
    // NvStreams(over PCIe)<->IpcDst1<->Cuda
    { "c2c_multicast", "IpcDst0/2=c2c=1,Enc;"
                       "IpcDst1/2=c2c=1,Cuda;" },

    /* === late-attach pipelines === */
    // <cameras>=>SIPL<->multicast|<->NvStreams<->Enc(early consumer)
    //                            |<->IpcSrc(late)<->NvStreams
    { "sipl_itc_early_ipc_late", "SIPL=latemods=Cuda,Enc;"
                                 "SIPL,IpcSrc=late=1;" },
    // <cameras>=>SIPL<->multicast|<->NvStreams<->Enc(early consumer)
    //                            |<->IpcSrc(late)<->NvStreams(over PCIe)
    { "sipl_itc_early_c2c_late", "SIPL=latemods=Cuda,Enc;"
                                 "SIPL,IpcSrc=late=1:c2c=1;" },
    // <cameras>=>SIPL<->multicast|<->NvStreams<->VirtualDst(early)
    //                            |<->IpcSrc0(early)<->NvStreams
    //                            |<->IpcSrc1(late)<->NvStreams(over PCIe)
    { "sipl_ipc_early_c2c_late", "SIPL,VirtualDst;"
                                 "SIPL,IpcSrc0/2;"
                                 "SIPL,IpcSrc1/2=late=1:c2c=1;" },
    // <cameras>=>SIPL<->multicast|<->NvStreams<->VirtualDst(early)
    //                            |<->IpcSrc0(early)<->NvStreams(over PCIe)
    //                            |<->IpcSrc1(late)<->NvStreams(over PCIe)
    { "sipl_c2c_early_c2c_late", "SIPL,VirtualDst;"
                                 "SIPL,IpcSrc0/2=c2c=1;"
                                 "SIPL,IpcSrc1/2=late=1:c2c=1;" },
    // NvStreams<->IpcDst0<->Enc
    { "ipc_early_enc", "IpcDst0/2,Enc;"},
    // NvStreams(over PCIe)<->IpcDst0<->Enc(early)
    { "c2c_early_enc", "IpcDst0/2=c2c=1,Enc;" },
    // NvStreams(over PCIe)<->IpcDst1<->Cuda(late)
    { "c2c_late_cuda", "IpcDst1/2=c2c=1,Cuda;" },

    // Below are pipeline configurations where the camera connection is specified.
    // {"elem<elemId></elemNum>_<sensor id list>=<parameters>"}
    // E.g. IpcDst0/2_01 identifies that The elem is IpcDst, whose elemId is 0 and elemNum is 2.
    // The streams from two cameras (corresponding sensor indexes are 0 and 1) pass the elem.

    /* === display pipelines === */
    // <cameras>=>SIPL<->NvStreams<->Display
    { "display_sst_1x", "SIPL_0,Display_0=portid=0;" },
    // camera0=>SIPL<->NvStreams<->Display0
    // camera1=>SIPL<->NvStreams<->Display1
    { "display_mst_2x", "SIPL_01,Display0_0=portid=0;"
                        "SIPL_01,Display1_1=portid=1;" },
    // camera0=>SIPL<->IpcSrc<->NvStreams
    { "display_sst_ipc_1x", "SIPL_0,IpcSrc_0;" },
    // (camera0)NvStreams<->IpcDst<->Display
    { "ipc_display_sst_1x", "IpcDst_0,Display_0=portid=0;" },
    // camera0=>SIPL<->IpcSrc0<->NvStreams
    // camera1=>SIPL<->IpcSrc1<->NvStreams
    { "display_mst_ipc_2x", "SIPL_01,IpcSrc0/2_0;"
                            "SIPL_01,IpcSrc1/2_1;" },
    // (camera0)NvStreams<->IpcDst0<->Display0
    // (camera1)NvStreams<->IpcDst1<->Display1
    { "ipc_display_mst_2x", "IpcDst0/2_0,Display0_0=portid=0;"
                            "IpcDst1/2_1,Display1_1=portid=1;" },

    /* === Pipelines for demostrating car detection feature === */
    // camera0=>SIPL<->NvStreams<->Cuda(with inference)(passthrough)<->NvStreams<->Display
    { "car_detect_1x", "SIPL_0,Cuda_0=passthrough=1,Display_0=portid=0;" },

    //// camera0=>SIPL<->NvStreams<->Cuda(with pva preprocessing and cuda inference)(passthrough)<->NvStreams<->Display
    { "car_detect_pva_1x", "SIPL_0,Nvm2d,Cuda_0=passthrough=1:width=3840:height=2160:usePva=1,Display_0=portid=0;" },

    // <cameras>=>SIPL<->NvStreams<->Pva(passthrough)<->NvStream<->Enc
    { "pva_low_power_mode_2x", "SIPL_01,Pva0_0=passthrough=1:vpuId=0,Enc0_0=filesink=1;"
                               "SIPL_01,Pva1_1=passthrough=1:vpuId=1,Enc1_1=filesink=1;" },

    /* === Pipelines for demostrating car model rendering feature === */
    // camera0, 1, 2, 3=>SIPL<->NvStreams<->VulkanSC(with car model rendering)(passthrough)<->NvStreams<->Display
    { "car_render_vulkansc", "SIPL_0123,*Nvm2d,*VulkanSC=width=1920:height=1536:colortype=ABGR:passthrough=1,*Display=portid=0:width=1920:height=1536:colortype=ABGR;"},

    // Typical complicated pipelines
    // camera0,1,2,3,8,9,A,B,C,D,E,F=>SIPL<->multicast|(camera0,1,2,3)<->NvStreams<->Cuda0(passthrough)<->Enc
    //                                                |(camera8,9,A,B)<->NvStreams<->Cuda1
    //                                                |(cameraC,D,E,F)<->NvStreams<->Nvm2d(stitching)<->Display
    { "pipeline1_12x", "SIPL_012389ABCDEF, Cuda0_0123=passthrough=1,Enc_0123;"
                       "SIPL_012389ABCDEF, Cuda1_89AB;"
                       "SIPL_012389ABCDEF=elems=icp, *Nvm2d_CDEF=elems=icp, *Display_CDEF=portid=0:width=1920:height=1536;"},
    { "pipeline1_ipc_12x", "SIPL_012389ABCDEF, IpcSrc0/3_0123;"
                           "SIPL_012389ABCDEF, IpcSrc1/3_89AB;"
                           "SIPL_012389ABCDEF=elems=icp, IpcSrc2/3_CDEF;"},
    { "ipc_pipeline1_12x", "IpcDst0/3_0123, Cuda0_0123=passthrough=1, Enc_0123;"
                           "IpcDst1/3_89AB, Cuda1_89AB;"
                           "IpcDst2/3_CDEF, *Nvm2d_CDEF=elems=icp, *Display_CDEF=portid=0:width=1920:height=1536;"},
    // clang-format on
};

#endif //CPIPELINECONFIG_HPP
