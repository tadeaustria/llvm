//==---------- spirv_vars.hpp --- SPIRV variables -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===-------------------------------------------------------------------=== //

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __SYCL_DEVICE_ONLY__

#define __SPIRV_VAR_QUALIFIERS extern "C" const

#if defined(__SYCL_NVPTX__) || defined(__SYCL_EXPLICIT_SIMD__)

SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_z();

SYCL_EXTERNAL size_t __spirv_GlobalSize_x();
SYCL_EXTERNAL size_t __spirv_GlobalSize_y();
SYCL_EXTERNAL size_t __spirv_GlobalSize_z();

SYCL_EXTERNAL size_t __spirv_GlobalOffset_x();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_y();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_z();

SYCL_EXTERNAL size_t __spirv_NumWorkgroups_x();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_y();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupSize_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupId_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_z();

SYCL_EXTERNAL size_t __spirv_LocalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_z();

SYCL_EXTERNAL uint32_t __spirv_SubgroupSize();
SYCL_EXTERNAL uint32_t __spirv_SubgroupMaxSize();
SYCL_EXTERNAL uint32_t __spirv_NumSubgroups();
SYCL_EXTERNAL uint32_t __spirv_SubgroupId();
SYCL_EXTERNAL uint32_t __spirv_SubgroupLocalInvocationId();

#else // __SYCL_NVPTX__

#ifdef __SYCL_VULKAN__
// Overwrite spirv qualifiers with global address space
#define __SPIRV_VAR_QUALIFIERS __attribute__((address_space(7))) extern "C" const
// Vulkan explicitly needs a 32-bit vector
typedef uint32_t size_t_vec __attribute__((ext_vector_type(3)));
#else
typedef size_t  size_t_vec __attribute__((ext_vector_type(3)));
#endif
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupSize;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInNumWorkgroups;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInLocalInvocationId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInWorkgroupId;
__SPIRV_VAR_QUALIFIERS size_t_vec __spirv_BuiltInGlobalOffset;

__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupMaxSize;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInNumSubgroups;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupId;
__SPIRV_VAR_QUALIFIERS uint32_t __spirv_BuiltInSubgroupLocalInvocationId;

SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_x() {
  return __spirv_BuiltInGlobalInvocationId.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_y() {
  return __spirv_BuiltInGlobalInvocationId.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalInvocationId_z() {
  return __spirv_BuiltInGlobalInvocationId.z;
}

SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_x() {
  return __spirv_BuiltInNumWorkgroups.x;
}
SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_y() {
  return __spirv_BuiltInNumWorkgroups.y;
}
SYCL_EXTERNAL inline size_t __spirv_NumWorkgroups_z() {
  return __spirv_BuiltInNumWorkgroups.z;
}

SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_x() {
  return __spirv_BuiltInWorkgroupSize.x;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_y() {
  return __spirv_BuiltInWorkgroupSize.y;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupSize_z() {
  return __spirv_BuiltInWorkgroupSize.z;
}

SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_x() {
  return __spirv_BuiltInWorkgroupId.x;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_y() {
  return __spirv_BuiltInWorkgroupId.y;
}
SYCL_EXTERNAL inline size_t __spirv_WorkgroupId_z() {
  return __spirv_BuiltInWorkgroupId.z;
}

SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_x() {
  return __spirv_BuiltInLocalInvocationId.x;
}
SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_y() {
  return __spirv_BuiltInLocalInvocationId.y;
}
SYCL_EXTERNAL inline size_t __spirv_LocalInvocationId_z() {
  return __spirv_BuiltInLocalInvocationId.z;
}

SYCL_EXTERNAL inline uint32_t __spirv_SubgroupSize() {
  return __spirv_BuiltInSubgroupSize;
}
SYCL_EXTERNAL inline uint32_t __spirv_NumSubgroups() {
  return __spirv_BuiltInNumSubgroups;
}
SYCL_EXTERNAL inline uint32_t __spirv_SubgroupId() {
  return __spirv_BuiltInSubgroupId;
}
SYCL_EXTERNAL inline uint32_t __spirv_SubgroupLocalInvocationId() {
  return __spirv_BuiltInSubgroupLocalInvocationId;
}

#ifdef __SYCL_VULKAN__
// Vulkan has no builtin for that
SYCL_EXTERNAL inline uint32_t __spirv_SubgroupMaxSize() {
  return __spirv_BuiltInSubgroupSize; //TODO: find equivalent for SubgroupMaxSize
}

// Vulkan offset is steps of workgroup sizes
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_x() {
  return __spirv_WorkgroupSize_x() * __spirv_BuiltInGlobalOffset.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_y() {
  return __spirv_WorkgroupSize_y() * __spirv_BuiltInGlobalOffset.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_z() {
  return __spirv_WorkgroupSize_z() * __spirv_BuiltInGlobalOffset.z;
}

// Vulkan has no builtin for these so calculate as workaround
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_x() {
  return __spirv_NumWorkgroups_x() * __spirv_WorkgroupSize_x();
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_y() {
  return __spirv_NumWorkgroups_y() * __spirv_WorkgroupSize_y();
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_z() {
  return __spirv_NumWorkgroups_z() * __spirv_WorkgroupSize_z();
}

#else
SYCL_EXTERNAL inline uint32_t __spirv_SubgroupMaxSize() {
  return __spirv_BuiltInSubgroupMaxSize;
}

SYCL_EXTERNAL inline size_t __spirv_GlobalSize_x() {
  return __spirv_BuiltInGlobalSize.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_y() {
  return __spirv_BuiltInGlobalSize.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalSize_z() {
  return __spirv_BuiltInGlobalSize.z;
}

SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_x() {
  return __spirv_BuiltInGlobalOffset.x;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_y() {
  return __spirv_BuiltInGlobalOffset.y;
}
SYCL_EXTERNAL inline size_t __spirv_GlobalOffset_z() {
  return __spirv_BuiltInGlobalOffset.z;
}

#endif // __SYCL_VULKAN__

#endif // __SYCL_NVPTX__

#undef __SPIRV_VAR_QUALIFIERS

namespace __spirv {

// Helper function templates to initialize and get vector component from SPIR-V
// built-in variables
#define __SPIRV_DEFINE_INIT_AND_GET_HELPERS(POSTFIX)                           \
  template <int ID> static size_t get##POSTFIX();                              \
  template <> size_t get##POSTFIX<0>() { return __spirv_##POSTFIX##_x(); }     \
  template <> size_t get##POSTFIX<1>() { return __spirv_##POSTFIX##_y(); }     \
  template <> size_t get##POSTFIX<2>() { return __spirv_##POSTFIX##_z(); }     \
                                                                               \
  template <int Dim, class DstT> struct InitSizesST##POSTFIX;                  \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<1, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<0>()}; }                     \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<2, DstT> {                 \
    static DstT initSize() { return {get##POSTFIX<1>(), get##POSTFIX<0>()}; }  \
  };                                                                           \
                                                                               \
  template <class DstT> struct InitSizesST##POSTFIX<3, DstT> {                 \
    static DstT initSize() {                                                   \
      return {get##POSTFIX<2>(), get##POSTFIX<1>(), get##POSTFIX<0>()};        \
    }                                                                          \
  };                                                                           \
                                                                               \
  template <int Dims, class DstT> static DstT init##POSTFIX() {                \
    return InitSizesST##POSTFIX<Dims, DstT>::initSize();                       \
  }

__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalSize);
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(WorkgroupSize)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(NumWorkgroups)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(LocalInvocationId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(WorkgroupId)
__SPIRV_DEFINE_INIT_AND_GET_HELPERS(GlobalOffset)

#undef __SPIRV_DEFINE_INIT_AND_GET_HELPERS

} // namespace __spirv

#endif // __SYCL_DEVICE_ONLY__
