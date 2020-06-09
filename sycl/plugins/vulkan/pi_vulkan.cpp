//==---------- pi_vulkan.cpp - Vulkan Plugin -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \defgroup sycl_pi_vulkan Vulkan Plugin
/// \ingroup sycl_pi

/// \file pi_vulkan.cpp
/// Implementation of Vulkan Plugin. It is the interface between device-agnostic
/// SYCL runtime layer and underlying Vulkan runtime.
///
/// \ingroup sycl_pi_vulkan

#include "pi_vulkan.hpp"
#include <CL/sycl/detail/pi.hpp>

#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#define CHECK_ERR_SET_NULL_RET(err, ptr, reterr)                               \
  if (err != CL_SUCCESS) {                                                     \
    if (ptr != nullptr)                                                        \
      *ptr = nullptr;                                                          \
    return cast<pi_result>(reterr);                                            \
  }

const char SupportedVersion[] = _PI_H_VERSION_STRING;

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value) {
  // TODO: see if more sanity checks are possible.
  static_assert(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
}

//// Older versions of GCC don't like "const" here
//#if defined(__GNUC__) && (__GNUC__ < 7 || (__GNU__C == 7 && __GNUC_MINOR__ <
//2)) #define CONSTFIX constexpr #else #define CONSTFIX const #endif
//
//// Names of USM functions that are queried from OpenCL
// CONSTFIX char clHostMemAllocName[] = "clHostMemAllocINTEL";
// CONSTFIX char clDeviceMemAllocName[] = "clDeviceMemAllocINTEL";
// CONSTFIX char clSharedMemAllocName[] = "clSharedMemAllocINTEL";
// CONSTFIX char clMemFreeName[] = "clMemFreeINTEL";
// CONSTFIX char clSetKernelArgMemPointerName[] =
// "clSetKernelArgMemPointerINTEL"; CONSTFIX char clEnqueueMemsetName[] =
// "clEnqueueMemsetINTEL"; CONSTFIX char clEnqueueMemcpyName[] =
// "clEnqueueMemcpyINTEL"; CONSTFIX char clEnqueueMigrateMemName[] =
// "clEnqueueMigrateMemINTEL"; CONSTFIX char clEnqueueMemAdviseName[] =
// "clEnqueueMemAdviseINTEL"; CONSTFIX char clGetMemAllocInfoName[] =
// "clGetMemAllocInfoINTEL"; CONSTFIX char
// clSetProgramSpecializationConstantName[] =
//    "clSetProgramSpecializationConstant";
//
//#undef CONSTFIX

// USM helper function to get an extension function pointer
template <const char *FuncName, typename T>
static pi_result getExtFuncFromContext(pi_context context, T *fptr) {
  // TODO
  // Potentially redo caching as PI interface changes.
  thread_local static std::map<pi_context, T> FuncPtrs;

  // if cached, return cached FuncPtr
  if (auto F = FuncPtrs[context]) {
    *fptr = F;
    return PI_SUCCESS;
  }

  size_t deviceCount;
  cl_int ret_err = clGetContextInfo(
      cast<cl_context>(context), CL_CONTEXT_DEVICES, 0, nullptr, &deviceCount);

  if (ret_err != CL_SUCCESS || deviceCount < 1) {
    return PI_INVALID_CONTEXT;
  }

  std::vector<cl_device_id> devicesInCtx(deviceCount);
  ret_err = clGetContextInfo(cast<cl_context>(context), CL_CONTEXT_DEVICES,
                             deviceCount * sizeof(cl_device_id),
                             devicesInCtx.data(), nullptr);

  if (ret_err != CL_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  cl_platform_id curPlatform;
  ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
                            sizeof(cl_platform_id), &curPlatform, nullptr);

  if (ret_err != CL_SUCCESS) {
    return PI_INVALID_CONTEXT;
  }

  T FuncPtr =
      (T)clGetExtensionFunctionAddressForPlatform(curPlatform, FuncName);

  if (!FuncPtr)
    return PI_INVALID_VALUE;

  *fptr = FuncPtr;
  FuncPtrs[context] = FuncPtr;

  return cast<pi_result>(ret_err);
}

/// Enables indirect access of pointers in kernels.
/// Necessary to avoid telling CL about every pointer that might be used.
///
/// \param kernel is the kernel to be launched
static pi_result USMSetIndirectAccess(pi_kernel kernel) {
  // We test that each alloc type is supported before we actually try to
  // set KernelExecInfo.
  cl_bool TrueVal = CL_TRUE;
  clHostMemAllocINTEL_fn HFunc = nullptr;
  clSharedMemAllocINTEL_fn SFunc = nullptr;
  clDeviceMemAllocINTEL_fn DFunc = nullptr;
  cl_context CLContext;
  cl_int CLErr = clGetKernelInfo(cast<cl_kernel>(kernel), CL_KERNEL_CONTEXT,
                                 sizeof(cl_context), &CLContext, nullptr);
  if (CLErr != CL_SUCCESS) {
    return cast<pi_result>(CLErr);
  }

  getExtFuncFromContext<clHostMemAllocName, clHostMemAllocINTEL_fn>(
      cast<pi_context>(CLContext), &HFunc);
  if (HFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }

  getExtFuncFromContext<clDeviceMemAllocName, clDeviceMemAllocINTEL_fn>(
      cast<pi_context>(CLContext), &DFunc);
  if (DFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }

  getExtFuncFromContext<clSharedMemAllocName, clSharedMemAllocINTEL_fn>(
      cast<pi_context>(CLContext), &SFunc);
  if (SFunc) {
    clSetKernelExecInfo(cast<cl_kernel>(kernel),
                        CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                        sizeof(cl_bool), &TrueVal);
  }
  return PI_SUCCESS;
}

pi_result mapVulkanErrToCLErr(vk::Result result) {
  switch (result) {
  case vk::Result::eSuccess:
    return pi_result::PI_SUCCESS;
  case vk::Result::eErrorInvalidExternalHandle:
    return pi_result::PI_INVALID_MEM_OBJECT;
  default:
    return pi_result::PI_ERROR_UNKNOWN;
  }
}

pi_result mapVulkanErrToCLErr(VkResult result) {
  switch (result) {
  case VkResult::VK_SUCCESS:
    return pi_result::PI_SUCCESS;
  default:
    return pi_result::PI_ERROR_UNKNOWN;
  }
}

pi_result mapVulkanErrToCLErr(const vk::SystemError &result) {
  return mapVulkanErrToCLErr(static_cast<vk::Result>(result.code().value()));
}

VkResult getBestComputeQueueNPH(vk::PhysicalDevice &physicalDevice,
                                uint32_t &queueFamilyIndex) {

  auto properties = physicalDevice.getQueueFamilyProperties();
  int i = 0;
  for (auto prop : properties) {
    vk::QueueFlags maskedFlags =
        (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) &
         prop.queueFlags);
    if (!(vk::QueueFlagBits::eGraphics & maskedFlags) &&
        (vk::QueueFlagBits::eCompute & maskedFlags)) {
      queueFamilyIndex = i;
      return VK_SUCCESS;
    }
    i++;
  }
  i = 0;
  for (auto prop : properties) {
    vk::QueueFlags maskedFlags =
        (~(vk::QueueFlagBits::eTransfer | vk::QueueFlagBits::eSparseBinding) &
         prop.queueFlags);
    if (vk::QueueFlagBits::eCompute & maskedFlags) {
      queueFamilyIndex = i;
      return VK_SUCCESS;
    }
    i++;
  }
  return VK_ERROR_INITIALIZATION_FAILED;
}

namespace {
/// \cond NODOXY
template <typename T, typename Assign>
pi_result getInfoImpl(size_t param_value_size, void *param_value,
                      size_t *param_value_size_ret, T value, size_t value_size,
                      Assign &&assign_func) {

  if (param_value != nullptr) {

    if (param_value_size < value_size) {
      return PI_INVALID_VALUE;
    }

    assign_func(param_value, value, value_size);
  }

  if (param_value_size_ret != nullptr) {
    *param_value_size_ret = value_size;
  }

  return PI_SUCCESS;
}

template <typename T>
pi_result getInfo(size_t param_value_size, void *param_value,
                  size_t *param_value_size_ret, T value) {

  auto assignment = [](void *param_value, T value, size_t value_size) {
    *static_cast<T *>(param_value) = value;
  };

  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     sizeof(T), assignment);
}

template <typename T>
pi_result getInfoArray(size_t array_length, size_t param_value_size,
                       void *param_value, size_t *param_value_size_ret,
                       T *value) {
  return getInfoImpl(param_value_size, param_value, param_value_size_ret, value,
                     array_length * sizeof(T), memcpy);
}

template <>
pi_result getInfo<const char *>(size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret,
                                const char *value) {
  return getInfoArray(strlen(value) + 1, param_value_size, param_value,
                      param_value_size_ret, value);
}
//
// int getAttribute(pi_device device, CUdevice_attribute attribute) {
//  int value;
//  cl::sycl::detail::pi::assertion(
//      cuDeviceGetAttribute(&value, attribute, device->get()) == VLK(SUCCESS);
//  return value;
//}
/// \endcond

} // anonymous namespace

/// ------ Error handling, matching OpenCL plugin semantics.
__SYCL_INLINE_NAMESPACE(cl) {
  namespace sycl {
  namespace detail {
  namespace pi {

  // Report error and no return (keeps compiler from printing warnings).
  // TODO: Probably change that to throw a catchable exception,
  //       but for now it is useful to see every failure.
  //
  [[noreturn]] void die(const char *Message) {
    std::cerr << "pi_die: " << Message << std::endl;
    std::terminate();
  }

  // void assertion(bool Condition, const char *Message) {
  //  if (!Condition)
  //    die(Message);
  //}

  } // namespace pi
  } // namespace detail
  } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

pi_result _pi_kernel::addArgument(pi_uint32 ArgIndex, pi_mem Memobj) {

  if (DescriptorSetLayoutBinding.size() <= ArgIndex) {
    DescriptorSetLayoutBinding.resize(ArgIndex + 1);
    Arguments.resize(ArgIndex + 1);
  }

  DescriptorSetLayoutBinding[ArgIndex] = {ArgIndex,
                                          vk::DescriptorType::eStorageBuffer, 1,
                                          vk::ShaderStageFlagBits::eCompute};

  Arguments[ArgIndex] = Memobj;

  return PI_SUCCESS;
}

extern "C" {

// Convenience macro makes source code search easier
#define VLK(pi_api) Vulkan##pi_api

#define NOT_IMPL(functionname, parameters)                                     \
  functionname parameters {                                                    \
    cl::sycl::detail::pi::die("functionname not implemented");                 \
    return {};                                                                 \
  }

// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result VLK(piPlatformsGet)(pi_uint32 num_entries, pi_platform *platforms,
                              pi_uint32 *num_platforms) {

  try {
    static std::once_flag InitFlag;
    static pi_uint32 NumPlatforms = 1;
    // global Vulkan Instance
    static _pi_platform Platform;

    if (num_entries == 0 && platforms != nullptr) {
      return PI_INVALID_VALUE;
    }
    if (platforms == nullptr && num_platforms == nullptr) {
      return PI_INVALID_VALUE;
    }

    pi_result err = PI_SUCCESS;

    std::call_once(
        InitFlag,
        [](pi_result &err) {
          /*               if (cuInit(0) != VLK(SUCCESS) {
                           NumPlatforms = 0;
                           return;
                         }*/

          // initialize the vk::ApplicationInfo structure
          vk::ApplicationInfo applicationInfo("SYCL LLVM", 1, "pi_vulkan.cpp",
                                              1, VK_API_VERSION_1_2);

          // initialize the vk::InstanceCreateInfo
          char *List[2];
          // List[0] = "VK_LAYER_LUNARG_api_dump";
          // List[1] = "VK_LAYER_KHRONOS_validation ";
          vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, 0,
                                                    List);
          Platform.Instance_ = vk::createInstance(instanceCreateInfo);

          /*VkApplicationInfo applInfo = {
              .sType =
          VkStructureType::VK_STRUCTURE_TYPE_APPLICATION_INFO,
              .pApplicationName = "SYCL LLVM",
              .applicationVersion = 1,
              .pEngineName = "pi_vulkan.cpp",
              .engineVersion = 1,
              .apiVersion = VK_API_VERSION_1_2};

          VkInstanceCreateInfo createInfo;
          vkCreateInstance(&createInfo, nullptr, &instance);*/

          // int numDevices = 0;
          // err = PI_CHECK_ERROR(cuDeviceGetCount(&numDevices));
          // if (numDevices == 0) {
          //  NumPlatforms = 0;
          //  return;
          //}
          // try {
          //  platformId.devices_.reserve(numDevices);
          //  for (int i = 0; i < numDevices; ++i) {
          //    CUdevice device;
          //    err = PI_CHECK_ERROR(cuDeviceGet(&device, i));
          //    platformId.devices_.emplace_back(
          //        new _pi_device{device, &platformId});
          //  }
          //} catch (const std::bad_alloc &) {
          //  // Signal out-of-memory situation
          //  platformId.devices_.clear();
          //  err = PI_OUT_OF_HOST_MEMORY;
          //} catch (...) {
          //  // Clear and rethrow to allow retry
          //  platformId.devices_.clear();
          //  throw;
          //}
        },
        err);

    if (num_platforms != nullptr) {
      *num_platforms = NumPlatforms;
    }

    if (platforms != nullptr) {
      *platforms = &Platform;
    }

    return err;
  } catch (pi_result err) {
    return err;
  } catch (...) {
    return PI_OUT_OF_RESOURCES;
  }
}

pi_result VLK(piPlatformGetInfo)(pi_platform platform,
                                 pi_platform_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {

  switch (param_name) {
  case PI_PLATFORM_INFO_NAME:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "PI_VULKAN");
  case PI_PLATFORM_INFO_PROFILE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "EMBEDDED_PROFILE");
  case PI_PLATFORM_INFO_VENDOR:
    return getInfo(param_value_size, param_value, param_value_size_ret, "None");
  case PI_PLATFORM_INFO_VERSION:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "Vulkan 1.0");
  case PI_PLATFORM_INFO_EXTENSIONS:
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result VLK(piDevicesGet)(pi_platform platform, pi_device_type device_type,
                            pi_uint32 num_entries, pi_device *devices,
                            pi_uint32 *num_devices) {
  /*VkResult result = vkEnumeratePhysicalDevices(
      Platform->Instance_, num_devices, cast<VkPhysicalDevice *>(devices));*/
  auto DevicesVec = platform->Instance_.enumeratePhysicalDevices();

  auto RelevantDevicesEnd =
      std::remove_if(DevicesVec.begin(), DevicesVec.end(),
                     [device_type](vk::PhysicalDevice &Device) {
                       switch (Device.getProperties().deviceType) {
                       case vk::PhysicalDeviceType::eCpu:
                         if (!(device_type & PI_DEVICE_TYPE_CPU))
                           return true;
                         break;
                       case vk::PhysicalDeviceType::eDiscreteGpu:
                       case vk::PhysicalDeviceType::eIntegratedGpu:
                       case vk::PhysicalDeviceType::eVirtualGpu:
                         if (!(device_type & PI_DEVICE_TYPE_GPU))
                           return true;
                         break;
                       default:
                         return true;
                       }
                       return false;
                     });

  uint32_t SizeRelevantDevices =
      std::distance(DevicesVec.begin(), RelevantDevicesEnd);
  if (devices) {
    uint32_t MaxEntries = std::min<uint32_t>(num_entries, SizeRelevantDevices);
    std::vector<vk::PhysicalDevice>::iterator Device = DevicesVec.begin();
    for (uint32_t i = 0; i < MaxEntries; i++, Device++) {
      devices[i] = new _pi_device(*Device, platform);
    }
  }
  if (num_devices) {
    *num_devices = SizeRelevantDevices;
  }
  return PI_SUCCESS;
}

pi_result VLK(piDeviceGetInfo)(pi_device Device, pi_device_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {
  auto Properties = Device->PhDevice.getProperties();
  auto Chain = Device->PhDevice.getProperties2<
      vk::PhysicalDeviceProperties2, vk::PhysicalDeviceMaintenance3Properties,
      vk::PhysicalDeviceShaderCorePropertiesAMD>();

  static constexpr pi_uint32 MaxWorkItemDims = 3u;

  assert(Device != nullptr);

  switch (param_name) {
  case PI_DEVICE_INFO_TYPE: {
    switch (Properties.deviceType) {
    case vk::PhysicalDeviceType::eDiscreteGpu:
    case vk::PhysicalDeviceType::eIntegratedGpu:
    case vk::PhysicalDeviceType::eVirtualGpu:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     PI_DEVICE_TYPE_GPU);
    case vk::PhysicalDeviceType::eCpu:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     PI_DEVICE_TYPE_CPU);
    default:
      return PI_INVALID_DEVICE;
    }
  }
  case PI_DEVICE_INFO_VENDOR_ID: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Properties.vendorID);
  }
  case PI_DEVICE_INFO_MAX_COMPUTE_UNITS: {

    auto CoreProperties =
        Chain.get<vk::PhysicalDeviceShaderCorePropertiesAMD>();
    // TODO: check if this is correct
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint32(CoreProperties.computeUnitsPerShaderArray));
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   MaxWorkItemDims);
  }
  case PI_DEVICE_INFO_MAX_WORK_ITEM_SIZES: {
    auto Limits = Properties.limits;
    return getInfoArray(MaxWorkItemDims, param_value_size, param_value,
                        param_value_size_ret, Limits.maxComputeWorkGroupSize);
  }
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    auto Limits = Properties.limits;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(Limits.maxComputeWorkGroupCount[0] +
                          Limits.maxComputeWorkGroupCount[1] +
                          Limits.maxComputeWorkGroupCount[2]));
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_PREFERRED_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  }
  case PI_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_MAX_CLOCK_FREQUENCY: {
    //????
  }
  case PI_DEVICE_INFO_ADDRESS_BITS: {
    auto Bits = pi_uint32{std::numeric_limits<uintptr_t>::digits};
    return getInfo(param_value_size, param_value, param_value_size_ret, Bits);
  }
  case PI_DEVICE_INFO_MAX_MEM_ALLOC_SIZE: {
    // Max size of memory object allocation in bytes.
    // The minimum value is max(min(1024 � 1024 �
    // 1024, 1/4th of CL_DEVICE_GLOBAL_MEM_SIZE),
    // 32 � 1024 � 1024) for devices that are not of type
    // CL_DEVICE_TYPE_CUSTOM.

    return getInfo(
        param_value_size, param_value, param_value_size_ret,
        pi_uint64{Chain.get<vk::PhysicalDeviceMaintenance3Properties>()
                      .maxMemoryAllocationSize});
  }
  case PI_DEVICE_INFO_IMAGE_SUPPORT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_FALSE);
  }
  case PI_DEVICE_INFO_MAX_READ_IMAGE_ARGS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0);
  }
  case PI_DEVICE_INFO_MAX_WRITE_IMAGE_ARGS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_HEIGHT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE2D_MAX_WIDTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_HEIGHT: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_WIDTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE3D_MAX_DEPTH: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_IMAGE_MAX_ARRAY_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(0));
  }
  case PI_DEVICE_INFO_MAX_SAMPLERS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_MAX_PARAMETER_SIZE: {
    // does this belong to entry point parameter? Well SPIRV does not
    // allow parameters for entry points
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{0});
  }
  case PI_DEVICE_INFO_MEM_BASE_ADDR_ALIGN: {
    auto limits = Properties.limits;
    // TODO: Which one exactly? minStorageBufferOffsetAlignment,
    // minUniformBufferOffsetAlignment or minStorageBufferOffsetAlignment
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   limits.minStorageBufferOffsetAlignment);
  }
  case PI_DEVICE_INFO_HALF_FP_CONFIG: {
    // TODO: copied from CUDA
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_SINGLE_FP_CONFIG: {
    // TODO: copied from CUDA
    auto config = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                  CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA |
                  CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_DOUBLE_FP_CONFIG: {
    // TODO: copied from CUDA
    auto config = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                  CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA;
    return getInfo(param_value_size, param_value, param_value_size_ret, config);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_TYPE: {
    // TODO: copied from CUDA
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   CL_READ_WRITE_CACHE);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHELINE_SIZE: {
    // TODO: How to find this value???
    return getInfo(param_value_size, param_value, param_value_size_ret, 128u);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_CACHE_SIZE: {
    // TODO: How to find this value???
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_GLOBAL_MEM_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{Properties.limits.maxComputeSharedMemorySize});
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_BUFFER_SIZE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   pi_uint64{Properties.limits.maxPushConstantsSize});
  }
  case PI_DEVICE_INFO_MAX_CONSTANT_ARGS: {
    // TODO: How to find this value???
    return getInfo(param_value_size, param_value, param_value_size_ret, 9u);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_TYPE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_DEVICE_LOCAL_MEM_TYPE_LOCAL);
  }
  case PI_DEVICE_INFO_LOCAL_MEM_SIZE: {
    auto MemoryProperties = Device->PhDevice.getMemoryProperties();
    uint64_t MemorySize = 0;
    for (uint32_t i = 0; i < MemoryProperties.memoryHeapCount; i++) {
      if (MemoryProperties.memoryHeaps[i].flags &
          vk::MemoryHeapFlagBits::eDeviceLocal)
        MemorySize += MemoryProperties.memoryHeaps[i].size;
    }
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   MemorySize);
  }
  case PI_DEVICE_INFO_ERROR_CORRECTION_SUPPORT: {
    // TODO: How to find this value???
    return getInfo(param_value_size, param_value, param_value_size_ret, false);
  }
  case PI_DEVICE_INFO_HOST_UNIFIED_MEMORY: {

    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Properties.deviceType ==
                       vk::PhysicalDeviceType::eIntegratedGpu);
  }
  case PI_DEVICE_INFO_PROFILING_TIMER_RESOLUTION: {
    // TODO: How to find this value???
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1000u});
  }
  case PI_DEVICE_INFO_ENDIAN_LITTLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_COMPILER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_LINKER_AVAILABLE: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_EXECUTION_CAPABILITIES: {
    // well is this true?
    auto Capability = CL_EXEC_KERNEL;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_DEVICE_PROPERTIES: {
    // The mandated minimum capability:
    auto Capability =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Capability);
  }
  case PI_DEVICE_INFO_QUEUE_ON_HOST_PROPERTIES: {
    // The mandated minimum capability:
    auto Capability = CL_QUEUE_PROFILING_ENABLE;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Capability);
  }
  case PI_DEVICE_INFO_BUILT_IN_KERNELS: {
    // An empty string is returned if no built-in kernels are supported by the
    // device.
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PLATFORM: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Device->Platform_);
  }
  case PI_DEVICE_INFO_NAME: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   Properties.deviceName);
  }
  case PI_DEVICE_INFO_VENDOR: {
    *param_value_size_ret =
        snprintf(cast<char *>(param_value), param_value_size, "%x",
                 Properties.vendorID) +
        1;
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_DRIVER_VERSION: {
    *param_value_size_ret =
        snprintf(cast<char *>(param_value), param_value_size, "%d.%d",
                 VK_VERSION_MAJOR(Properties.driverVersion),
                 VK_VERSION_MINOR(Properties.driverVersion)) +
        1;
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_PROFILE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "VULKAN");
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  }
  case PI_DEVICE_INFO_VERSION: {
    *param_value_size_ret =
        snprintf(cast<char *>(param_value), param_value_size, "VULKAN %d.%d",
                 VK_VERSION_MAJOR(Properties.apiVersion),
                 VK_VERSION_MINOR(Properties.apiVersion)) +
        1;
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_OPENCL_C_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_EXTENSIONS: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_PRINTF_BUFFER_SIZE: {
    // The minimum value for the FULL profile is 1 MB.
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t{1024u});
  }
  case PI_DEVICE_INFO_PREFERRED_INTEROP_USER_SYNC: {
    return getInfo(param_value_size, param_value, param_value_size_ret, true);
  }
  case PI_DEVICE_INFO_PARENT_DEVICE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   nullptr);
  }
  case PI_DEVICE_INFO_PARTITION_MAX_SUB_DEVICES: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_PROPERTIES: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }
  case PI_DEVICE_INFO_PARTITION_AFFINITY_DOMAIN: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 0u);
  }
  case PI_DEVICE_INFO_PARTITION_TYPE: {
    // TODO: uncouple from OpenCL
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   static_cast<cl_device_partition_property>(0u));
  }

    // Intel USM extensions

  case PI_DEVICE_INFO_USM_HOST_SUPPORT: {
    // from cl_intel_unified_shared_memory: "The host memory access capabilities
    // apply to any host allocation."
    //
    // query if/how the device can access page-locked host memory, possibly
    // through PCIe, using the same pointer as the host
    // TODO: How to find this value???
    pi_bitfield Value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, Value);
  }
  case PI_DEVICE_INFO_USM_DEVICE_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The device memory access capabilities apply to any device allocation
    // associated with this device."
    //
    // query how the device can access memory allocated on the device itself (?)
    // TODO: How to find this value???
    pi_bitfield Value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, Value);
  }
  case PI_DEVICE_INFO_USM_SINGLE_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The single device shared memory access capabilities apply to any shared
    // allocation associated with this device."
    //
    // query if/how the device can access managed memory associated to it
    // TODO: How to find this value???
    pi_bitfield Value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, Value);
  }
  case PI_DEVICE_INFO_USM_CROSS_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The cross-device shared memory access capabilities apply to any shared
    // allocation associated with this device, or to any shared memory
    // allocation on another device that also supports the same cross-device
    // shared memory access capability."
    //
    // query if/how the device can access managed memory associated to other
    // devices
    // TODO: How to find this value???
    pi_bitfield Value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, Value);
  }
  case PI_DEVICE_INFO_USM_SYSTEM_SHARED_SUPPORT: {
    // from cl_intel_unified_shared_memory:
    // "The shared system memory access capabilities apply to any allocations
    // made by a system allocator, such as malloc or new."
    //
    // query if/how the device can access pageable host memory allocated by the
    // system allocator
    // TODO: How to find this value???
    pi_bitfield Value = {};
    return getInfo(param_value_size, param_value, param_value_size_ret, Value);
  }

  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Device info request not implemented");
  return {};
}

/// \return PI_SUCCESS if the function is executed successfully
/// CUDA devices are always root devices so retain always returns success.
pi_result VLK(piDeviceRetain)(pi_device Device) { return PI_SUCCESS; }

/// Not applicable to VULKAN, devices cannot be partitioned.
///
pi_result VLK(piDevicePartition)(
    pi_device Device,
    const cl_device_partition_property *properties, // TODO: untie from OpenCL
    pi_uint32 num_devices, pi_device *out_devices, pi_uint32 *out_num_devices) {
  return {};
}

/// \return PI_SUCCESS always since CUDA devices are always root devices.
///
pi_result VLK(piDeviceRelease)(pi_device Device) {
  free(Device);
  return PI_SUCCESS;
}

/// Gets the native handle of a PI device object.
///
/// \param device is the PI device to get the native handle of.
/// \param nativeHandle is the native handle of device.
pi_result VLK(piextDeviceGetNativeHandle)(pi_device Device,
                                          pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle = reinterpret_cast<pi_native_handle>(&Device->PhDevice);
  return PI_SUCCESS;
}

/// Creates PI device object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI device from.
/// \param device is the PI device created from the native handle.
pi_result VLK(piextDeviceCreateWithNativeHandle)(pi_native_handle nativeHandle,
                                                 pi_device *Device) {
  cl::sycl::detail::pi::die(
      "vulkan_piextDeviceCreateWithNativeHandle not implemented");
  return {};
}

/// \return If available, the first binary that is Vulkan Compatible
///
pi_result VLK(piextDeviceSelectBinary)(pi_device Device,
                                       pi_device_binary *binaries,
                                       pi_uint32 num_binaries,
                                       pi_device_binary *selected_binary) {
  if (!binaries) {
    cl::sycl::detail::pi::die("No list of device images provided");
  }
  if (num_binaries < 1) {
    cl::sycl::detail::pi::die("No binary images in the list");
  }
  if (!selected_binary) {
    cl::sycl::detail::pi::die("No storage for device binary provided");
  }

  // Look for an image for the NVPTX64 target, and return the first one that is
  // found
  for (pi_uint32 i = 0; i < num_binaries; i++) {
    if (strcmp(binaries[i]->DeviceTargetSpec,
               PI_DEVICE_BINARY_TARGET_SPIRV64_VULKAN) == 0) {
      *selected_binary = binaries[i];
      return PI_SUCCESS;
    }
  }

  // No image can be loaded for the given device
  return PI_INVALID_BINARY;
}

pi_result VLK(piextGetDeviceFunctionPointer)(pi_device Device,
                                             pi_program program,
                                             const char *function_name,
                                             pi_uint64 *function_pointer_ret) {
  cl::sycl::detail::pi::die(
      "VLK(piextGetDeviceFunctionPointer not implemented");
  return {};
}

/// Create a PI VULKAN context.
///
///
/// \param[in] properties 0 terminated array of key/id-value combinations. Can
/// be nullptr.
/// \param[in] num_devices Number of devices to create the context for.
/// \param[in] devices Devices to create the context for.
/// \param[in] pfn_notify Callback, currently unused.
/// \param[in] user_data User data for callback.
/// \param[out] retcontext Set to created context on success.
///
/// \return PI_SUCCESS on success, otherwise an error return code.
pi_result VLK(piContextCreate)(const pi_context_properties *properties,
                               pi_uint32 num_devices, const pi_device *devices,
                               void (*pfn_notify)(const char *errinfo,
                                                  const void *private_info,
                                                  size_t cb, void *user_data1),
                               void *user_data, pi_context *retcontext) {
  assert(devices != nullptr);
  // TODO: How to implement context callback?
  assert(pfn_notify == nullptr);
  assert(user_data == nullptr);
  assert(num_devices == 1);
  // Need input context
  assert(retcontext != nullptr);
  pi_result Errcode_ret = PI_SUCCESS;

  // What to do with these properties. How does CUDA got its own property?
  //// Parse properties.
  // bool property_VLK(primary = false;
  // while (properties && (0 != *properties)) {
  //  // Consume property ID.
  //  pi_context_properties id = *properties;
  //  ++properties;
  //  // Consume property value.
  //  pi_context_properties value = *properties;
  //  ++properties;
  //  switch (id) {
  //  case PI_CONTEXT_PROPERTIES_VLK(PRIMARY:
  //    assert(value == PI_FALSE || value == PI_TRUE);
  //    property_VLK(primary = static_cast<bool>(value);
  //    break;
  //  default:
  //    // Unknown property.
  //    assert(!"Unknown piContextCreate property in property list");
  //    return PI_INVALID_VALUE;
  //  }
  //}

  auto Device = devices[0];
  auto PhysicalDevice = Device->PhDevice;

  // get the QueueFamilyProperties of the first PhysicalDevice
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      PhysicalDevice.getQueueFamilyProperties();

  uint32_t ComputeQueueFamilyIndex = 0;

  // get the best index into queueFamiliyProperties which supports compute and
  // stuff
  getBestComputeQueueNPH(PhysicalDevice, ComputeQueueFamilyIndex);

  // create a UniqueDevice
  float QueuePriority = 0.0f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
      vk::DeviceQueueCreateFlags(),
      static_cast<uint32_t>(ComputeQueueFamilyIndex), 1, &QueuePriority);

  pi_context Context = new _pi_context();
  Context->RefCounter_ = 1;
  Context->PhDevice_ = Device;
  Context->ComputeQueueFamilyIndex = ComputeQueueFamilyIndex;
  Context->Device = PhysicalDevice.createDevice(
      vk::DeviceCreateInfo(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo));

  *retcontext = Context;
  return Errcode_ret;
}

pi_result VLK(piContextGetInfo)(pi_context context, pi_context_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_CONTEXT_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  case PI_CONTEXT_INFO_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->PhDevice_);
  case PI_CONTEXT_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   context->RefCounter_);
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }

  return PI_OUT_OF_RESOURCES;
}

NOT_IMPL(pi_result VLK(piextContextSetExtendedDeleter),
         (pi_context context, pi_context_extended_deleter func,
          void *user_data))

/// Gets the native handle of a PI context object.
///
/// \param context is the PI context to get the native handle of.
/// \param nativeHandle is the native handle of context.
pi_result VLK(piextContextGetNativeHandle)(pi_context context,
                                           pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle = reinterpret_cast<pi_native_handle>(&context->Device);
  return PI_SUCCESS;
}

/// Creates PI context object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI context from.
/// \param context is the PI context created from the native handle.
NOT_IMPL(pi_result VLK(piextContextCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_context *context))

pi_result VLK(piContextRetain)(pi_context context) {
  context->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piContextRelease)(pi_context context) {
  context->RefCounter_--;
  if (context->RefCounter_ < 1)
    free(context);
  return PI_SUCCESS;
}

pi_result VLK(piQueueCreate)(pi_context context, pi_device Device,
                             pi_queue_properties properties, pi_queue *Queue) {
  assert(Queue && "piQueueCreate failed, queue argument is null");

  /*newQueue->commandBuffer_ =
      context->device.createCommandPoolUnique(vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlags(), context->computeQueueFamilyIndex));*/
  // TODO: think about incrementing Queue Index

  auto CommandPool =
      context->Device.createCommandPoolUnique(vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlags(), context->ComputeQueueFamilyIndex));

  *Queue = new _pi_queue(
      context->Device.getQueue(context->ComputeQueueFamilyIndex, 0),
      std::move(context->Device
                    .allocateCommandBuffers(vk::CommandBufferAllocateInfo(
                        CommandPool.get(), vk::CommandBufferLevel::ePrimary, 1))
                    .front()),
      context, properties);
  return PI_SUCCESS;
}

pi_result VLK(piQueueGetInfo)(pi_queue command_queue, pi_queue_info param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret) {
  assert(command_queue != nullptr);

  switch (param_name) {
  case PI_QUEUE_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->Context_);
  case PI_QUEUE_INFO_DEVICE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->Context_->PhDevice_);
  case PI_QUEUE_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->RefCounter_);
  case PI_QUEUE_INFO_PROPERTIES:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   command_queue->Properties_);
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Queue info request not implemented");
  return {};
}

pi_result VLK(piQueueRetain)(pi_queue Queue) {
  Queue->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piQueueRelease)(pi_queue Queue) {
  Queue->RefCounter_--;
  if (Queue->RefCounter_ < 1)
    free(Queue);
  return PI_SUCCESS;
}

pi_result VLK(piQueueFinish)(pi_queue command_queue) {
  // TODO: Does this belong here?

  /*auto commandPool = command_queue->context_->device.createCommandPoolUnique(
      vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlags(),
          command_queue->context_->computeQueueFamilyIndex));

  auto CommandBuffer = std::move(
      command_queue->context_->device
          .allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
              commandPool.get(), vk::CommandBufferLevel::ePrimary, 1))
          .front());

  CommandBuffer->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));

  CommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.get());

  CommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                    pipelineLayout.get(), 0, 1,
                                    &descriptorSet.get(), 0, nullptr);

  CommandBuffer->dispatch(bufferSize / sizeof(int32_t), 1, 1);

  CommandBuffer->end();*/
  return PI_ERROR_UNKNOWN;
}

/// Gets the native handle of a PI queue object.
///
/// \param queue is the PI queue to get the native handle of.
/// \param nativeHandle is the native handle of queue.
pi_result VLK(piextQueueGetNativeHandle)(pi_queue Queue,
                                         pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle = reinterpret_cast<pi_native_handle>(&Queue->Queue);
  return PI_SUCCESS;
}

/// Creates PI queue object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI queue from.
/// \param queue is the PI queue created from the native handle.
NOT_IMPL(pi_result VLK(piextQueueCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_queue *Queue))

pi_result VLK(piMemBufferCreate)(pi_context context, pi_mem_flags flags,
                                 size_t size, void *host_ptr, pi_mem *ret_mem) {
  // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
  uint32_t MemoryTypeIndex = VK_MAX_MEMORY_TYPES;

  auto MemoryProperties = context->PhDevice_->PhDevice.getMemoryProperties();

  for (uint32_t k = 0; k < MemoryProperties.memoryTypeCount; k++) {
    // if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    // |VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &
    // memoryProperties.memoryTypes[k].propertyFlags &&
    if ((vk::MemoryPropertyFlagBits::eHostVisible |
         vk::MemoryPropertyFlagBits::eHostCoherent) &
            MemoryProperties.memoryTypes[k].propertyFlags &&
        (size <
         MemoryProperties.memoryHeaps[MemoryProperties.memoryTypes[k].heapIndex]
             .size)) {

      MemoryTypeIndex = k;
      break;
    }
  }

  if (MemoryTypeIndex == VK_MAX_MEMORY_TYPES)
    return PI_OUT_OF_HOST_MEMORY; // or PI_OUT_OF_RESOURCES?

  try {
    if (flags & PI_MEM_FLAGS_HOST_PTR_USE) {
      // TODO: Implement Host pointer copy using
      // VkImportMemoryHostPointerInfoEXT and maybe
      // VkExportMemoryWin32HandleInfoNV
      vk::StructureChain<vk::MemoryAllocateInfo,
                         vk::ImportMemoryHostPointerInfoEXT>
          allocInfo = {
              vk::MemoryAllocateInfo(size, MemoryTypeIndex),
              vk::ImportMemoryHostPointerInfoEXT(
                  vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
                  host_ptr)};
      *ret_mem = new _pi_mem{context->Device.allocateMemory(
                                 allocInfo.get<vk::MemoryAllocateInfo>()),
                             context->Device.createBuffer(vk::BufferCreateInfo(
                                 vk::BufferCreateFlags(), size,
                                 vk::BufferUsageFlagBits::eStorageBuffer,
                                 vk::SharingMode::eExclusive)),
                             context};
      // cl::sycl::detail::pi::die("HOST_PTR_USE not implemented");
    } else {

      *ret_mem = new _pi_mem(context->Device.allocateMemory(
                                 vk::MemoryAllocateInfo(size, MemoryTypeIndex)),
                             context->Device.createBuffer(vk::BufferCreateInfo(
                                 vk::BufferCreateFlags(), size,
                                 vk::BufferUsageFlagBits::eStorageBuffer,
                                 vk::SharingMode::eExclusive)),
                             context);
    }
  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  return PI_SUCCESS;
}

// TODO: Implement
pi_result VLK(piMemImageCreate)(pi_context context, pi_mem_flags flags,
                                const pi_image_format *image_format,
                                const pi_image_desc *image_desc, void *host_ptr,
                                pi_mem *ret_mem) {
  cl::sycl::detail::pi::die(
      "VLK(piextGetDeviceFunctionPointer) not implemented");
  return {};
}

pi_result VLK(piMemGetInfo)(pi_mem mem,
                            cl_mem_info param_name, // TODO: untie from OpenCL
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("VLK(piMemGetInfo) not implemented");
  return {};
}
pi_result VLK(piMemImageGetInfo)(pi_mem image, pi_image_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("VLK(piMemImageGetInfo) not implemented");
  return {};
}

pi_result VLK(piMemRetain)(pi_mem mem) {
  mem->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piMemRelease)(pi_mem mem) {
  mem->RefCounter_--;
  if (mem->RefCounter_ < 1) {
    mem->Context_->Device.freeMemory(mem->Memory);
    free(mem);
  }
  return PI_SUCCESS;
}

pi_result VLK(piMemBufferPartition)(pi_mem Buffer, pi_mem_flags flags,
                                    pi_buffer_create_type buffer_create_type,
                                    void *buffer_create_info, pi_mem *ret_mem) {
  // assert((buffer != nullptr) && "PI_INVALID_MEM_OBJECT");
  //// Default value for flags means PI_MEM_FLAGS_ACCCESS_RW.
  // if (flags == 0) {
  //  flags = PI_MEM_FLAGS_ACCESS_RW;
  //}

  // assert((flags == PI_MEM_FLAGS_ACCESS_RW) && "PI_INVALID_VALUE");
  // assert((buffer_create_type == PI_BUFFER_CREATE_TYPE_REGION) &&
  //       "PI_INVALID_VALUE");
  // assert((buffer_create_info != nullptr) && "PI_INVALID_VALUE");
  // assert(ret_mem != nullptr);

  // const auto bufferRegion =
  //    *reinterpret_cast<const cl_buffer_region *>(buffer_create_info);
  // assert((bufferRegion.size != 0u) && "PI_INVALID_BUFFER_SIZE");

  // assert((bufferRegion.origin <= (bufferRegion.origin + bufferRegion.size))
  // &&
  //       "Overflow");
  ////assert(((bufferRegion.origin + bufferRegion.size) <= buffer->get_size())
  ///&& /       "PI_INVALID_BUFFER_SIZE"); / Retained indirectly due to
  /// retaining parent buffer below.
  // pi_context context = buffer->context_;

  //_pi_mem::native_type ptr = buffer->ptr_ + bufferRegion.origin;

  // void *hostPtr = nullptr;
  // if (buffer->hostPtr_) {
  //  hostPtr = static_cast<char *>(buffer->hostPtr_) + bufferRegion.origin;
  //}

  // ReleaseGuard<pi_mem> releaseGuard(buffer);

  // std::unique_ptr<_pi_mem> retMemObj{nullptr};
  // try {
  //  ScopedContext active(context);

  //  retMemObj = std::unique_ptr<_pi_mem>{new _pi_mem{
  //      context, parent_buffer, allocMode, ptr, hostPtr, bufferRegion.size}};
  //} catch (pi_result err) {
  //  *ret_mem = nullptr;
  //  return err;
  //} catch (...) {
  //  *ret_mem = nullptr;
  //  return PI_OUT_OF_HOST_MEMORY;
  //}

  // releaseGuard.dismiss();
  //*ret_mem = retMemObj.release();
  // return PI_SUCCESS;
  cl::sycl::detail::pi::die("VLK(piMemBufferPartition) not implemented");
  return {};
}

/// Gets the native handle of a PI mem object.
///
/// \param mem is the PI mem to get the native handle of.
/// \param nativeHandle is the native handle of mem.
pi_result VLK(piextMemGetNativeHandle)(pi_mem mem,
                                       pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle =
      reinterpret_cast<pi_native_handle>(&mem->Memory); // Or return Buffer?
  return PI_SUCCESS;
}

/// Creates PI mem object from a native handle.
/// NOTE: The created PI object takes ownership of the native handle.
///
/// \param nativeHandle is the native handle to create PI mem from.
/// \param mem is the PI mem created from the native handle.
NOT_IMPL(pi_result VLK(piextMemCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_mem *mem))

pi_result VLK(piProgramCreate)(pi_context context, const void *il,
                               size_t length, pi_program *ret_program) {
  // size_t deviceCount;

  // cl_int ret_err = clGetContextInfo(
  //    cast<cl_context>(context), CL_CONTEXT_DEVICES, 0, nullptr,
  //    &deviceCount);

  // std::vector<cl_device_id> devicesInCtx(deviceCount);

  // if (ret_err != CL_SUCCESS || deviceCount < 1) {
  //  if (res_program != nullptr)
  //    *res_program = nullptr;
  //  return cast<pi_result>(CL_INVALID_CONTEXT);
  //}

  // ret_err = clGetContextInfo(cast<cl_context>(context), CL_CONTEXT_DEVICES,
  //                           deviceCount * sizeof(cl_device_id),
  //                           devicesInCtx.data(), nullptr);

  // CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  // cl_platform_id curPlatform;
  // ret_err = clGetDeviceInfo(devicesInCtx[0], CL_DEVICE_PLATFORM,
  //                          sizeof(cl_platform_id), &curPlatform, nullptr);

  // CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  // size_t devVerSize;
  // ret_err = clGetPlatformInfo(curPlatform, CL_PLATFORM_VERSION, 0, nullptr,
  //                            &devVerSize);
  // std::string devVer(devVerSize, '\0');
  // ret_err = clGetPlatformInfo(curPlatform, CL_PLATFORM_VERSION, devVerSize,
  //                            &devVer.front(), nullptr);

  // CHECK_ERR_SET_NULL_RET(ret_err, res_program, CL_INVALID_CONTEXT);

  // pi_result err = PI_SUCCESS;
  // if (devVer.find("OpenCL 1.0") == std::string::npos &&
  //    devVer.find("OpenCL 1.1") == std::string::npos &&
  //    devVer.find("OpenCL 1.2") == std::string::npos &&
  //    devVer.find("OpenCL 2.0") == std::string::npos) {
  //  if (res_program != nullptr)
  //    *res_program = cast<pi_program>(clCreateProgramWithIL(
  //        cast<cl_context>(context), il, length, cast<cl_int *>(&err)));
  //  return err;
  //}

  // devide length here due to reinterprete?
  auto Program = std::make_unique<_pi_program>(
      context->Device.createShaderModule(
          vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), length,
                                     reinterpret_cast<const uint32_t *>(il))),
      context, reinterpret_cast<const char *>(il), length);
  *ret_program = Program.release();
  return PI_SUCCESS;
}

// TODO: Implement
pi_result VLK(piclProgramCreateWithSource)(pi_context context, pi_uint32 count,
                                           const char **strings,
                                           const size_t *lengths,
                                           pi_program *ret_program) {
  cl::sycl::detail::pi::die("VLK(piextProgramConvert not implemented");
  return {};
}

pi_result VLK(piclProgramCreateWithBinary)(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    pi_int32 *binary_status, pi_program *ret_program) {
  // TODO: Only one device for now
  // TODO: is binary_status needed to be filled
  // devide length here due to reinterprete?
  auto Program = std::make_unique<_pi_program>(
      context->Device.createShaderModule(vk::ShaderModuleCreateInfo(
          vk::ShaderModuleCreateFlags(), lengths[0],
          reinterpret_cast<const uint32_t *>(binaries[0]))),
      context, reinterpret_cast<const char *>(binaries[0]), lengths[0]);
  *ret_program = Program.release();
  return PI_SUCCESS;
}

pi_result VLK(piProgramGetInfo)(pi_program program, pi_program_info param_name,
                                size_t param_value_size, void *param_value,
                                size_t *param_value_size_ret) {
  assert(program != nullptr);

  // Program->context_->device.getShaderInfoAMD();

  switch (param_name) {
  case PI_PROGRAM_INFO_REFERENCE_COUNT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->RefCounter_);
  case PI_PROGRAM_INFO_CONTEXT:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->Context_);
  case PI_PROGRAM_INFO_NUM_DEVICES:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1u);
  case PI_PROGRAM_INFO_DEVICES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->Context_->PhDevice_);
  case PI_PROGRAM_INFO_SOURCE:
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   program->Source_);
  case PI_PROGRAM_INFO_BINARY_SIZES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->SourceLength_);
  case PI_PROGRAM_INFO_BINARIES:
    return getInfoArray(1, param_value_size, param_value, param_value_size_ret,
                        &program->Source_);
  case PI_PROGRAM_INFO_NUM_KERNELS:
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  case PI_PROGRAM_INFO_KERNEL_NAMES:
    // TODO: How to get this in Vulkan API?
    cl::sycl::detail::pi::die(
        "KERNEL_NAMES Program info request not implemented");
    return {};

  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result VLK(piProgramLink)(pi_context context, pi_uint32 num_devices,
                             const pi_device *device_list, const char *options,
                             pi_uint32 num_input_programs,
                             const pi_program *input_programs,
                             void (*pfn_notify)(pi_program program,
                                                void *user_data),
                             void *user_data, pi_program *ret_program) {
  cl::sycl::detail::pi::die("VLK(piProgramLink) not implemented");
  return {};
}

pi_result VLK(piProgramCompile)(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  cl::sycl::detail::pi::die("VLK(piProgramCompile) not implemented");
  return {};
}

pi_result VLK(piProgramBuild)(pi_program program, pi_uint32 num_devices,
                              const pi_device *device_list, const char *options,
                              void (*pfn_notify)(pi_program program,
                                                 void *user_data),
                              void *user_data) {
  cl::sycl::detail::pi::die("VLK(piProgramBuild) not implemented");
  return {};
}

pi_result VLK(piProgramGetBuildInfo)(pi_program program, pi_device Device,
                                     cl_program_build_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) {
  assert(program != nullptr);

  switch (param_name) {
  case PI_PROGRAM_BUILD_INFO_STATUS: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_PROGRAM_BUILD_STATUS_NONE);
  }
  case PI_PROGRAM_BUILD_INFO_OPTIONS:
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  case PI_PROGRAM_BUILD_INFO_LOG:
    // TODO: Implement
  default:
    PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  cl::sycl::detail::pi::die("Program Build info request not implemented");
  return {};
}

pi_result VLK(piProgramRetain)(pi_program program) {
  program->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piProgramRelease)(pi_program program) {
  program->RefCounter_--;
  if (program->RefCounter_ < 1)
    program->Context_->Device.destroyShaderModule(program->Module);
  free(program);
  return PI_SUCCESS;
}

// TODO: Implement
pi_result VLK(piextProgramSetSpecializationConstant)(pi_program prog,
                                                     pi_uint32 spec_id,
                                                     size_t spec_size,
                                                     const void *spec_value) {
  cl::sycl::detail::pi::die(
      "VLK(piextProgramSetSpecializationConstant) not implemented");
  return {};
}

/// Gets the native handle of a PI program object.
///
/// \param program is the PI program to get the native handle of.
/// \param nativeHandle is the native handle of program.
pi_result VLK(piextProgramGetNativeHandle)(pi_program program,
                                           pi_native_handle *nativeHandle) {
  assert(nativeHandle != nullptr);
  *nativeHandle = reinterpret_cast<pi_native_handle>(&program->Module);
  return PI_SUCCESS;
}

/// \TODO Not implemented
pi_result VLK(piextProgramCreateWithNativeHandle)(pi_native_handle nativeHandle,
                                                  pi_program *program) {
  cl::sycl::detail::pi::die(
      "VLK(piextProgramCreateWithNativeHandle) not implemented");
  return {};
}

pi_result VLK(piKernelCreate)(pi_program program, const char *kernel_name,
                              pi_kernel *ret_kernel) {

  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piKernelSetArg),
         (pi_kernel kernel, pi_uint32 arg_index, size_t arg_size,
          const void *arg_value))

pi_result VLK(piKernelGetInfo)(pi_kernel kernel, pi_kernel_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_INFO_FUNCTION_NAME:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->Name);
    case PI_KERNEL_INFO_NUM_ARGS:
      return getInfo(param_value_size, param_value, param_value_size_ret, 0);
    case PI_KERNEL_INFO_REFERENCE_COUNT:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->RefCounter_);
    case PI_KERNEL_INFO_CONTEXT: {
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->Program_->Context_);
    }
    case PI_KERNEL_INFO_PROGRAM: {
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->Program_);
    }
    case PI_KERNEL_INFO_ATTRIBUTES: {
      return getInfo(param_value_size, param_value, param_value_size_ret, "");
    }
    default: { PI_HANDLE_UNKNOWN_PARAM_NAME(param_name); }
    }
  }

  return PI_INVALID_KERNEL;
}

pi_result VLK(piKernelGetGroupInfo)(pi_kernel kernel, pi_device Device,
                                    pi_kernel_group_info param_name,
                                    size_t param_value_size, void *param_value,
                                    size_t *param_value_size_ret) {

  // how to find detailed information for kernels?
  // needs the SPIRV to be parsed?

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
      /*int max_threads = 0;
      cl::sycl::detail::pi::assertion(
          cuFuncGetAttribute(&max_threads,
                             CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             kernel->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     size_t(max_threads));*/
    }
    case PI_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
      // Returns the work-group size specified in the kernel source or IL.
      // If the work-group size is not specified in the kernel source or IL,
      // (0, 0, 0) is returned.
      // https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clGetKernelWorkGroupInfo.html

      // TODO: can we extract the work group size from the SPIRV?
      /*size_t group_size[3] = {0, 0, 0};
      return getInfoArray(3, param_value_size, param_value,
                          param_value_size_ret, group_size);*/
    }
    case PI_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
      // OpenCL LOCAL == CUDA SHARED
      /*int bytes = 0;
      cl::sycl::detail::pi::assertion(
          cuFuncGetAttribute(&bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                             kernel->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));*/
    }
    case PI_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
      // Work groups should be multiples of the warp size
      /*int warpSize = 0;
      cl::sycl::detail::pi::assertion(
          cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                               device->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     static_cast<size_t>(warpSize));*/
    }
    case PI_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
      // OpenCL PRIVATE == CUDA LOCAL
      /*int bytes = 0;
      cl::sycl::detail::pi::assertion(
          cuFuncGetAttribute(&bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                             kernel->get()) == CUDA_SUCCESS);
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     pi_uint64(bytes));*/
    }
    default:
      PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    }
  }

  return PI_INVALID_KERNEL;
}

// \TODO: Not implemented
pi_result VLK(piKernelGetSubGroupInfo)(
    pi_kernel kernel, pi_device Device,
    cl_kernel_sub_group_info param_name, // TODO: untie from OpenCL
    size_t input_value_size, const void *input_value, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) {
  cl::sycl::detail::pi::die("VLK(piKernelGetSubGroupInfo) not implemented");
  return {};
}

pi_result VLK(piKernelRetain)(pi_kernel kernel) {
  kernel->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piKernelRelease)(pi_kernel kernel) {
  kernel->RefCounter_--;
  if (kernel->RefCounter_ < 1) {
    free(kernel);
  }
  return PI_SUCCESS;
}

pi_result VLK(piextKernelSetArgPointer)(pi_kernel kernel, pi_uint32 arg_index,
                                        size_t arg_size,
                                        const void *arg_value) {
  // TODO: Implement this
  return {};
}

/// API to set attributes controlling kernel execution
///
/// \param kernel is the pi kernel to execute
/// \param param_name is a pi_kernel_exec_info value that specifies the info
///        passed to the kernel
/// \param param_value_size is the size of the value in bytes
/// \param param_value is a pointer to the value to set for the kernel
///
/// If param_name is PI_USM_INDIRECT_ACCESS, the value will be a ptr to
///    the pi_bool value PI_TRUE
/// If param_name is PI_USM_PTRS, the value will be an array of ptrs
pi_result VLK(piKernelSetExecInfo)(pi_kernel kernel,
                                   pi_kernel_exec_info param_name,
                                   size_t param_value_size,
                                   const void *param_value) {
  if (param_name != PI_USM_PTRS) {
    return PI_ERROR_UNKNOWN;
  }

  //?? what to do here?

  for (size_t i = 0; i < param_value_size; i++) { // param_value[i]
  }

  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEventCreate),
         (pi_context context, pi_event *ret_event))
NOT_IMPL(pi_result VLK(piEventGetInfo),
         (pi_event event,
          cl_event_info param_name, // TODO: untie from OpenCL
          size_t param_value_size, void *param_value,
          size_t *param_value_size_ret))
NOT_IMPL(pi_result VLK(piEventGetProfilingInfo),
         (pi_event event, pi_profiling_info param_name, size_t param_value_size,
          void *param_value, size_t *param_value_size_ret))
NOT_IMPL(pi_result VLK(piEventsWait),
         (pi_uint32 num_events, const pi_event *event_list))
NOT_IMPL(pi_result VLK(piEventSetCallback),
         (pi_event event, pi_int32 command_exec_callback_type,
          void (*pfn_notify)(pi_event event, pi_int32 event_command_status,
                             void *user_data),
          void *user_data))
NOT_IMPL(pi_result VLK(piEventSetStatus),
         (pi_event event, pi_int32 execution_status))
NOT_IMPL(pi_result VLK(piEventRetain), (pi_event event))
NOT_IMPL(pi_result VLK(piEventRelease), (pi_event event))
NOT_IMPL(pi_result VLK(piextEventGetNativeHandle),
         (pi_event event, pi_native_handle *nativeHandle))
NOT_IMPL(pi_result VLK(piextEventCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_event *event))
NOT_IMPL(pi_result VLK(piSamplerCreate),
         (pi_context context, const pi_sampler_properties *sampler_properties,
          pi_sampler *result_sampler))
NOT_IMPL(pi_result VLK(piSamplerGetInfo),
         (pi_sampler sampler, pi_sampler_info param_name,
          size_t param_value_size, void *param_value,
          size_t *param_value_size_ret))
NOT_IMPL(pi_result VLK(piSamplerRetain), (pi_sampler sampler))
NOT_IMPL(pi_result VLK(piSamplerRelease), (pi_sampler sampler))

pi_result VLK(piEnqueueKernelLaunch)(
    pi_queue Queue, pi_kernel kernel, pi_uint32 work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  auto &Device = Queue->Context_->Device;

  try {
    vk::UniqueDescriptorSetLayout DescriptorSetLayout =
        Device.createDescriptorSetLayoutUnique(
            vk::DescriptorSetLayoutCreateInfo(
                vk::DescriptorSetLayoutCreateFlags(),
                static_cast<uint32_t>(
                    kernel->DescriptorSetLayoutBinding.size()),
                kernel->DescriptorSetLayoutBinding.data()));

    // create a PipelineLayout using that DescriptorSetLayout
    vk::UniquePipelineLayout PipelineLayout =
        Device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(
            vk::PipelineLayoutCreateFlags(), 1, &DescriptorSetLayout.get()));

    vk::ComputePipelineCreateInfo computePipelineInfo(
        vk::PipelineCreateFlags(),
        vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                          vk::ShaderStageFlagBits::eCompute,
                                          kernel->Program_->Module,
                                          kernel->Name),
        PipelineLayout.get());

    auto Pipeline =
        Device.createComputePipelineUnique(nullptr, computePipelineInfo);

    auto DescriptorPoolSize =
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
    auto DescriptorPool =
        Device.createDescriptorPool(vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlags(), 1, 1, &DescriptorPoolSize));

    auto DescriptorSet = std::move(
        Device
            .allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                DescriptorPool, 1, &DescriptorSetLayout.get()))
            .front());

    std::vector<vk::WriteDescriptorSet> WriteSets{kernel->Arguments.size()};
    std::vector<std::unique_ptr<vk::DescriptorBufferInfo>>
        DescriptorBufferInfos{kernel->Arguments.size()};

    for (uint32_t i = 0; i < kernel->Arguments.size(); i++) {
      DescriptorBufferInfos[i] = std::make_unique<vk::DescriptorBufferInfo>(
          kernel->Arguments[i]->Buffer, 0, VK_WHOLE_SIZE);
      WriteSets[i] = vk::WriteDescriptorSet{DescriptorSet.get(),
                                            i,
                                            0,
                                            1,
                                            vk::DescriptorType::eStorageBuffer,
                                            nullptr,
                                            DescriptorBufferInfos[i].get(),
                                            nullptr};
    }

    Device.updateDescriptorSets(WriteSets, nullptr);

    auto &CommandBuffer = Queue->CmdBuffer;

    CommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));

    CommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, Pipeline.get());

    CommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                     PipelineLayout.get(), 0, 1,
                                     &DescriptorSet.get(), 0, nullptr);

    CommandBuffer.dispatch(work_dim >= 1 ? global_work_size[0] : 1,
                           work_dim >= 2 ? global_work_size[1] : 1,
                           work_dim >= 3 ? global_work_size[2] : 1);

    CommandBuffer.end();

    vk::SubmitInfo SubmitInfo(0, nullptr, nullptr, 1, &CommandBuffer, 0,
                              nullptr);

    auto VulkanQueue =
        Device.getQueue(Queue->Context_->ComputeQueueFamilyIndex, 0);
    VulkanQueue.submit(1, &SubmitInfo, vk::Fence());
  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEnqueueNativeKernel),
         (pi_queue Queue, void (*user_func)(void *), void *args, size_t cb_args,
          pi_uint32 num_mem_objects, const pi_mem *mem_list,
          const void **args_mem_loc, pi_uint32 num_events_in_wait_list,
          const pi_event *event_wait_list, pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueEventsWait),
         (pi_queue command_queue, pi_uint32 num_events_in_wait_list,
          const pi_event *event_wait_list, pi_event *event))

pi_result VLK(piEnqueueMemBufferRead)(pi_queue command_queue, pi_mem memobj,
                                      pi_bool blocking_read, size_t offset,
                                      size_t size, void *ptr,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  try {

    void *BufferPtr = memobj->Context_->Device.mapMemory(
        memobj->Memory, offset, size, vk::MemoryMapFlags());

    if (std::memcpy(ptr, BufferPtr, size) != ptr) {
      return PI_INVALID_MEM_OBJECT;
    }

  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  memobj->Context_->Device.unmapMemory(memobj->Memory);
  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEnqueueMemBufferReadRect),
         (pi_queue command_queue, pi_mem Buffer, pi_bool blocking_read,
          const size_t *buffer_offset, const size_t *host_offset,
          const size_t *region, size_t buffer_row_pitch,
          size_t buffer_slice_pitch, size_t host_row_pitch,
          size_t host_slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
          const pi_event *event_wait_list, pi_event *event))

pi_result VLK(piEnqueueMemBufferWrite)(pi_queue command_queue, pi_mem memobj,
                                       pi_bool blocking_write, size_t offset,
                                       size_t size, const void *ptr,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event) {

  command_queue->CmdBuffer.updateBuffer(memobj->Buffer, offset, size, ptr);
  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEnqueueMemBufferWriteRect),
         (pi_queue command_queue, pi_mem Buffer, pi_bool blocking_write,
          const size_t *buffer_offset, const size_t *host_offset,
          const size_t *region, size_t buffer_row_pitch,
          size_t buffer_slice_pitch, size_t host_row_pitch,
          size_t host_slice_pitch, const void *ptr,
          pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
          pi_event *event))

pi_result VLK(piEnqueueMemBufferCopy)(pi_queue command_queue, pi_mem src_buffer,
                                      pi_mem dst_buffer, size_t src_offset,
                                      size_t dst_offset, size_t size,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {

  std::array<vk::BufferCopy, 1> Range = {
      vk::BufferCopy{src_offset, dst_offset, size}};

  command_queue->CmdBuffer.copyBuffer(src_buffer->Buffer, dst_buffer->Buffer,
                                      Range);
  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEnqueueMemBufferCopyRect),
         (pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
          const size_t *src_origin, const size_t *dst_origin,
          const size_t *region, size_t src_row_pitch, size_t src_slice_pitch,
          size_t dst_row_pitch, size_t dst_slice_pitch,
          pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
          pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueMemBufferFill),
         (pi_queue command_queue, pi_mem Buffer, const void *pattern,
          size_t pattern_size, size_t offset, size_t size,
          pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
          pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueMemImageRead),
         (pi_queue command_queue, pi_mem image, pi_bool blocking_read,
          const size_t *origin, const size_t *region, size_t row_pitch,
          size_t slice_pitch, void *ptr, pi_uint32 num_events_in_wait_list,
          const pi_event *event_wait_list, pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueMemImageWrite),
         (pi_queue command_queue, pi_mem image, pi_bool blocking_write,
          const size_t *origin, const size_t *region, size_t input_row_pitch,
          size_t input_slice_pitch, const void *ptr,
          pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
          pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueMemImageCopy),
         (pi_queue command_queue, pi_mem src_image, pi_mem dst_image,
          const size_t *src_origin, const size_t *dst_origin,
          const size_t *region, pi_uint32 num_events_in_wait_list,
          const pi_event *event_wait_list, pi_event *event))

NOT_IMPL(pi_result VLK(piEnqueueMemImageFill),
         (pi_queue command_queue, pi_mem image, const void *fill_color,
          const size_t *origin, const size_t *region,
          pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
          pi_event *event))

pi_result VLK(piEnqueueMemBufferMap)(
    pi_queue command_queue, pi_mem memobj, pi_bool blocking_map,
    cl_map_flags map_flags, // TODO: untie from OpenCL
    size_t offset, size_t size, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event, void **ret_map) {

  // cl_map_flags map_flags  for read/write is not (yet?) supported in VULKAN
  vk::Result Error = memobj->Context_->Device.mapMemory(
      memobj->Memory, offset, size, vk::MemoryMapFlags(), ret_map);
  return mapVulkanErrToCLErr(Error);
}
pi_result VLK(piEnqueueMemUnmap)(pi_queue command_queue, pi_mem memobj,
                                 void *mapped_ptr,
                                 pi_uint32 num_events_in_wait_list,
                                 const pi_event *event_wait_list,
                                 pi_event *event) {
  memobj->Context_->Device.unmapMemory(memobj->Memory);
  return PI_SUCCESS;
}

pi_result VLK(piextKernelSetArgMemObj)(pi_kernel kernel, pi_uint32 arg_index,
                                       const pi_mem *arg_value) {
  return kernel->addArgument(arg_index, *arg_value);
}

//
// USM
//

NOT_IMPL(pi_result VLK(piextUSMHostAlloc),
         (void **result_ptr, pi_context context,
          pi_usm_mem_properties *properties, size_t size, pi_uint32 alignment))
NOT_IMPL(pi_result VLK(piextUSMDeviceAlloc),
         (void **result_ptr, pi_context context, pi_device Device,
          pi_usm_mem_properties *properties, size_t size, pi_uint32 alignment))
NOT_IMPL(pi_result VLK(piextUSMSharedAlloc),
         (void **result_ptr, pi_context context, pi_device Device,
          pi_usm_mem_properties *properties, size_t size, pi_uint32 alignment))
NOT_IMPL(pi_result VLK(piextUSMFree), (pi_context context, void *ptr))
NOT_IMPL(pi_result VLK(piextUSMEnqueueMemset),
         (pi_queue Queue, void *ptr, pi_int32 value, size_t count,
          pi_uint32 num_events_in_waitlist, const pi_event *events_waitlist,
          pi_event *event))
NOT_IMPL(pi_result VLK(piextUSMEnqueueMemcpy),
         (pi_queue Queue, pi_bool blocking, void *dst_ptr, const void *src_ptr,
          size_t size, pi_uint32 num_events_in_waitlist,
          const pi_event *events_waitlist, pi_event *event))
NOT_IMPL(pi_result VLK(piextUSMEnqueuePrefetch),
         (pi_queue Queue, const void *ptr, size_t size,
          pi_usm_migration_flags flags, pi_uint32 num_events_in_waitlist,
          const pi_event *events_waitlist, pi_event *event))
NOT_IMPL(pi_result VLK(piextUSMEnqueueMemAdvise),
         (pi_queue Queue, const void *ptr, size_t length, int advice,
          pi_event *event))

NOT_IMPL(pi_result VLK(piextUSMGetMemAllocInfo),
         (pi_context context, const void *ptr, pi_mem_info param_name,
          size_t param_value_size, void *param_value,
          size_t *param_value_size_ret))

pi_result piPluginInit(pi_plugin *PluginInit) {
  int CompareVersions = strcmp(PluginInit->PiVersion, SupportedVersion);
  if (CompareVersions < 0) {
    // PI interface supports lower version of PI.
    // TODO: Take appropriate actions.
    return PI_INVALID_OPERATION;
  }

  // PI interface supports higher version or the same version.
  strncpy(PluginInit->PluginVersion, SupportedVersion, 4);

#define _PI_CL(pi_api, ocl_api)                                                \
  (PluginInit->PiFunctionTable).pi_api = (decltype(&::pi_api))(&ocl_api);

  // Platform
  _PI_CL(piPlatformsGet, VLK(piPlatformsGet))
  _PI_CL(piPlatformGetInfo, VLK(piPlatformGetInfo))
  // Device
  _PI_CL(piDevicesGet, VLK(piDevicesGet))
  _PI_CL(piDeviceGetInfo, VLK(piDeviceGetInfo))
  _PI_CL(piDevicePartition, VLK(piDevicePartition))
  _PI_CL(piDeviceRetain, VLK(piDeviceRetain))
  _PI_CL(piDeviceRelease, VLK(piDeviceRelease))
  _PI_CL(piextDeviceGetNativeHandle, VLK(piextDeviceGetNativeHandle))
  _PI_CL(piextDeviceCreateWithNativeHandle,
         VLK(piextDeviceCreateWithNativeHandle))
  _PI_CL(piextDeviceSelectBinary, VLK(piextDeviceSelectBinary))
  _PI_CL(piextGetDeviceFunctionPointer, VLK(piextGetDeviceFunctionPointer))
  // Context
  _PI_CL(piContextCreate, VLK(piContextCreate))
  _PI_CL(piContextGetInfo, VLK(piContextGetInfo))
  _PI_CL(piContextRetain, VLK(piContextRetain))
  _PI_CL(piContextRelease, VLK(piContextRelease))
  _PI_CL(piextContextGetNativeHandle, VLK(piextContextGetNativeHandle))
  _PI_CL(piextContextCreateWithNativeHandle,
         VLK(piextContextCreateWithNativeHandle))
  // Queue
  _PI_CL(piQueueCreate, VLK(piQueueCreate))
  _PI_CL(piQueueGetInfo, VLK(piQueueGetInfo))
  _PI_CL(piQueueFinish, VLK(piQueueFinish))
  _PI_CL(piQueueRetain, VLK(piQueueRetain))
  _PI_CL(piQueueRelease, VLK(piQueueRelease))
  _PI_CL(piextQueueGetNativeHandle, VLK(piextQueueGetNativeHandle))
  _PI_CL(piextQueueCreateWithNativeHandle,
         VLK(piextQueueCreateWithNativeHandle))
  // Memory
  _PI_CL(piMemBufferCreate, VLK(piMemBufferCreate))
  _PI_CL(piMemImageCreate, VLK(piMemImageCreate))
  _PI_CL(piMemGetInfo, VLK(piMemGetInfo))
  _PI_CL(piMemImageGetInfo, VLK(piMemImageGetInfo))
  _PI_CL(piMemRetain, VLK(piMemRetain))
  _PI_CL(piMemRelease, VLK(piMemRelease))
  _PI_CL(piMemBufferPartition, VLK(piMemBufferPartition))
  _PI_CL(piextMemGetNativeHandle, VLK(piextMemGetNativeHandle))
  _PI_CL(piextMemCreateWithNativeHandle, VLK(piextMemCreateWithNativeHandle))
  // Program
  _PI_CL(piProgramCreate, VLK(piProgramCreate))
  _PI_CL(piclProgramCreateWithSource, VLK(piclProgramCreateWithSource))
  _PI_CL(piclProgramCreateWithBinary, VLK(piclProgramCreateWithBinary))
  _PI_CL(piProgramGetInfo, VLK(piProgramGetInfo))
  _PI_CL(piProgramCompile, VLK(piProgramCompile))
  _PI_CL(piProgramBuild, VLK(piProgramBuild))
  _PI_CL(piProgramLink, VLK(piProgramLink))
  _PI_CL(piProgramGetBuildInfo, VLK(piProgramGetBuildInfo))
  _PI_CL(piProgramRetain, VLK(piProgramRetain))
  _PI_CL(piProgramRelease, VLK(piProgramRelease))
  _PI_CL(piextProgramSetSpecializationConstant,
         VLK(piextProgramSetSpecializationConstant))
  _PI_CL(piextProgramGetNativeHandle, VLK(piextProgramGetNativeHandle))
  _PI_CL(piextProgramCreateWithNativeHandle,
         VLK(piextProgramCreateWithNativeHandle))
  // Kernel
  _PI_CL(piKernelCreate, VLK(piKernelCreate))
  _PI_CL(piKernelSetArg, VLK(piKernelSetArg))
  _PI_CL(piKernelGetInfo, VLK(piKernelGetInfo))
  _PI_CL(piKernelGetGroupInfo, VLK(piKernelGetGroupInfo))
  _PI_CL(piKernelGetSubGroupInfo, VLK(piKernelGetSubGroupInfo))
  _PI_CL(piKernelRetain, VLK(piKernelRetain))
  _PI_CL(piKernelRelease, VLK(piKernelRelease))
  _PI_CL(piKernelSetExecInfo, VLK(piKernelSetExecInfo))
  _PI_CL(piextKernelSetArgPointer, VLK(piextKernelSetArgPointer))
  _PI_CL(piextKernelSetArgMemObj, VLK(piextKernelSetArgMemObj))
  // Event
  _PI_CL(piEventCreate, VLK(piEventCreate))
  _PI_CL(piEventGetInfo, VLK(piEventGetInfo))
  _PI_CL(piEventGetProfilingInfo, VLK(piEventGetProfilingInfo))
  _PI_CL(piEventsWait, VLK(piEventsWait))
  _PI_CL(piEventSetCallback, VLK(piEventSetCallback))
  _PI_CL(piEventSetStatus, VLK(piEventSetStatus))
  _PI_CL(piEventRetain, VLK(piEventRetain))
  _PI_CL(piEventRelease, VLK(piEventRelease))
  _PI_CL(piextEventGetNativeHandle, VLK(piextEventGetNativeHandle))
  _PI_CL(piextEventCreateWithNativeHandle,
         VLK(piextEventCreateWithNativeHandle))
  // Sampler
  _PI_CL(piSamplerCreate, VLK(piSamplerCreate))
  _PI_CL(piSamplerGetInfo, VLK(piSamplerGetInfo))
  _PI_CL(piSamplerRetain, VLK(piSamplerRetain))
  _PI_CL(piSamplerRelease, VLK(piSamplerRelease))
  // Queue commands
  _PI_CL(piEnqueueKernelLaunch, VLK(piEnqueueKernelLaunch))
  _PI_CL(piEnqueueNativeKernel, VLK(piEnqueueNativeKernel))
  _PI_CL(piEnqueueEventsWait, VLK(piEnqueueEventsWait))
  _PI_CL(piEnqueueMemBufferRead, VLK(piEnqueueMemBufferRead))
  _PI_CL(piEnqueueMemBufferReadRect, VLK(piEnqueueMemBufferReadRect))
  _PI_CL(piEnqueueMemBufferWrite, VLK(piEnqueueMemBufferWrite))
  _PI_CL(piEnqueueMemBufferWriteRect, VLK(piEnqueueMemBufferWriteRect))
  _PI_CL(piEnqueueMemBufferCopy, VLK(piEnqueueMemBufferCopy))
  _PI_CL(piEnqueueMemBufferCopyRect, VLK(piEnqueueMemBufferCopyRect))
  _PI_CL(piEnqueueMemBufferFill, VLK(piEnqueueMemBufferFill))
  _PI_CL(piEnqueueMemImageRead, VLK(piEnqueueMemImageRead))
  _PI_CL(piEnqueueMemImageWrite, VLK(piEnqueueMemImageWrite))
  _PI_CL(piEnqueueMemImageCopy, VLK(piEnqueueMemImageCopy))
  _PI_CL(piEnqueueMemImageFill, VLK(piEnqueueMemImageFill))
  _PI_CL(piEnqueueMemBufferMap, VLK(piEnqueueMemBufferMap))
  _PI_CL(piEnqueueMemUnmap, VLK(piEnqueueMemUnmap))
  // USM
  _PI_CL(piextUSMHostAlloc, VLK(piextUSMHostAlloc))
  _PI_CL(piextUSMDeviceAlloc, VLK(piextUSMDeviceAlloc))
  _PI_CL(piextUSMSharedAlloc, VLK(piextUSMSharedAlloc))
  _PI_CL(piextUSMFree, VLK(piextUSMFree))
  _PI_CL(piextUSMEnqueueMemset, VLK(piextUSMEnqueueMemset))
  _PI_CL(piextUSMEnqueueMemcpy, VLK(piextUSMEnqueueMemcpy))
  _PI_CL(piextUSMEnqueuePrefetch, VLK(piextUSMEnqueuePrefetch))
  _PI_CL(piextUSMEnqueueMemAdvise, VLK(piextUSMEnqueueMemAdvise))
  _PI_CL(piextUSMGetMemAllocInfo, VLK(piextUSMGetMemAllocInfo))

#undef _PI_CL

  return PI_SUCCESS;
}

} // end extern 'C'
