#pragma once
//===-- pi_vulkan.hpp - VULKAN Plugin -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_vulkan CUDA Plugin
/// \ingroup sycl_pi

/// \file pi_vulkan.hpp
/// Declarations for VULKAN Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying VULKAN runtime.
///
/// \ingroup sycl_pi_vulkan

#include "CL/sycl/detail/pi.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

extern "C" {
// Convenience macro makes source code search easier
#define VLK(pi_api) Vulkan##pi_api
/// \cond INGORE_BLOCK_IN_DOXYGEN
pi_result VLK(piContextRetain)(pi_context);
pi_result VLK(piContextRelease)(pi_context);
pi_result VLK(piDeviceRelease)(pi_device);
pi_result VLK(piDeviceRetain)(pi_device);
pi_result VLK(piProgramRetain)(pi_program);
pi_result VLK(piProgramRelease)(pi_program);
pi_result VLK(piQueueRelease)(pi_queue);
pi_result VLK(piQueueRetain)(pi_queue);
pi_result VLK(piMemRetain)(pi_mem);
pi_result VLK(piMemRelease)(pi_mem);
pi_result VLK(piKernelRetain)(pi_kernel);
pi_result VLK(piKernelRelease)(pi_kernel);
/// \endcond
}

/// A PI platform stores all known PI devices,
///  in the CUDA plugin this is just a vector of
///  available devices since initialization is done
///  when devices are used.
///
struct _pi_platform {
  vk::Instance Instance_;
};

struct _pi_device {
  vk::PhysicalDevice PhDevice;
  pi_platform Platform_;

  _pi_device(vk::PhysicalDevice PhDevice_, pi_platform Platform)
      : PhDevice(PhDevice_), Platform_(Platform) {}
};

struct _ref_counter {
  uint32_t RefCounter_;
};

struct _pi_context : public _ref_counter {
  vk::Device Device;
  uint32_t ComputeQueueFamilyIndex;
  uint32_t TransferQueueFamilyIndex;
  vk::UniqueSemaphore Timeline;
  uint64_t lastTimelineValue = 1;
  pi_device PhDevice_;

  ~_pi_context() {}
};

struct execution {
  vk::UniqueDescriptorSetLayout DescriptorSetLayout;
  vk::UniqueDescriptorPool DescriptorPool;
  vk::UniquePipelineLayout PipelineLayout;
  vk::UniquePipeline Pipeline;
  vk::UniqueDescriptorSet DescriptorSet;

  using uptr = std::unique_ptr<execution>;
};

struct _pi_queue : public _ref_counter {
  vk::Queue Queue;
  vk::UniqueCommandPool CommandPool;
  vk::CommandBuffer CmdBuffer;

  std::vector<execution::uptr> storedExecutions;

  pi_context Context_;
  pi_queue_properties Properties_;

  _pi_queue(vk::Queue &&queue_, vk::UniqueCommandPool &&pool,
            vk::CommandBuffer &&buffer_, pi_context context,
            pi_queue_properties properties)
      : _ref_counter{1}, Queue(queue_), CommandPool(std::move(pool)),
        CmdBuffer(buffer_), Context_(context), Properties_(properties) {
    if (Context_) {
      VLK(piContextRetain)(Context_);
    }
  }

  ~_pi_queue() {
    if (Context_) {
      Context_->Device.freeCommandBuffers(CommandPool.get(), 1u, &CmdBuffer);
      VLK(piContextRelease)(Context_);
    }
  }

  bool isProfilingEnabled() { return Properties_ & PI_QUEUE_PROFILING_ENABLE; }

};

struct _pi_mem : public _ref_counter {
  vk::DeviceMemory HostMemory;
  vk::Buffer HostBuffer;
  vk::DeviceMemory DeviceMemory;
  vk::Buffer DeviceBuffer;
  pi_context Context_;
  pi_mem_flags MemFlags;
  size_t TotalMemorySize;
  void *HostPtr;

  cl_map_flags LastMapFlags = 0ul;

  _pi_mem(pi_context Context, pi_mem_flags MemFlags_, size_t TotalMemorySize_,
          void *HostPtr_)
      : _ref_counter{1}, Context_(Context), MemFlags(MemFlags_),
        TotalMemorySize(TotalMemorySize_),
        HostPtr(MemFlags_ & PI_MEM_FLAGS_HOST_PTR_USE ? HostPtr_ : nullptr) {
    if (Context_) {
      VLK(piContextRetain)(Context_);
      // Allocated Memory must be at least the required size for Buffer
      // This is either the requested memory size or at least the
      // required minimal size of the hardware
    }
  }

  void allocMemory(vk::Buffer &Buffer, uint32_t MemoryTypeIndex,
                   vk::Buffer &BufferTarget, vk::DeviceMemory &MemoryTarget);

  void allocHostMemory(vk::Buffer Buffer_, uint32_t MemoryTypeIndex) {
    allocMemory(Buffer_, MemoryTypeIndex, HostBuffer, HostMemory);
  }
  void allocDeviceMemory(vk::Buffer Buffer_, uint32_t MemoryTypeIndex) {
    allocMemory(Buffer_, MemoryTypeIndex, DeviceBuffer, DeviceMemory);
  }

  void copy(vk::Buffer &from, vk::Buffer &to,
            vk::ArrayProxy<const vk::BufferCopy> regions, bool isBlocking);
  void copy(vk::Buffer &from, vk::Buffer &to, bool isBlocking) {
    std::array<const vk::BufferCopy, 1> totalSize = {
        vk::BufferCopy(0, 0, TotalMemorySize)};
    copy(from, to, totalSize, isBlocking);
  }
  void copyHtoD() { copy(HostBuffer, DeviceBuffer, false); }
  void copyDtoH() { copy(DeviceBuffer, HostBuffer, false); }
  void copyHtoDblocking() { copy(HostBuffer, DeviceBuffer, true); }
  void copyDtoHblocking() { copy(DeviceBuffer, HostBuffer, true); }

  void releaseMemories() {
    Context_->Device.destroyBuffer(HostBuffer);
    Context_->Device.freeMemory(HostMemory);
    Context_->Device.destroyBuffer(DeviceBuffer);
    Context_->Device.freeMemory(DeviceMemory);
  }

  ~_pi_mem() {
    if (Context_)
      releaseMemories();
    VLK(piContextRelease)(Context_);
  }
};

struct _pi_program : public _ref_counter {
  vk::ShaderModule Module;
  pi_context Context_;
  const char *Source_;
  size_t SourceLength_;

  _pi_program(vk::ShaderModule &&Module_, pi_context Context,
              const char *Source, size_t SourceLength)
      : _ref_counter{1}, Module(Module_), Context_(Context), Source_(Source),
        SourceLength_(SourceLength) {
    if (Context_)
      VLK(piContextRetain)(Context_);
  }
  
  ~_pi_program() {
    if (Context_)
      VLK(piContextRelease)(Context_);
  }
};

struct _pi_kernel : public _ref_counter {
  const char *Name;
  std::map<pi_uint32, size_t> ArgIndexToInternalIndex;
  std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding;

  std::map<pi_uint32, pi_mem> Arguments;

  pi_program Program_;
  _pi_kernel(const char *Name_, pi_program Program)
      : _ref_counter{1}, Name(Name_), Program_(Program) {
    if (Program_)
      VLK(piProgramRetain)(Program_);
  }

  ~_pi_kernel() {
    if (Program_)
      VLK(piProgramRelease)(Program_);
    for (auto args : Arguments) {
      VLK(piMemRelease)(args.second);
    }
  }

  size_t getInternalIndex(pi_uint32 ArgIdx);
  pi_uint32 getArgumentIndex(size_t IntIdx) const;

  pi_result addArgument(pi_uint32 ArgIndex, const pi_mem *Memobj);
  pi_result addArgument(pi_uint32 ArgIndex, size_t arg_size,
                        const void *arg_value);
};

struct _pi_event : public _ref_counter {
  _pi_event() : _ref_counter{1} {}
  virtual ~_pi_event(){};

  virtual void wait() = 0;

  virtual pi_result getProfilingInfo(pi_profiling_info param_name,
                                     size_t param_value_size, void *param_value,
                                     size_t *param_value_size_ret) = 0;
};

struct _pi_empty_event : public _pi_event {
  void wait() override { return; }
  pi_result getProfilingInfo(pi_profiling_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) override {
    return PI_PROFILING_INFO_NOT_AVAILABLE;
  }
};

struct _pi_timeline_event : public _pi_event {
  _pi_timeline_event(pi_context context_, vk::Semaphore semaphore_,
                     uint64_t waitValue)
      : Context(context_), Semaphore(semaphore_), Value(waitValue),
        QueryPool(nullptr) {
    if (Context)
      VLK(piContextRetain)(Context);
  }
  _pi_timeline_event(pi_context context_, vk::Semaphore semaphore_,
                     uint64_t waitValue, vk::UniqueQueryPool &pool)
      : Context(context_), Semaphore(semaphore_), Value(waitValue), QueryPool(std::move(pool)) {
    if (Context)
      VLK(piContextRetain)(Context);   
  }

  ~_pi_timeline_event() {
    QueryPool.reset(nullptr);
    if (Context)
      VLK(piContextRelease)(Context);
  }

  pi_context Context;
  vk::Semaphore Semaphore;
  uint64_t Value;
  vk::UniqueQueryPool QueryPool;

  void wait() override {
    vk::DispatchLoaderDynamic dldid(Context->PhDevice_->Platform_->Instance_,
                                    vkGetInstanceProcAddr, Context->Device,
                                    vkGetDeviceProcAddr);
    Context->Device.waitSemaphoresKHR(
        vk::SemaphoreWaitInfoKHR(vk::SemaphoreWaitFlags(), 1, &Semaphore,
                                 &Value),
        UINT64_MAX, dldid);
  }

  pi_result getProfilingInfo(pi_profiling_info param_name,
                             size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) override;
};

template <typename T> struct _pi_event_impl : public _pi_event {

  T &myObj;
  bool Waited = false;
  _pi_event_impl(T &obj) : _pi_event(), myObj(obj) {}
  void wait() override {
    if (!Waited) {
      myObj.waitIdle();
      Waited = true;
    }
  }
};

#undef VLK