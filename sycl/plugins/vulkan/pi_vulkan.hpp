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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
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
pi_result VLK(piEventRelease)(pi_event);
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
  vk::UniqueDevice Device;
  uint32_t ComputeQueueFamilyIndex;
  uint32_t TransferQueueFamilyIndex;
  bool EXTExternalMemoryImportEnabled = false;
  pi_device PhDevice_;
  std::queue<uint32_t> AvailableQueueIndizes;

  ~_pi_context() {}
};

class _mem_ref {
private:
  pi_mem Mem;

public:
  _mem_ref(pi_mem mem) : Mem(mem) { VLK(piMemRetain)(Mem); }
  _mem_ref(_mem_ref &&mv) : Mem(mv.Mem) { mv.Mem = nullptr; };
  _mem_ref(const _mem_ref &cp) : Mem(cp.Mem) { VLK(piMemRetain)(Mem); };
  _mem_ref &operator=(_mem_ref &&mv) {
    Mem = mv.Mem;
    mv.Mem = nullptr;
    return *this;
  }
  _mem_ref &operator=(const _mem_ref &cp) {
    Mem = cp.Mem;
    VLK(piMemRetain)(Mem);
    return *this;
  }
  ~_mem_ref() {
    if (Mem)
      VLK(piMemRelease)(Mem);
  }
};

struct _pi_semaphore {
  _pi_semaphore(vk::UniqueSemaphore &&Sema) : Semaphore(std::move(Sema)){};

  vk::UniqueSemaphore Semaphore;

  static std::shared_ptr<_pi_semaphore> createNew(vk::Device &device);

  typedef std::shared_ptr<_pi_semaphore> sptr_t;
};

struct _pi_execution {

  pi_context Context_;
  std::vector<_mem_ref> MemoryReferences;
  _pi_semaphore::sptr_t Semaphore;
  vk::UniqueFence Fence;
  std::unique_ptr<_pi_queue, void (*)(pi_queue)> Queue;
  vk::UniqueCommandPool CommandPool;
  vk::UniqueCommandBuffer CommandBuffer;
  vk::UniqueDescriptorSetLayout DescriptorSetLayout;
  vk::UniqueDescriptorPool DescriptorPool;
  vk::UniquePipelineLayout PipelineLayout;
  vk::UniquePipeline Pipeline;
  vk::UniqueDescriptorSet DescriptorSet;
  vk::UniqueQueryPool QueryPool;
  std::vector<pi_event> DependendEvents;
  std::vector<_pi_semaphore::sptr_t> DependendSemaphores;

  using uptr_t = std::unique_ptr<_pi_execution>;

  _pi_execution(pi_context Context)
      : Context_(Context), Queue(nullptr, [](auto Value) {
          VLK(piQueueRelease)
          (Value);
        }) {}

  ~_pi_execution() {
    if (!isDone()) {
      wait();
    }
    for (auto Element : DependendEvents) {
      VLK(piEventRelease)(Element);
    }
  }

  bool isDone() const;
  void wait() const;
  void addEventDependency(pi_event event);
  void addAllEventsDependencies(pi_uint32 num_events_in_wait_list,
                                const pi_event *event_wait_list) {
    for (pi_uint32 i = 0; i < num_events_in_wait_list; i++) {
      addEventDependency(event_wait_list[i]);
    }
  }
};

struct _pi_queue : public _ref_counter {
  vk::Queue Queue;
  vk::UniqueCommandPool CommandPool;

  pi_context Context_;
  pi_queue_properties Properties_;
  uint32_t Index;

  std::list<_pi_execution::uptr_t> StoredExecutions;

  _pi_queue(vk::Queue &&queue_, vk::UniqueCommandPool &&pool,
            pi_context context, pi_queue_properties properties,
            uint32_t queueIndex)
      : _ref_counter{1}, Queue(queue_), CommandPool(std::move(pool)),
        Context_(context), Properties_(properties), Index(queueIndex),
        StoredExecutions() {
    if (Context_) {
      VLK(piContextRetain)(Context_);
    }
  }

  ~_pi_queue() {
    if (Context_) {
      // Release Queue Index and set it available again
      Context_->AvailableQueueIndizes.push(Index);
      VLK(piContextRelease)(Context_);
    }
    for (auto &Exec : StoredExecutions) {
      while (Context_->Device->getFenceStatus(Exec->Fence.get()) ==
             vk::Result::eNotReady)
        ;
    }
  }

  bool isProfilingEnabled() { return Properties_ & PI_QUEUE_PROFILING_ENABLE; }
  vk::UniqueCommandBuffer createCommandBuffer();
  void cleanupFinishedExecutions();
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
  _pi_execution::uptr_t InitExecution;
  /// This flag marks memories, which have been written blocking
  /// so that on mapping the memory on host, has to be updated
  /// from the device first.
  bool DeviceDirty = false;

  cl_map_flags LastMapFlags = 0ul;
  bool LastMapBlocking = false;

  _pi_mem(pi_context Context, pi_mem_flags MemFlags_, size_t TotalMemorySize_,
          void *HostPtr_)
      : _ref_counter{1}, Context_(Context), MemFlags(MemFlags_),
        TotalMemorySize(TotalMemorySize_), HostPtr(HostPtr_), InitExecution() {
    if (Context_) {
      VLK(piContextRetain)(Context_);
    }
  }

  void allocMemory(vk::Buffer &Buffer, uint32_t MemoryTypeIndex,
                   vk::Buffer &BufferTarget, vk::DeviceMemory &MemoryTarget,
                   void *host_ptr = nullptr);

  void allocHostMemory(vk::Buffer Buffer_, uint32_t MemoryTypeIndex,
                       void *host_ptr = nullptr) {
    allocMemory(Buffer_, MemoryTypeIndex, HostBuffer, HostMemory, host_ptr);
  }
  void allocDeviceMemory(vk::Buffer Buffer_, uint32_t MemoryTypeIndex) {
    allocMemory(Buffer_, MemoryTypeIndex, DeviceBuffer, DeviceMemory);
  }

  _pi_execution::uptr_t copy(vk::Buffer &from, vk::Buffer &to,
                             vk::ArrayProxy<const vk::BufferCopy> regions,
                             bool isBlocking, pi_uint32 num_events,
                             const pi_event *event_list);
  _pi_execution::uptr_t copy(vk::Buffer &from, vk::Buffer &to, bool isBlocking,
                             pi_uint32 num_events, const pi_event *event_list) {
    std::array<const vk::BufferCopy, 1> totalSize = {
        vk::BufferCopy(0, 0, TotalMemorySize)};
    return copy(from, to, totalSize, isBlocking, num_events, event_list);
  }
  _pi_execution::uptr_t copyHtoD(pi_uint32 num_events,
                                 const pi_event *event_list) {
    return copy(HostBuffer, DeviceBuffer, false, num_events, event_list);
  }
  _pi_execution::uptr_t copyDtoH(pi_uint32 num_events,
                                 const pi_event *event_list) {
    DeviceDirty = false;
    return copy(DeviceBuffer, HostBuffer, false, num_events, event_list);
  }
  void copyHtoDblocking(pi_uint32 num_events, const pi_event *event_list) {
    copy(HostBuffer, DeviceBuffer, true, num_events, event_list);
  }
  void copyDtoHblocking(pi_uint32 num_events, const pi_event *event_list) {
    DeviceDirty = false;
    copy(DeviceBuffer, HostBuffer, true, num_events, event_list);
  }

  void releaseMemories() {
    Context_->Device->destroyBuffer(HostBuffer);
    Context_->Device->freeMemory(HostMemory);
    Context_->Device->destroyBuffer(DeviceBuffer);
    Context_->Device->freeMemory(DeviceMemory);
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
  std::string Name;
  std::map<pi_uint32, size_t> ArgIndexToInternalIndex;
  std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding;
  std::vector<pi_event> AdditionalMemoryEvents;
  vk::UniquePipelineCache PipelineCache;
  pi_event LastLaunch;

  std::map<pi_uint32, pi_mem> Arguments;

  pi_program Program_;
  _pi_kernel(const char *Name_, pi_program Program,
             vk::UniquePipelineCache &&PipelineCache_)
      : _ref_counter{1}, Name(Name_), AdditionalMemoryEvents(),
        PipelineCache(std::move(PipelineCache_)), LastLaunch(nullptr),
        Program_(Program) {
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
  _pi_timeline_event(pi_context context_, _pi_execution::uptr_t &&execution_,
                     uint64_t waitValue)
      : Context(context_), Execution(std::move(execution_)), Value(waitValue) {
    if (Context)
      VLK(piContextRetain)(Context);
  }

  uint64_t getSemaphoreValue() {
    vk::DispatchLoaderDynamic dldid(Context->PhDevice_->Platform_->Instance_,
                                    vkGetInstanceProcAddr, *Context->Device,
                                    vkGetDeviceProcAddr);
    return Context->Device->getSemaphoreCounterValueKHR(
        Execution->Semaphore->Semaphore.get(), dldid);
  }

  void addKernel(pi_kernel kernel) {
    kernel->LastLaunch = this;
    Kernel = kernel;
    VLK(piKernelRetain)(kernel);
  }

  ~_pi_timeline_event() {
    // If event is released but the execution is not finished
    // transfer the execution to the queue object
    if (Execution && !Execution->isDone()) {
      Execution->Queue->StoredExecutions.push_back(std::move(Execution));
      // wait();
    }
    if (Kernel) {
      if (Kernel->LastLaunch == this)
        Kernel->LastLaunch = nullptr;
      VLK(piKernelRelease)(Kernel);
    }
    if (Context)
      VLK(piContextRelease)(Context);
  }

  pi_context Context;
  _pi_execution::uptr_t Execution;
  uint64_t Value;
  pi_kernel Kernel = nullptr;

  void wait() override {
    vk::DispatchLoaderDynamic dldid(Context->PhDevice_->Platform_->Instance_,
                                    vkGetInstanceProcAddr, *Context->Device,
                                    vkGetDeviceProcAddr);
    auto Result = Context->Device->waitSemaphoresKHR(
        vk::SemaphoreWaitInfoKHR(vk::SemaphoreWaitFlags(), 1,
                                 &Execution->Semaphore->Semaphore.get(),
                                 &Value),
        UINT64_MAX, dldid);
    assert(Result == vk::Result::eSuccess &&
           "Semaphore waiting not successful");
  }

  pi_result getProfilingInfo(pi_profiling_info param_name,
                             size_t param_value_size, void *param_value,
                             size_t *param_value_size_ret) override;

  using event_info_t =
      std::tuple<std::vector<vk::Semaphore>, std::vector<uint64_t>,
                 std::vector<vk::PipelineStageFlags>>;

  static event_info_t extractSemaphores(pi_uint32 num_events,
                                        const pi_event *event_list);

  static void setWaitingSemaphores(
      pi_uint32 num_events, const pi_event *event_list,
      vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
          &submitInfo);

private:
  static _pi_timeline_event::event_info_t WaitSemaphores;
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