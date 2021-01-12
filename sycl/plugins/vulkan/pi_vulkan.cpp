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

#include "LLVMSPIRVLib.h"
#include "SPIRVFunction.h"
#include "SPIRVModule.h"
#include "renderdoc_app.h"

#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <thread>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#define CHECK_ERR_SET_NULL_RET(err, ptr, reterr)                               \
  if (err != CL_SUCCESS) {                                                     \
    if (ptr != nullptr)                                                        \
      *ptr = nullptr;                                                          \
    return sycl::detail::pi::cast<pi_result>(reterr);                          \
  }

static_assert(VK_HEADER_VERSION >= 141, "Vulkan Header Version too low");

const char SupportedVersion[] = _PI_H_VERSION_STRING;

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

// Want all the needed casts be explicit, do not define conversion operators.
template <class To, class From> To cast(From value) {
  // TODO: see if more sanity checks are possible.
  static_assert(sizeof(From) == sizeof(To), "cast failed size check");
  return (To)(value);
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

/// \endcond

} // anonymous namespace

template <typename... Args>
void mySaveSnprintf(size_t param_value_size, void *param_value,
                    size_t *param_value_size_ret, const char *format,
                    Args... args) {
  if (param_value) {
    snprintf(reinterpret_cast<char *>(param_value), param_value_size, format,
             args...);
  } else if (param_value_size_ret) {
    *param_value_size_ret = snprintf(nullptr, 0, format, args...) + 1;
  }
}

pi_result mapVulkanErrToCLErr(vk::Result result) {
  switch (result) {
  case vk::Result::eSuccess:
    return pi_result::PI_SUCCESS;
  case vk::Result::eErrorOutOfHostMemory:
    return pi_result::PI_OUT_OF_HOST_MEMORY;
  case vk::Result::eErrorOutOfDeviceMemory:
  case vk::Result::eErrorOutOfPoolMemory:
    return pi_result::PI_OUT_OF_RESOURCES;
  case vk::Result::eErrorDeviceLost:
    return pi_result::PI_INVALID_DEVICE;
  case vk::Result::eErrorInvalidExternalHandle:
    return pi_result::PI_INVALID_MEM_OBJECT;
  case vk::Result::eErrorInvalidShaderNV:
    return pi_result::PI_BUILD_PROGRAM_FAILURE;
  case vk::Result::eErrorMemoryMapFailed:
    return pi_result::PI_INVALID_MEM_OBJECT;
  default:
    return pi_result::PI_ERROR_UNKNOWN;
  }
}

pi_result mapVulkanErrToCLErr(const vk::SystemError &result) {
  return mapVulkanErrToCLErr(static_cast<vk::Result>(result.code().value()));
}

vk::Result getBestQueueNPH(vk::PhysicalDevice &physicalDevice,
                           uint32_t &queueFamilyIndex, vk::QueueFlags include,
                           vk::QueueFlags exclude) {

  auto Properties = physicalDevice.getQueueFamilyProperties();
  // Find optimal queue with included and excluded properties
  int i = 0;
  for (auto Property : Properties) {
    vk::QueueFlags MaskedFlags = Property.queueFlags;
    if (!(exclude & MaskedFlags) && (include & MaskedFlags)) {
      queueFamilyIndex = i;
      return vk::Result::eSuccess;
    }
    i++;
  }
  // Fallback find at least a queue with included properties
  i = 0;
  for (auto Property : Properties) {
    vk::QueueFlags MaskedFlags = Property.queueFlags;
    if (include & MaskedFlags) {
      queueFamilyIndex = i;
      return vk::Result::eSuccess;
    }
    i++;
  }
  // If there is no queue with requested properties -> fail
  return vk::Result::eErrorInitializationFailed;
}

void _pi_mem::allocMemory(vk::Buffer &Buffer, uint32_t MemoryTypeIndex,
                          vk::Buffer &BufferTarget,
                          vk::DeviceMemory &MemoryTarget, void *host_ptr) {

  BufferTarget = Buffer;
  auto BufferRequirements =
      Context_->Device->getBufferMemoryRequirements(Buffer);

  vk::StructureChain<vk::MemoryAllocateInfo, vk::ImportMemoryHostPointerInfoEXT>
      AllocInfo = {
          vk::MemoryAllocateInfo(BufferRequirements.size, MemoryTypeIndex),
          vk::ImportMemoryHostPointerInfoEXT()};

  // If Host Pointer is given, import it
  if (host_ptr) {
    AllocInfo.get<vk::ImportMemoryHostPointerInfoEXT>().setHandleType(
        vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT);
    AllocInfo.get<vk::ImportMemoryHostPointerInfoEXT>().setPHostPointer(
        host_ptr);
  } else {
    AllocInfo.unlink<vk::ImportMemoryHostPointerInfoEXT>();
  }

  MemoryTarget =
      Context_->Device->allocateMemory(AllocInfo.get<vk::MemoryAllocateInfo>());
  Context_->Device->bindBufferMemory(Buffer, MemoryTarget, 0);
}

void lateFree(pi_context Context_, _pi_execution *Execution, uint64_t waitFor) {
  vk::DispatchLoaderDynamic dldid(Context_->PhDevice_->Platform_->Instance_,
                                  vkGetInstanceProcAddr, *Context_->Device,
                                  vkGetDeviceProcAddr);
  Context_->Device->waitSemaphoresKHR(
      vk::SemaphoreWaitInfoKHR(vk::SemaphoreWaitFlagsKHR(), 1,
                               &Execution->Semaphore->Semaphore.get(),
                               &waitFor),
      UINT64_MAX, dldid);
}

_pi_execution::uptr_t
_pi_mem::copy(vk::Buffer &from, vk::Buffer &to,
              vk::ArrayProxy<const vk::BufferCopy> regions, bool isBlocking,
              pi_uint32 num_events, const pi_event *event_list) {

  auto TransferQueue =
      Context_->Device->getQueue(Context_->TransferQueueFamilyIndex, 0);
  _pi_execution::uptr_t Execution = std::make_unique<_pi_execution>(Context_);
  Execution->CommandPool = Context_->Device->createCommandPoolUnique(
      vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eTransient,
                                Context_->TransferQueueFamilyIndex));
  Execution->CommandBuffer = std::move(
      Context_->Device
          ->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
              Execution->CommandPool.get(), vk::CommandBufferLevel::ePrimary,
              1))
          .front());

  Execution->CommandBuffer->begin(
      vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(
          vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
  Execution->CommandBuffer->copyBuffer(from, to, regions);
  Execution->CommandBuffer->end();

  Execution->addAllEventsDependencies(num_events, event_list);
  Execution->MemoryReferences.emplace_back(this);

  Execution->Semaphore = _pi_semaphore::createNew(*Context_->Device);
  const uint64_t Counter = 1;

  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfo>
      SubmitInfo = {vk::SubmitInfo(0, nullptr, nullptr, 1,
                                   &Execution->CommandBuffer.get(), 1,
                                   &Execution->Semaphore->Semaphore.get()),
                    vk::TimelineSemaphoreSubmitInfo(0, nullptr, 1, &Counter)};

  _pi_timeline_event::setWaitingSemaphores(num_events, event_list, SubmitInfo);

  Execution->Fence = Context_->Device->createFenceUnique(
      vk::FenceCreateInfo(vk::FenceCreateFlags()));
  auto Result = TransferQueue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                                     Execution->Fence.get());
  assert(Result == vk::Result::eSuccess && "Memory transfer not successful");
  if (isBlocking) {
    lateFree(Context_, Execution.get(), Counter);
  }
  return Execution;
}

pi_result _pi_timeline_event::getProfilingInfo(pi_profiling_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret) {

  auto limits = Context->PhDevice_->PhDevice.getProperties().limits;

  uint32_t QueryCounter;
  pi_uint64 Timestamp;
  switch (param_name) {
  case PI_PROFILING_INFO_COMMAND_QUEUED:
  case PI_PROFILING_INFO_COMMAND_SUBMIT:
  case PI_PROFILING_INFO_COMMAND_START:
    QueryCounter = 0;
    break;
  case PI_PROFILING_INFO_COMMAND_END:
    QueryCounter = 1;
    break;
  default:
    return PI_PROFILING_INFO_NOT_AVAILABLE;
  }
  Context->Device->getQueryPoolResults(
      Execution->QueryPool.get(), QueryCounter, /*queryCount=*/1,
      sizeof(Timestamp), &Timestamp, /*Stride=*/sizeof(Timestamp),
      vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64);
  // Timestamp may not have full 64bit size, but specification 
  // (timestampValidBits in https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/chap6.html#vkGetPhysicalDeviceQueueFamilyProperties)
  // claims that "Bits outside the valid range are guaranteed to be zeros"
  // so no bitmasking is needed, however it can limit the maximum 
  // range of measurements
  return getInfo(param_value_size, param_value, param_value_size_ret,
                 static_cast<decltype(Timestamp)>(Timestamp *
                     static_cast<double>(limits.timestampPeriod)));
}

_pi_timeline_event::event_info_t
_pi_timeline_event::extractSemaphores(pi_uint32 num_events,
                                      const pi_event *event_list) {

  _pi_timeline_event::event_info_t ExtractionResult;

  for (pi_uint32 i = 0; i < num_events; i++) {
    if (auto TimelineEvent =
            dynamic_cast<const _pi_timeline_event *>(event_list[i])) {
      std::get<0>(ExtractionResult)
          .push_back(TimelineEvent->Execution->Semaphore->Semaphore.get());
      std::get<1>(ExtractionResult).push_back(TimelineEvent->Value);
      std::get<2>(ExtractionResult)
          .push_back(vk::PipelineStageFlagBits::eTopOfPipe);
    }
  }

  return ExtractionResult;
}

_pi_timeline_event::event_info_t _pi_timeline_event::WaitSemaphores;

void _pi_timeline_event::setWaitingSemaphores(
    pi_uint32 num_events, const pi_event *event_list,
    vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
        &submitInfo) {
  if (num_events > 0) {
    WaitSemaphores =
        _pi_timeline_event::extractSemaphores(num_events, event_list);
    submitInfo.get<vk::SubmitInfo>().setWaitSemaphoreCount(
        std::get<0>(WaitSemaphores).size());
    submitInfo.get<vk::SubmitInfo>().setPWaitSemaphores(
        std::get<0>(WaitSemaphores).data());
    submitInfo.get<vk::SubmitInfo>().setPWaitDstStageMask(
        std::get<2>(WaitSemaphores).data());
    submitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setWaitSemaphoreValueCount(std::get<1>(WaitSemaphores).size());
    submitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setPWaitSemaphoreValues(std::get<1>(WaitSemaphores).data());
  }
}

void localCopy(pi_mem memobj, void *targetPtr, size_t size, size_t offset,
               _pi_semaphore::sptr_t semaphore, uint64_t waitValue) {
  vk::DispatchLoaderDynamic dldid(
      memobj->Context_->PhDevice_->Platform_->Instance_, vkGetInstanceProcAddr,
      *memobj->Context_->Device, vkGetDeviceProcAddr);
  memobj->Context_->Device->waitSemaphoresKHR(
      vk::SemaphoreWaitInfoKHR(vk::SemaphoreWaitFlagsKHR(), 1,
                               &semaphore->Semaphore.get(), &waitValue),
      UINT64_MAX, dldid);

  void *BufferPtr =
      memobj->Context_->Device->mapMemory(memobj->HostMemory, offset, size);

  if (std::memcpy(targetPtr, BufferPtr, size) != targetPtr) {
    assert(false && "Error on Memcopy");
  }
  memobj->Context_->Device->unmapMemory(memobj->HostMemory);

  // vk::SemaphoreSignalInfo SignalInfo(memobj->Context_->Timeline.get(),
  // waitValue + 1);
  memobj->Context_->Device->signalSemaphoreKHR(
      vk::SemaphoreSignalInfoKHR(semaphore->Semaphore.get(), waitValue + 1),
      dldid);
}

vk::UniqueQueryPool enableProfiling(pi_queue Queue,
                                    vk::CommandBuffer &CommandBuffer) {
  vk::UniqueQueryPool QueryPool(nullptr);
  if (Queue->isProfilingEnabled()) {
    QueryPool =
        Queue->Context_->Device->createQueryPoolUnique(vk::QueryPoolCreateInfo(
            vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2));
    CommandBuffer.resetQueryPool(QueryPool.get(), 0, 2);
    CommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                 QueryPool.get(), 0);
  }
  return QueryPool;
}

void writeFinishTimestamp(pi_queue Queue, vk::CommandBuffer &CommandBuffer,
                          vk::QueryPool QueryPool) {
  if (Queue->isProfilingEnabled()) {
    CommandBuffer.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                                 QueryPool, 1);
  }
}

/// ------ Error handling, matching OpenCL plugin semantics.
__SYCL_INLINE_NAMESPACE(cl) {
  namespace sycl {
  namespace detail {
  namespace pi {

  // Report error and no return (keeps compiler from printing warnings).
  // TODO: Probably change that to throw a catchable exception,
  //       but for now it is useful to see every failure.
  //
  [[noreturn]] __SYCL_EXPORT void die(const char *Message) {
    std::cerr << "pi_die: " << Message << std::endl;
    std::terminate();
  }

  // void assertion(bool Condition, const char *Message) {
  //  if (!Condition)
  //    die(Message);
  //}

  } // namespace pi

  __SYCL_EXPORT const char *stringifyErrorCode(cl_int error) {
    return "Vulkan Error Code (not implemented)";
  }

  } // namespace detail

  const char *exception::what() const noexcept { return MMsg.c_str(); }

  } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

// Convenience macro makes source code search easier
#define VLK(pi_api) Vulkan##pi_api

extern "C" {
// Predefine
pi_result VLK(piMemBufferCreate)(pi_context context, pi_mem_flags flags,
                                 size_t size, void *host_ptr, pi_mem *ret_mem,
                                 const pi_mem_properties *properties = nullptr);
pi_result VLK(piMemRetain)(pi_mem mem);
pi_result VLK(piEnqueueMemBufferWrite)(pi_queue command_queue, pi_mem memobj,
                                       pi_bool blocking_write, size_t offset,
                                       size_t size, const void *ptr,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event);
pi_result VLK(piEnqueueMemBufferMap)(pi_queue command_queue, pi_mem memobj,
                                     pi_bool blocking_map,
                                     cl_map_flags map_flags, size_t offset,
                                     size_t size,
                                     pi_uint32 num_events_in_wait_list,
                                     const pi_event *event_wait_list,
                                     pi_event *event, void **ret_map);
pi_result VLK(piEnqueueMemUnmap)(pi_queue command_queue, pi_mem memobj,
                                 void *mapped_ptr,
                                 pi_uint32 num_events_in_wait_list,
                                 const pi_event *event_wait_list,
                                 pi_event *event);
pi_result VLK(piEventRetain)(pi_event event);
}

void _pi_execution::addEventDependency(pi_event event) {
  /*VLK(piEventRetain)(event);
  DependendEvents.push_back(event);*/
  if (auto TimelineEvent = dynamic_cast<_pi_timeline_event *>(event))
    DependendSemaphores.push_back(TimelineEvent->Execution->Semaphore);
}

bool _pi_execution::isDone() const {
  return Context_->Device->getFenceStatus(Fence.get()) == vk::Result::eSuccess;
}

void _pi_execution::wait() const {
  auto Result =
      Context_->Device->waitForFences(1u, &Fence.get(), true, UINT64_MAX);
  assert(Result == vk::Result::eSuccess);
}

size_t _pi_kernel::getInternalIndex(pi_uint32 ArgIdx) {
  auto InternalIndex = ArgIndexToInternalIndex.find(ArgIdx);
  if (InternalIndex != ArgIndexToInternalIndex.end())
    return InternalIndex->second;
  // Two lines only allowed to be combined above C++17
  // otherwise undefined evaluation order
  auto Size = ArgIndexToInternalIndex.size();
  return ArgIndexToInternalIndex[ArgIdx] = Size;
}

pi_uint32 _pi_kernel::getArgumentIndex(size_t IntIdx) const {
  for (auto it : ArgIndexToInternalIndex) {
    if (it.second == IntIdx)
      return it.first;
  }
  return -1;
}

pi_result _pi_kernel::addArgument(pi_uint32 ArgIndex, const pi_mem *Memobj) {

  auto InternalIndex = getInternalIndex(ArgIndex);

  if (DescriptorSetLayoutBinding.size() <= InternalIndex) {
    DescriptorSetLayoutBinding.resize(InternalIndex + 1);
  }

  DescriptorSetLayoutBinding[InternalIndex] = {
      ArgIndex, vk::DescriptorType::eStorageBuffer, 1,
      vk::ShaderStageFlagBits::eCompute};

  Arguments[InternalIndex] = *Memobj;
  if (Arguments[InternalIndex]->InitExecution.get()) {
    AdditionalMemoryEvents.push_back(new _pi_timeline_event(
        Program_->Context_, std::move(Arguments[InternalIndex]->InitExecution),
        1ull));
  }
  VLK(piMemRetain)(Arguments[InternalIndex]);

  return PI_SUCCESS;
}

pi_result _pi_kernel::addArgument(pi_uint32 ArgIndex, size_t arg_size,
                                  const void *arg_value) {

  auto InternalIndex = getInternalIndex(ArgIndex);

  auto vec = Arguments.find(InternalIndex);
  if (vec == Arguments.end()) {
    // Argument is set the first time
    // Prepare DescriptorSetLayout and create an own Buffer
    if (DescriptorSetLayoutBinding.size() <= InternalIndex)
      DescriptorSetLayoutBinding.resize(InternalIndex + 1);
    DescriptorSetLayoutBinding[InternalIndex] = {
        ArgIndex, vk::DescriptorType::eStorageBuffer, 1,
        vk::ShaderStageFlagBits::eCompute};

    pi_mem mem;
    VLK(piMemBufferCreate)
    (Program_->Context_, PI_MEM_FLAGS_HOST_PTR_COPY, arg_size,
     const_cast<void *>(arg_value), &mem);

    AdditionalMemoryEvents.push_back(new _pi_timeline_event(
        Program_->Context_, std::move(mem->InitExecution), 1ull));

    Arguments[InternalIndex] = mem;
  } else {
    // Argument is reset at another time
    // Only update Buffer on device if data has changed, for performance boost
    void *HostMemory = Program_->Context_->Device->mapMemory(
        vec->second->HostMemory, 0, arg_size);
    if (memcmp(HostMemory, arg_value, arg_size) != 0) {
      // Copy to Host buffer
      memcpy(HostMemory, arg_value, arg_size);
      // Enqueue copy to device
      // If memory is used in a call, wait for the job to finish
      // before overwriting it
      auto Execution = vec->second->copyHtoD(LastLaunch ? 1 : 0, &LastLaunch);
      // runtime does not support events here, so push
      // wait event outside of runtime knowledge, so kernel launches
      // can wait until memory is written
      AdditionalMemoryEvents.push_back(new _pi_timeline_event(
          Program_->Context_, std::move(Execution), 1ull));
    }
    Program_->Context_->Device->unmapMemory(vec->second->HostMemory);
  }
  return PI_SUCCESS;
}

std::shared_ptr<_pi_semaphore> _pi_semaphore::createNew(vk::Device &device) {
  vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfoKHR>
      SemaphoreCreate = {
          vk::SemaphoreCreateInfo(vk::SemaphoreCreateFlags()),
          vk::SemaphoreTypeCreateInfoKHR(vk::SemaphoreTypeKHR::eTimeline, 0)};

  return std::make_shared<_pi_semaphore>(device.createSemaphoreUnique(
      SemaphoreCreate.get<vk::SemaphoreCreateInfo>()));
}

void _pi_queue::cleanupFinishedExecutions() {
  std::remove_if(StoredExecutions.begin(), StoredExecutions.end(),
                 [this](auto &execution) {
                   return Context_->Device->getFenceStatus(
                              execution->Fence.get()) != vk::Result::eNotReady;
                 });
}

vk::UniqueCommandBuffer _pi_queue::createCommandBuffer() {
  cleanupFinishedExecutions();

  return std::move(
      Context_->Device
          ->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
              CommandPool.get(), vk::CommandBufferLevel::ePrimary, 1u))
          .front());
}

extern "C" {

#define NOT_IMPL(functionname, parameters)                                     \
  functionname parameters {                                                    \
    std::stringstream ss;                                                      \
    ss << #functionname << " not implemented";                                 \
    sycl::detail::pi::die(ss.str().c_str());                                   \
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
#ifdef WIN32
          // At init, on windows
          if (HMODULE mod = GetModuleHandleA("renderdoc.dll")) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI =
                (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2,
                                       (void **)&rdoc_api);
            assert(ret == 1);
          }
#else
          // At init, on linux/android.
          // For android replace librenderdoc.so with
          // libVkLayer_GLES_RenderDoc.so
          if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
            pRENDERDOC_GetAPI RENDERDOC_GetAPI =
                (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
            int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2,
                                       (void **)&rdoc_api);
            assert(ret == 1);
          }
#endif

          // initialize the vk::ApplicationInfo structure
          vk::ApplicationInfo applicationInfo("SYCL LLVM", 1, "pi_vulkan.cpp",
                                              1, VK_API_VERSION_1_1);

          // initialize the vk::InstanceCreateInfo
          std::vector<const char *> List = {
              //"VK_LAYER_LUNARG_vktrace",
              //"VK_LAYER_LUNARG_api_dump",
              //"VK_LAYER_KHRONOS_validation"
          };
          vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo,
                                                    List.size(), List.data());
          Platform.Instance_ = vk::createInstance(instanceCreateInfo);
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
                   "Vulkan 1.1");
  case PI_PLATFORM_INFO_EXTENSIONS:
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Platform info request not implemented");
  return {};
}

// Example of a PI interface that does not map exactly to an OpenCL one.
pi_result VLK(piDevicesGet)(pi_platform platform, pi_device_type device_type,
                            pi_uint32 num_entries, pi_device *devices,
                            pi_uint32 *num_devices) {

  // Needed that getDevice of Queue always returns the same
  // Device, also if two different queues where created
  // See testcase device_equality.cpp
  static std::map<vk::PhysicalDevice, _pi_device> DeviceStorage;

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
                         if (!(device_type & PI_DEVICE_TYPE_GPU) &&
                             !(device_type & PI_DEVICE_TYPE_DEFAULT))
                           return true;
                         break;
                       default:
                         return true;
                       }
                       return false;
                     });

  uint32_t SizeRelevantDevices =
      std::distance(DevicesVec.begin(), RelevantDevicesEnd);
  if (SizeRelevantDevices < 1) {
    // Absorb the CL_DEVICE_NOT_FOUND and just return 0 in num_devices
    // to satisfy test: 'basic_tests/diagnostics/device-check.cpp'
    // return PI_DEVICE_NOT_FOUND;
  }
  if (devices) {
    uint32_t MaxEntries = std::min<uint32_t>(num_entries, SizeRelevantDevices);
    std::vector<vk::PhysicalDevice>::iterator Device = DevicesVec.begin();
    for (uint32_t i = 0; i < MaxEntries; i++, Device++) {
      auto Result = DeviceStorage.find(*Device);
      if (Result == DeviceStorage.end()) {
        Result =
            DeviceStorage
                .emplace(std::make_pair(*Device, _pi_device(*Device, platform)))
                .first;
      }
      devices[i] = &(Result->second);
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
    std::array<size_t, MaxWorkItemDims> Limits = {
        Properties.limits.maxComputeWorkGroupSize[0],
        Properties.limits.maxComputeWorkGroupSize[1],
        Properties.limits.maxComputeWorkGroupSize[2]};
    return getInfoArray(MaxWorkItemDims, param_value_size, param_value,
                        param_value_size_ret, Limits.data());
  }
  case PI_DEVICE_INFO_MAX_WORK_GROUP_SIZE: {
    auto Limits = Properties.limits;
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   size_t(Limits.maxComputeWorkGroupCount[0]));
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
                   pi_uint32(limits.minStorageBufferOffsetAlignment));
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
    return getInfoArray(Properties.deviceName.size(), param_value_size,
                        param_value, param_value_size_ret,
                        Properties.deviceName.data());
  }
  case PI_DEVICE_INFO_VENDOR: {
    const char *OutputString = "Unknown vendor: 0x%x";
    switch (Properties.vendorID) {
    case 0x1002:
      OutputString = "AMD";
      break;
    case 0x1010:
      OutputString = "ImgTec";
      break;
    case 0x10DE:
      OutputString = "NVIDIA";
      break;
    case 0x13B5:
      OutputString = "ARM";
      break;
    case 0x5143:
      OutputString = "Qualcomm";
      break;
    case 0x8086:
      OutputString = "INTEL";
      break;
    default:
      break;
    }
    mySaveSnprintf(param_value_size, param_value, param_value_size_ret,
                   OutputString, Properties.vendorID);
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_DRIVER_VERSION: {
    mySaveSnprintf(param_value_size, param_value, param_value_size_ret, "%d.%d",
                   VK_VERSION_MAJOR(Properties.driverVersion),
                   VK_VERSION_MINOR(Properties.driverVersion));
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_PROFILE: {
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "Vulkan");
  }
  case PI_DEVICE_INFO_REFERENCE_COUNT: {
    return getInfo(param_value_size, param_value, param_value_size_ret, 1);
  }
  case PI_DEVICE_INFO_VERSION: {
    mySaveSnprintf(param_value_size, param_value, param_value_size_ret,
                   "Vulkan %d.%d", VK_VERSION_MAJOR(Properties.apiVersion),
                   VK_VERSION_MINOR(Properties.apiVersion));
    return PI_SUCCESS;
  }
  case PI_DEVICE_INFO_OPENCL_C_VERSION: {
    return getInfo(param_value_size, param_value, param_value_size_ret, "");
  }
  case PI_DEVICE_INFO_EXTENSIONS: {
    // std::vector<vk::ExtensionProperties> extensions =
    //     Device->PhDevice.enumerateDeviceExtensionProperties();
    // return getInfo(param_value_size, param_value, param_value_size_ret,
    //                &extensions);
    // Pretend Vulkan is capable of these extensions, since
    // fallback SPIRV code is not Vulkan compatible yet
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   "cl_intel_devicelib_assert cl_intel_devicelib_math "
                   "cl_intel_devicelib_math_fp64 cl_intel_devicelib_complex "
                   "cl_intel_devicelib_complex_fp64");
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Device info request not implemented");
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
  // No Release of Physical Devices possible in Vulkan
  // API. They are bound to the Instance
  // delete Device;
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

pi_result VLK(piextDeviceCreateWithNativeHandle)(pi_native_handle nativeHandle,
                                                 pi_platform platform,
                                                 pi_device *device) {
  sycl::detail::pi::die(
      "vulkan_piextDeviceCreateWithNativeHandle not implemented");
  return {};
}

/// \return If available, the first binary that is Vulkan Compatible
///
pi_result VLK(piextDeviceSelectBinary)(pi_device Device,
                                       pi_device_binary *binaries,
                                       pi_uint32 num_binaries,
                                       pi_uint32 *selected_image_ind) {
  if (!binaries) {
    sycl::detail::pi::die("No list of device images provided");
  }
  if (num_binaries < 1) {
    sycl::detail::pi::die("No binary images in the list");
  }
  if (!selected_image_ind) {
    sycl::detail::pi::die("No storage for device binary index provided");
  }

  // Look for an image for the SPIRV Image compatible for vulkan
  // If SPV injection with environmental variable is required
  // also allow SPIRV64 binaries, since the loader
  // cannot differ between them if the MAGIC number is SPIRV in both cases
  // TODO: Maybe use a SPIRV Reader and look for Shader capability and
  // compatible Memory model for clear differentiation
  for (pi_uint32 i = 0; i < num_binaries; i++) {
    if (strcmp(binaries[i]->DeviceTargetSpec,
               __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_VULKAN) == 0 ||
        (std::getenv("SYCL_USE_KERNEL_SPV") &&
         strcmp(binaries[i]->DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64) == 0)) {
      *selected_image_ind = i;
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
  sycl::detail::pi::die("VLK(piextGetDeviceFunctionPointer not implemented");
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
  std::vector<vk::QueueFamilyProperties> QueueFamilyProperties =
      PhysicalDevice.getQueueFamilyProperties();

  uint32_t ComputeQueueFamilyIndex = 0;

  // get the best index into queueFamiliyProperties which supports compute and
  // stuff
  vk::Result VkRes = getBestQueueNPH(PhysicalDevice, ComputeQueueFamilyIndex,
                                     vk::QueueFlagBits::eCompute,
                                     vk::QueueFlagBits::eGraphics);
  assert(VkRes == vk::Result::eSuccess);

  pi_context Context = new _pi_context();
  Context->RefCounter_ = 1;
  Context->PhDevice_ = Device;
  Context->ComputeQueueFamilyIndex = ComputeQueueFamilyIndex;
  VkRes = getBestQueueNPH(PhysicalDevice, Context->TransferQueueFamilyIndex,
                          vk::QueueFlagBits::eTransfer,
                          vk::QueueFlagBits::eGraphics |
                              vk::QueueFlagBits::eCompute);
  assert(VkRes == vk::Result::eSuccess);

  const char *EXTExternalMemoryImport = "VK_EXT_external_memory_host";
  std::vector<const char *> EnabledExtensions = {"VK_KHR_variable_pointers",
                                                 "VK_KHR_timeline_semaphore",
                                                 "VK_KHR_shader_float16_int8"};

  auto AvailableExtensions =
      Context->PhDevice_->PhDevice.enumerateDeviceExtensionProperties();
  for (auto Extension : AvailableExtensions) {
    if (strcmp(Extension.extensionName, EXTExternalMemoryImport) == 0) {
      EnabledExtensions.push_back(EXTExternalMemoryImport);
      Context->EXTExternalMemoryImportEnabled = true;
      break;
    }
  }

  const uint32_t MaximumNumberQueues = 1;

  std::vector<float> QueuePriority(MaximumNumberQueues, 0.0f);
  for (uint32_t i = 0; i < MaximumNumberQueues; i++) {
    Context->AvailableQueueIndizes.push(i);
  }

  // create a UniqueDevice
  std::vector<vk::DeviceQueueCreateInfo> DeviceQueueCreateInfo = {
      vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(),
                                Context->ComputeQueueFamilyIndex,
                                MaximumNumberQueues, QueuePriority.data())};

  if (Context->TransferQueueFamilyIndex != Context->ComputeQueueFamilyIndex) {
    DeviceQueueCreateInfo.emplace_back(vk::DeviceQueueCreateFlags(),
                                       Context->TransferQueueFamilyIndex, 1,
                                       QueuePriority.data());
  };

  vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceShaderFloat16Int8Features,
                     vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR>
      CreateDeviceInfo = {
          vk::DeviceCreateInfo(vk::DeviceCreateFlags(),
                               DeviceQueueCreateInfo.size(),
                               DeviceQueueCreateInfo.data(), 0, nullptr,
                               static_cast<uint32_t>(EnabledExtensions.size()),
                               EnabledExtensions.data()),
          vk::PhysicalDeviceFeatures2(), vk::PhysicalDeviceVulkan11Features(),
          vk::PhysicalDeviceShaderFloat16Int8Features(false, true),
          vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR(true)};
  CreateDeviceInfo.get<vk::PhysicalDeviceFeatures2>().features.setShaderInt64(
      true);
  CreateDeviceInfo.get<vk::PhysicalDeviceVulkan11Features>()
      .setVariablePointers(true);
  CreateDeviceInfo.get<vk::PhysicalDeviceVulkan11Features>()
      .setVariablePointersStorageBuffer(true);

  Context->Device = PhysicalDevice.createDeviceUnique(
      CreateDeviceInfo.get<vk::DeviceCreateInfo>());

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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
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

NOT_IMPL(pi_result VLK(piextContextCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_uint32 numDevices,
          const pi_device *devices, pi_context *context))

pi_result VLK(piContextRetain)(pi_context context) {
  context->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piContextRelease)(pi_context context) {
  context->RefCounter_--;
  if (context->RefCounter_ < 1)
    delete context;
  return PI_SUCCESS;
}

pi_result VLK(piQueueCreate)(pi_context context, pi_device Device,
                             pi_queue_properties properties, pi_queue *Queue) {
  assert(Queue && "piQueueCreate failed, queue argument is null");

  if (context->AvailableQueueIndizes.empty()) {
    return PI_OUT_OF_RESOURCES;
  }

  // TODO: think about incrementing Queue Index
  auto CommandPool =
      context->Device->createCommandPoolUnique(vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          context->ComputeQueueFamilyIndex));
  *Queue = new _pi_queue(
      /*queue=*/context->Device->getQueue(
          context->ComputeQueueFamilyIndex,
          context->AvailableQueueIndizes.front()),
      std::move(CommandPool), context, properties,
      /*queueindex=*/context->AvailableQueueIndizes.front());
  context->AvailableQueueIndizes.pop();
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Queue info request not implemented");
  return {};
}

pi_result VLK(piQueueRetain)(pi_queue Queue) {
  Queue->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piQueueRelease)(pi_queue Queue) {
  Queue->RefCounter_--;
  if (Queue->RefCounter_ < 1)
    delete Queue;
  return PI_SUCCESS;
}

pi_result VLK(piQueueFinish)(pi_queue command_queue) {
  // TODO: Does this belong here?

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

NOT_IMPL(pi_result VLK(piextQueueCreateWithNativeHandle),
         (pi_native_handle nativeHandle, pi_context context, pi_queue *queue))

pi_result VLK(piMemBufferCreate)(pi_context context, pi_mem_flags flags,
                                 size_t size, void *host_ptr, pi_mem *ret_mem,
                                 const pi_mem_properties *properties) {

  // set memoryTypeIndex to an invalid entry in the properties.memoryTypes array
  uint32_t MemoryTypeIndexHost = VK_MAX_MEMORY_TYPES;
  uint32_t MemoryTypeIndexDevice = VK_MAX_MEMORY_TYPES;

  auto MemoryProperties = context->PhDevice_->PhDevice.getMemoryProperties();
  auto MemImportProperties =
      context->PhDevice_->PhDevice
          .getProperties2<vk::PhysicalDeviceProperties2,
                          vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();

  vk::MemoryHostPointerPropertiesEXT extMemProperties{UINT32_MAX};

  if (context->EXTExternalMemoryImportEnabled &&
      flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    vk::DispatchLoaderDynamic dll(context->PhDevice_->Platform_->Instance_,
                                  vkGetInstanceProcAddr, *context->Device,
                                  vkGetDeviceProcAddr);
    extMemProperties = context->Device->getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBitsKHR::eHostAllocationEXT, host_ptr,
        dll);
  }

  // Find valid memory indizes for host and device buffer
  // TODO: what if too much memory is allocated in smaller steps?
  for (uint32_t k = 0; k < MemoryProperties.memoryTypeCount; k++) {
    // Host memory must be host visible, size must fit
    // and if import the memory index must be supported for import
    if (vk::MemoryPropertyFlagBits::eHostVisible &
            MemoryProperties.memoryTypes[k].propertyFlags &&
        (size <
         MemoryProperties.memoryHeaps[MemoryProperties.memoryTypes[k].heapIndex]
             .size) &&
        (extMemProperties.memoryTypeBits & (1 << k)) != 0) {

      if (MemoryTypeIndexHost == VK_MAX_MEMORY_TYPES)
        MemoryTypeIndexHost = k;
      // Device memory should be device local and size must fit
    } else if (vk::MemoryPropertyFlagBits::eDeviceLocal &
                   MemoryProperties.memoryTypes[k].propertyFlags &&
               (size <
                MemoryProperties
                    .memoryHeaps[MemoryProperties.memoryTypes[k].heapIndex]
                    .size)) {

      if (MemoryTypeIndexDevice == VK_MAX_MEMORY_TYPES)
        MemoryTypeIndexDevice = k;
    }
  }

  if (MemoryTypeIndexHost == VK_MAX_MEMORY_TYPES ||
      MemoryTypeIndexDevice == VK_MAX_MEMORY_TYPES)
    return PI_OUT_OF_HOST_MEMORY; // or PI_OUT_OF_RESOURCES?

  bool importAllowed = false;
  if (context->EXTExternalMemoryImportEnabled &&
      flags & PI_MEM_FLAGS_HOST_PTR_USE) {
    if (size % MemImportProperties
                    .get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
                    .minImportedHostPointerAlignment ==
            0 &&
        (reinterpret_cast<std::uintptr_t>(host_ptr) %
             MemImportProperties
                 .get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
                 .minImportedHostPointerAlignment ==
         0) &&
        (extMemProperties.memoryTypeBits & (1 << MemoryTypeIndexHost)) != 0) {
      importAllowed = true;
    }
  }

  try {
    // It seems that the PI does not say, whether the buffer
    // is used for read or write only. If it works somehow,
    // Dst & Src could be specified more clearly.
    vk::BufferUsageFlags BufferCreationFlags =
        vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eTransferSrc;

    pi_mem NewMem =
        new _pi_mem(context, flags, size, importAllowed ? nullptr : host_ptr);

    std::vector<uint32_t> FamiliyIndizes = {context->ComputeQueueFamilyIndex};
    if (context->ComputeQueueFamilyIndex != context->TransferQueueFamilyIndex) {
      FamiliyIndizes.push_back(context->TransferQueueFamilyIndex);
    }

    NewMem->allocHostMemory(
        context->Device->createBuffer(vk::BufferCreateInfo(
            vk::BufferCreateFlags(), size,
            vk::BufferUsageFlagBits::eStorageBuffer | BufferCreationFlags,
            FamiliyIndizes.size() == 1 ? vk::SharingMode::eExclusive
                                       : vk::SharingMode::eConcurrent,
            FamiliyIndizes.size(), FamiliyIndizes.data())),
        MemoryTypeIndexHost, importAllowed ? host_ptr : nullptr);
    NewMem->allocDeviceMemory(
        context->Device->createBuffer(vk::BufferCreateInfo(
            vk::BufferCreateFlags(), size,
            vk::BufferUsageFlagBits::eStorageBuffer | BufferCreationFlags,
            FamiliyIndizes.size() == 1 ? vk::SharingMode::eExclusive
                                       : vk::SharingMode::eConcurrent,
            FamiliyIndizes.size(), FamiliyIndizes.data())),
        MemoryTypeIndexDevice);
    //}
    *ret_mem = NewMem;
    // Host pointer import is very restricted, but know by variable
    // importAllowed If import is not allowed, remember original host pointer
    // but act like PI_MEM_FLAGS_HOST_PTR_COPY at creation.
    // Original host pointer must be returned on memoryMap
    // otherwise the runtime would try to free device memory
    if (flags & PI_MEM_FLAGS_HOST_PTR_USE ||
        flags & PI_MEM_FLAGS_HOST_PTR_COPY) {
      if (flags & PI_MEM_FLAGS_HOST_PTR_COPY || !importAllowed) {
        auto DeviceData =
            context->Device->mapMemory(NewMem->HostMemory, 0, size);
        memcpy(DeviceData, host_ptr, size);
        context->Device->unmapMemory(NewMem->HostMemory);
      }
      // Move data to device asynchronously, store
      // execution to be able to wait for it upon kernel launch
      NewMem->InitExecution =
          NewMem->copyHtoD(/*num_events=*/0, /*evt_ptr=*/nullptr);
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
  sycl::detail::pi::die("VLK(piextGetDeviceFunctionPointer) not implemented");
  return {};
}

pi_result VLK(piMemGetInfo)(pi_mem mem,
                            cl_mem_info param_name, // TODO: untie from OpenCL
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret) {
  sycl::detail::pi::die("VLK(piMemGetInfo) not implemented");
  return {};
}
pi_result VLK(piMemImageGetInfo)(pi_mem image, pi_image_info param_name,
                                 size_t param_value_size, void *param_value,
                                 size_t *param_value_size_ret) {
  sycl::detail::pi::die("VLK(piMemImageGetInfo) not implemented");
  return {};
}

pi_result VLK(piMemRetain)(pi_mem mem) {
  mem->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piMemRelease)(pi_mem mem) {
  mem->RefCounter_--;
  if (mem->RefCounter_ < 1) {
    delete mem;
  }
  return PI_SUCCESS;
}

pi_result VLK(piMemBufferPartition)(pi_mem Buffer, pi_mem_flags flags,
                                    pi_buffer_create_type buffer_create_type,
                                    void *buffer_create_info, pi_mem *ret_mem) {

  sycl::detail::pi::die("VLK(piMemBufferPartition) not implemented");
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
      reinterpret_cast<pi_native_handle>(&mem->HostMemory); // Or return Buffer?
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

  auto Program = std::make_unique<_pi_program>(
      context->Device->createShaderModule(
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
  sycl::detail::pi::die("VLK(piextProgramConvert not implemented");
  return {};
}

pi_result VLK(piProgramCreateWithBinary)(
    pi_context context, pi_uint32 num_devices, const pi_device *device_list,
    const size_t *lengths, const unsigned char **binaries,
    pi_int32 *binary_status, pi_program *ret_program) {
  // TODO: Only one device for now
  // TODO: is binary_status needed to be filled
  auto Program = std::make_unique<_pi_program>(
      context->Device->createShaderModule(vk::ShaderModuleCreateInfo(
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
  case PI_PROGRAM_INFO_NUM_KERNELS: {
    std::istringstream ISS(
        std::string(program->Source_, program->SourceLength_),
        std::ios::binary);
    std::string Err;
    SPIRV::TranslatorOpts::ExtensionsStatusMap map;
    map[SPIRV::ExtensionID::SPV_KHR_variable_pointers] = true;

    auto BM = SPIRV::readSpirvModule(
        ISS, SPIRV::TranslatorOpts(SPIRV::VersionNumber::MaximumVersion, map),
        Err);
    if (BM.get()) {
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     BM->getNumEntryPoints(ExecutionModelGLCompute));
    }
    break;
  }
  case PI_PROGRAM_INFO_KERNEL_NAMES: {
    std::istringstream ISS(
        std::string(program->Source_, program->SourceLength_),
        std::ios::binary);
    std::string Err;

    SPIRV::TranslatorOpts::ExtensionsStatusMap map;
    map[SPIRV::ExtensionID::SPV_KHR_variable_pointers] = true;

    auto BM = SPIRV::readSpirvModule(
        ISS, SPIRV::TranslatorOpts(SPIRV::VersionNumber::MaximumVersion, map),
        Err);
    if (BM.get()) {
      auto NumPoints = BM->getNumEntryPoints(ExecutionModelGLCompute);
      std::string EntryPointNames(
          BM->getEntryPoint(ExecutionModelGLCompute, 0)->getName());
      for (unsigned i = 1; i < NumPoints; i++) {
        EntryPointNames.push_back(';');
        EntryPointNames.append(
            BM->getEntryPoint(ExecutionModelGLCompute, i)->getName());
      }
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     EntryPointNames.c_str());
    }
    break;
  }
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Program info request not implemented");
  return {};
}

pi_result VLK(piProgramLink)(pi_context context, pi_uint32 num_devices,
                             const pi_device *device_list, const char *options,
                             pi_uint32 num_input_programs,
                             const pi_program *input_programs,
                             void (*pfn_notify)(pi_program program,
                                                void *user_data),
                             void *user_data, pi_program *ret_program) {

  // well, Vulkan does not support linking of multiple programs
  // for now, return the original program
  // but there are automatically added programs to simulate
  // features ...
  // Linking is a planned feature of SPIRV-tools wait until this
  // is available
  // TODO: implement this if SPIRV-tool support linking
  *ret_program = input_programs[num_input_programs - 1];
  return VLK(piProgramRetain)(*ret_program);
}

pi_result VLK(piProgramCompile)(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  // sycl::detail::pi::die("VLK(piProgramCompile) not implemented");
  return PI_SUCCESS;
}

pi_result VLK(piProgramBuild)(pi_program program, pi_uint32 num_devices,
                              const pi_device *device_list, const char *options,
                              void (*pfn_notify)(pi_program program,
                                                 void *user_data),
                              void *user_data) {
  // sycl::detail::pi::die("VLK(piProgramBuild) not implemented");
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
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
  }
  sycl::detail::pi::die("Program Build info request not implemented");
  return {};
}

pi_result VLK(piProgramRetain)(pi_program program) {
  program->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piProgramRelease)(pi_program program) {
  program->RefCounter_--;
  if (program->RefCounter_ < 1) {
    program->Context_->Device->destroyShaderModule(program->Module);
    delete program;
  }
  return PI_SUCCESS;
}

// TODO: Implement
pi_result VLK(piextProgramSetSpecializationConstant)(pi_program prog,
                                                     pi_uint32 spec_id,
                                                     size_t spec_size,
                                                     const void *spec_value) {
  sycl::detail::pi::die(
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
                                                  pi_context context,
                                                  pi_program *program) {
  sycl::detail::pi::die(
      "VLK(piextProgramCreateWithNativeHandle) not implemented");
  return {};
}

pi_result VLK(piKernelCreate)(pi_program program, const char *kernel_name,
                              pi_kernel *ret_kernel) {
  *ret_kernel = new _pi_kernel(
      kernel_name, program,
      program->Context_->Device->createPipelineCacheUnique(
          vk::PipelineCacheCreateInfo(vk::PipelineCacheCreateFlags())));
  return PI_SUCCESS;
}

pi_result VLK(piKernelSetArg)(pi_kernel kernel, pi_uint32 arg_index,
                              size_t arg_size, const void *arg_value) {
  // store arguments temporarily
  return kernel->addArgument(arg_index, arg_size, arg_value);
}

pi_result VLK(piKernelGetInfo)(pi_kernel kernel, pi_kernel_info param_name,
                               size_t param_value_size, void *param_value,
                               size_t *param_value_size_ret) {

  if (kernel != nullptr) {

    switch (param_name) {
    case PI_KERNEL_INFO_FUNCTION_NAME:
      return getInfo(param_value_size, param_value, param_value_size_ret,
                     kernel->Name.c_str());
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
    default: { __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name); }
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
      __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
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
  sycl::detail::pi::die("VLK(piKernelGetSubGroupInfo) not implemented");
  return {};
}

pi_result VLK(piKernelRetain)(pi_kernel kernel) {
  kernel->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piKernelRelease)(pi_kernel kernel) {
  kernel->RefCounter_--;
  if (kernel->RefCounter_ < 1) {
    delete kernel;
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
  // NOP for now...
  //?? what to do here?
  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEventCreate),
         (pi_context context, pi_event *ret_event))
pi_result
VLK(piEventGetInfo)(pi_event event,
                    cl_event_info param_name, // TODO: untie from OpenCL
                    size_t param_value_size, void *param_value,
                    size_t *param_value_size_ret) {
  switch (param_name) {
  case PI_EVENT_INFO_COMMAND_EXECUTION_STATUS:
    // FIXME: Get the actual Status
    return getInfo(param_value_size, param_value, param_value_size_ret,
                   PI_EVENT_COMPLETE);
  default:
    __SYCL_PI_HANDLE_UNKNOWN_PARAM_NAME(param_name);
    return PI_INVALID_VALUE;
  }
}
pi_result VLK(piEventGetProfilingInfo)(pi_event event,
                                       pi_profiling_info param_name,
                                       size_t param_value_size,
                                       void *param_value,
                                       size_t *param_value_size_ret) {
  return event->getProfilingInfo(param_name, param_value_size, param_value,
                                 param_value_size_ret);
}

/// Wait for all given Events
pi_result VLK(piEventsWait)(pi_uint32 num_events, const pi_event *event_list) {

  for (pi_uint32 i = 0; i < num_events; i++) {
    event_list[i]->wait();
  }
  return PI_SUCCESS;
}

NOT_IMPL(pi_result VLK(piEventSetCallback),
         (pi_event event, pi_int32 command_exec_callback_type,
          void (*pfn_notify)(pi_event event, pi_int32 event_command_status,
                             void *user_data),
          void *user_data))
NOT_IMPL(pi_result VLK(piEventSetStatus),
         (pi_event event, pi_int32 execution_status))

pi_result VLK(piEventRetain)(pi_event event) {
  event->RefCounter_++;
  return PI_SUCCESS;
}

pi_result VLK(piEventRelease)(pi_event event) {
  if (--event->RefCounter_ < 1)
    delete event;
  return PI_SUCCESS;
}

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
  vk::Result Result = vk::Result::eErrorUnknown;

  // To start a frame capture, call StartFrameCapture.
  // You can specify NULL, NULL for the device to capture on if you
  // have only one device and either no windows at all or only one
  // window, and it will capture from that device. See the
  // documentation below for a longer explanation
  if (rdoc_api)
    rdoc_api->StartFrameCapture(NULL, NULL);

  try {
    _pi_execution::uptr_t Execution =
        std::make_unique<_pi_execution>(Queue->Context_);
    Execution->DescriptorSetLayout = Device->createDescriptorSetLayoutUnique(
        vk::DescriptorSetLayoutCreateInfo(
            vk::DescriptorSetLayoutCreateFlags(),
            static_cast<uint32_t>(kernel->DescriptorSetLayoutBinding.size()),
            kernel->DescriptorSetLayoutBinding.data()));

    // create a PipelineLayout using that DescriptorSetLayout
    Execution->PipelineLayout = Device->createPipelineLayoutUnique(
        vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), 1,
                                     &Execution->DescriptorSetLayout.get()));

    std::vector<uint32_t> Values;
    std::vector<vk::SpecializationMapEntry> Entries;
    if (local_work_size) {
      for (pi_uint32 i = 0; i < work_dim; i++) {
        Values.push_back(local_work_size[i]);
      }
    } else {
      switch (work_dim) {
      case 2:
        Values = {16u, 16u, 1u};
        break;
      case 3:
        Values = {8u, 8u, 8u};
        break;
      default:
        Values = {512u, 1u, 1u};
        break;
      }

      // If there is any conflict with default
      // workgroup sizes simply do 1, 1, 1
      if (global_work_size[0] % Values[0] != 0 ||
          global_work_offset[0] % Values[0] != 0 ||
          (work_dim >= 2 && (global_work_size[1] % Values[1] != 0 ||
                             global_work_offset[1] % Values[1] != 0)) ||
          (work_dim >= 3 && (global_work_size[2] % Values[2] != 0 ||
                             global_work_offset[2] % Values[2] != 0))) {
        Values = {1u, 1u, 1u};
      }
    }

    // Fillup to 3 Workgroupsize values
    // needed that preceding values start with 103
    while (Values.size() < 3)
      Values.push_back(1u);

    // Push GlobalOffset as Specification Constant
    // Vulkan has no Builtin for that unfortunately
    for (pi_uint32 i = 0; i < work_dim; i++) {
      Values.push_back(global_work_offset[i] / Values[i]);
    }

    for (size_t i = 0; i < Values.size(); i++) {
      Entries.emplace_back(100 + i, sizeof(uint32_t) * i, sizeof(uint32_t));
    }

    vk::SpecializationInfo SpecializationInfo(
        static_cast<uint32_t>(Entries.size()), Entries.data(),
        Values.size() * sizeof(uint32_t), Values.data());
    vk::ComputePipelineCreateInfo computePipelineInfo(
        vk::PipelineCreateFlags(),
        vk::PipelineShaderStageCreateInfo(
            vk::PipelineShaderStageCreateFlags(),
            vk::ShaderStageFlagBits::eCompute, kernel->Program_->Module,
            kernel->Name.c_str(), &SpecializationInfo),
        Execution->PipelineLayout.get());

    {
      auto PipelineResult = Device->createComputePipelineUnique(
          kernel->PipelineCache.get(), computePipelineInfo);
      assert(PipelineResult.result == vk::Result::eSuccess);
      Execution->Pipeline = std::move(PipelineResult.value);
    }

    auto DescriptorPoolSize = vk::DescriptorPoolSize(
        vk::DescriptorType::eStorageBuffer, kernel->Arguments.size());
    Execution->DescriptorPool =
        Device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
            vk::DescriptorPoolCreateFlags(
                vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet),
            1, 1, &DescriptorPoolSize));

    Execution->DescriptorSet = std::move(
        Device
            ->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(
                Execution->DescriptorPool.get(), 1,
                &Execution->DescriptorSetLayout.get()))
            .front());

    std::vector<vk::WriteDescriptorSet> WriteSets{kernel->Arguments.size()};
    std::vector<std::unique_ptr<vk::DescriptorBufferInfo>>
        DescriptorBufferInfos{kernel->Arguments.size()};

    for (size_t i = 0; i < kernel->Arguments.size(); i++) {
      DescriptorBufferInfos[i] = std::make_unique<vk::DescriptorBufferInfo>(
          kernel->Arguments[i]->DeviceBuffer, 0, VK_WHOLE_SIZE);
      WriteSets[i] = vk::WriteDescriptorSet{
          Execution->DescriptorSet.get(),
          static_cast<uint32_t>(kernel->getArgumentIndex(i)),
          0,
          1,
          vk::DescriptorType::eStorageBuffer,
          nullptr,
          DescriptorBufferInfos[i].get(),
          nullptr};
      Execution->MemoryReferences.emplace_back(kernel->Arguments[i]);
    }

    Device->updateDescriptorSets(WriteSets, nullptr);

    Execution->CommandBuffer = Queue->createCommandBuffer();
    auto &CommandBuffer = Execution->CommandBuffer.get();

    CommandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlags(
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));

    Execution->QueryPool = enableProfiling(Queue, CommandBuffer);

    CommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute,
                               Execution->Pipeline.get());

    CommandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, Execution->PipelineLayout.get(), 0, 1,
        &Execution->DescriptorSet.get(), 0, nullptr);

    assert(global_work_size[0] % Values[0] == 0);
    assert(global_work_offset[0] % Values[0] == 0);
    if (work_dim >= 2) {
      assert(global_work_size[1] % Values[1] == 0);
      assert(global_work_offset[1] % Values[1] == 0);
    }
    if (work_dim >= 3) {
      assert(global_work_size[2] % Values[2] == 0);
      assert(global_work_offset[2] % Values[2] == 0);
    }

    CommandBuffer.dispatchBase(
        work_dim >= 1 ? global_work_offset[0] / Values[0] : 0,
        work_dim >= 2 ? global_work_offset[1] / Values[1] : 0,
        work_dim >= 3 ? global_work_offset[2] / Values[2] : 0,
        work_dim >= 1 ? global_work_size[0] / Values[0] : 1,
        work_dim >= 2 ? global_work_size[1] / Values[1] : 1,
        work_dim >= 3 ? global_work_size[2] / Values[2] : 1);

    writeFinishTimestamp(Queue, CommandBuffer, Execution->QueryPool.get());
    CommandBuffer.end();

    const uint64_t Counter = 1;
    vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
        SubmitInfo = {
            vk::SubmitInfo(0, nullptr, nullptr, 1, &CommandBuffer),
            vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 0, nullptr)};

    Execution->DependendEvents = kernel->AdditionalMemoryEvents;

    for (pi_uint32 i = 0; i < num_events_in_wait_list; i++) {
      kernel->AdditionalMemoryEvents.push_back(event_wait_list[i]);
      Execution->addEventDependency(event_wait_list[i]);
    }

    _pi_timeline_event::setWaitingSemaphores(
        kernel->AdditionalMemoryEvents.size(),
        kernel->AdditionalMemoryEvents.data(), SubmitInfo);

    // Clean up set arg events
    kernel->AdditionalMemoryEvents.clear();

    // Set Resulting Semaphore if event is aquired
    if (event) {
      Execution->Semaphore = _pi_semaphore::createNew(*Device);
      SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
          .setSignalSemaphoreValueCount(1);
      SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
          .setPSignalSemaphoreValues(&Counter);

      SubmitInfo.get<vk::SubmitInfo>().setSignalSemaphoreCount(1);
      SubmitInfo.get<vk::SubmitInfo>().setPSignalSemaphores(
          &Execution->Semaphore->Semaphore.get());
    }

    Execution->Fence =
        Device->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlags()));
    Result = Queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                                 Execution->Fence.get());
    Execution->Queue.reset(Queue);
    VLK(piQueueRetain)(Queue);

    if (event) {
      auto NewEvent = new _pi_timeline_event(Queue->Context_,
                                             std::move(Execution), Counter);
      // Store latest kernel launch, to permit overwriting of arguments
      // before finishing execution
      NewEvent->addKernel(kernel);
      *event = NewEvent;
    } else {
      Queue->StoredExecutions.push_back(std::move(Execution));
    }

  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  // stop the capture
  if (rdoc_api)
    rdoc_api->EndFrameCapture(NULL, NULL);

  return mapVulkanErrToCLErr(Result);
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

  pi_result ret = PI_SUCCESS;

  try {
    if (blocking_read) {
      memobj->copyDtoHblocking(num_events_in_wait_list, event_wait_list);
      void *BufferPtr =
          memobj->Context_->Device->mapMemory(memobj->HostMemory, offset, size);

      if (std::memcpy(ptr, BufferPtr, size) != ptr) {
        ret = PI_INVALID_MEM_OBJECT;
      }
      memobj->Context_->Device->unmapMemory(memobj->HostMemory);
    } else {
      _pi_execution::uptr_t Execution =
          memobj->copyDtoH(num_events_in_wait_list, event_wait_list);

      std::thread(localCopy, memobj, ptr, size, offset, Execution->Semaphore, 1)
          .detach();
      if (event)
        *event =
            new _pi_timeline_event(memobj->Context_, std::move(Execution), 2);
    }
  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  return ret;
}

pi_result VLK(piEnqueueMemBufferReadRect)(
    pi_queue command_queue, pi_mem memobj, pi_bool blocking_read,
    const size_t *buffer_offset, const size_t *host_offset,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, void *ptr,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {

  assert(region[0] > 0);
  assert(region[1] > 0);
  assert(region[2] > 0);

  pi_result ret = PI_SUCCESS;

  // Due to openCL specification
  // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html
  if (buffer_row_pitch == 0)
    buffer_row_pitch = region[0];
  if (buffer_slice_pitch == 0)
    buffer_slice_pitch = region[1] * buffer_row_pitch;
  if (host_row_pitch == 0)
    host_row_pitch = region[0];
  if (host_slice_pitch == 0)
    host_slice_pitch = region[1] * host_row_pitch;

  try {
    // FIXME: Add support for nonblocking
    // if (blocking_read) {
    // FIXME: Copy only needed things from Device
    memobj->copyDtoHblocking(num_events_in_wait_list, event_wait_list);
    char *HostPtr = reinterpret_cast<char *>(ptr);
    char *BufferPtr =
        reinterpret_cast<char *>(memobj->Context_->Device->mapMemory(
            memobj->HostMemory, 0, VK_WHOLE_SIZE));
    for (size_t Slice = 0; Slice < region[2]; Slice++) {
      for (size_t Row = 0; Row < region[1]; Row++) {
        if (std::memcpy(
                HostPtr + (host_offset[2] + Slice) * host_slice_pitch +
                    (host_offset[1] + Row) * host_row_pitch + host_offset[0],
                BufferPtr + (buffer_offset[2] + Slice) * buffer_slice_pitch +
                    (buffer_offset[1] + Row) * buffer_row_pitch +
                    buffer_offset[0],
                region[0]) !=
            HostPtr + (host_offset[2] + Slice) * host_slice_pitch +
                (host_offset[1] + Row) * host_row_pitch + host_offset[0]) {
          ret = PI_INVALID_MEM_OBJECT;
        }
      }
    }
    memobj->Context_->Device->unmapMemory(memobj->HostMemory);
    if (event)
      *event = new _pi_empty_event();
    /*} else {
      memobj->copyDtoH();
      std::thread(localCopy, memobj, ptr, size, offset,
                  memobj->Context_->lastTimelineValue)
          .detach();
      memobj->Context_->lastTimelineValue++;*/
    /*if (event)
      *event = new _pi_timeline_event(memobj->Context_,
                                      memobj->Context_->Timeline.get(),
                                      memobj->Context_->lastTimelineValue);*/
    //}
  } catch (vk::SystemError const &Err) {
    return mapVulkanErrToCLErr(Err);
  }

  return ret;
}

pi_result VLK(piEnqueueMemBufferWrite)(pi_queue command_queue, pi_mem memobj,
                                       pi_bool blocking_write, size_t offset,
                                       size_t size, const void *ptr,
                                       pi_uint32 num_events_in_wait_list,
                                       const pi_event *event_wait_list,
                                       pi_event *event) {

  auto Execution = std::make_unique<_pi_execution>(command_queue->Context_);

  Execution->CommandBuffer = command_queue->createCommandBuffer();

  Execution->CommandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  Execution->QueryPool =
      enableProfiling(command_queue, Execution->CommandBuffer.get());
  Execution->CommandBuffer->updateBuffer(memobj->DeviceBuffer, offset, size,
                                         ptr);
  writeFinishTimestamp(command_queue, Execution->CommandBuffer.get(),
                       Execution->QueryPool.get());
  Execution->CommandBuffer->end();
  Execution->MemoryReferences.emplace_back(memobj);

  Execution->addAllEventsDependencies(num_events_in_wait_list, event_wait_list);

  Execution->Semaphore =
      _pi_semaphore::createNew(*command_queue->Context_->Device);
  const uint64_t Counter = 1;

  // This Execution must be released before the pi_queue
  Execution->Queue.reset(command_queue);
  VLK(piQueueRetain)(command_queue);

  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
      SubmitInfo = {
          vk::SubmitInfo(0, nullptr, nullptr, 1,
                         &Execution->CommandBuffer.get(), 1,
                         &Execution->Semaphore->Semaphore.get()),
          vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 1, &Counter)};

  _pi_timeline_event::setWaitingSemaphores(num_events_in_wait_list,
                                           event_wait_list, SubmitInfo);

  Execution->Fence = command_queue->Context_->Device->createFenceUnique(
      vk::FenceCreateInfo(vk::FenceCreateFlags()));
  command_queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                              Execution->Fence.get());
  auto Event = std::make_unique<_pi_timeline_event>(
      memobj->Context_, std::move(Execution), Counter);
  memobj->DeviceDirty = true;
  if (blocking_write) {
    Event->wait();
  }
  if (event) {
    *event = Event.release();
  } else {
    command_queue->StoredExecutions.push_back(std::move(Event->Execution));
  }
  return PI_SUCCESS;
}

pi_result VLK(piEnqueueMemBufferWriteRect)(
    pi_queue command_queue, pi_mem Buffer, pi_bool blocking_write,
    const size_t *buffer_origin, const size_t *host_origin,
    const size_t *region, size_t buffer_row_pitch, size_t buffer_slice_pitch,
    size_t host_row_pitch, size_t host_slice_pitch, const void *ptr,
    pi_uint32 num_events_in_wait_list, const pi_event *event_wait_list,
    pi_event *event) {

  assert(region[0] > 0);
  assert(region[1] > 0);
  assert(region[2] > 0);

  // Due to openCL specification
  // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html
  if (host_row_pitch == 0)
    host_row_pitch = region[0];
  if (host_slice_pitch == 0)
    host_slice_pitch = region[1] * host_row_pitch;
  if (buffer_row_pitch == 0)
    buffer_row_pitch = region[0];
  if (buffer_slice_pitch == 0)
    buffer_slice_pitch = region[1] * buffer_row_pitch;

  auto Execution = std::make_unique<_pi_execution>(command_queue->Context_);
  Execution->CommandBuffer = command_queue->createCommandBuffer();

  Execution->CommandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  Execution->QueryPool =
      enableProfiling(command_queue, Execution->CommandBuffer.get());
  for (size_t Slice = 0; Slice < region[2]; Slice++) {
    for (size_t Row = 0; Row < region[1]; Row++) {
      const void *source = reinterpret_cast<const char *>(ptr) +
                           (host_origin[2] + Slice) * host_slice_pitch +
                           (host_origin[1] + Row) * host_row_pitch +
                           host_origin[0];
      Execution->CommandBuffer->updateBuffer(
          Buffer->DeviceBuffer,
          (buffer_origin[2] + Slice) * buffer_slice_pitch +
              (buffer_origin[1] + Row) * buffer_row_pitch + buffer_origin[0],
          region[0], source);
    }
  }
  writeFinishTimestamp(command_queue, Execution->CommandBuffer.get(),
                       Execution->QueryPool.get());
  Execution->CommandBuffer->end();
  Execution->MemoryReferences.emplace_back(Buffer);

  Execution->addAllEventsDependencies(num_events_in_wait_list, event_wait_list);

  Execution->Semaphore =
      _pi_semaphore::createNew(*command_queue->Context_->Device);
  const uint64_t Counter = 1;

  // This Execution must be released before the pi_queue
  Execution->Queue.reset(command_queue);
  VLK(piQueueRetain)(command_queue);

  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
      SubmitInfo = {
          vk::SubmitInfo(0, nullptr, nullptr, 1,
                         &Execution->CommandBuffer.get(), 1,
                         &Execution->Semaphore->Semaphore.get()),
          vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 1, &Counter)};

  _pi_timeline_event::setWaitingSemaphores(num_events_in_wait_list,
                                           event_wait_list, SubmitInfo);

  Execution->Fence = command_queue->Context_->Device->createFenceUnique(
      vk::FenceCreateInfo(vk::FenceCreateFlags()));
  command_queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                              Execution->Fence.get());
  auto Event = std::make_unique<_pi_timeline_event>(
      command_queue->Context_, std::move(Execution), Counter);
  Buffer->DeviceDirty = true;
  if (blocking_write) {
    Event->wait();
  }
  if (event) {
    *event = Event.release();
  } else {
    command_queue->StoredExecutions.push_back(std::move(Event->Execution));
  }
  return PI_SUCCESS;
}

pi_result VLK(piEnqueueMemBufferCopy)(pi_queue command_queue, pi_mem src_buffer,
                                      pi_mem dst_buffer, size_t src_offset,
                                      size_t dst_offset, size_t size,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {

  std::array<vk::BufferCopy, 1> Range = {
      vk::BufferCopy{src_offset, dst_offset, size}};

  auto Execution = std::make_unique<_pi_execution>(command_queue->Context_);
  Execution->CommandBuffer = command_queue->createCommandBuffer();

  Execution->CommandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  Execution->QueryPool =
      enableProfiling(command_queue, Execution->CommandBuffer.get());
  Execution->CommandBuffer->copyBuffer(src_buffer->DeviceBuffer,
                                       dst_buffer->DeviceBuffer, Range);
  writeFinishTimestamp(command_queue, Execution->CommandBuffer.get(),
                       Execution->QueryPool.get());
  Execution->CommandBuffer->end();
  Execution->MemoryReferences.emplace_back(dst_buffer);
  Execution->MemoryReferences.emplace_back(src_buffer);

  // This Execution must be released before the pi_queue
  Execution->Queue.reset(command_queue);
  VLK(piQueueRetain)(command_queue);

  //_pi_semaphore::sptr_t Semaphore(nullptr);
  const uint64_t Counter = 1;

  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
      SubmitInfo = {vk::SubmitInfo(0, nullptr, nullptr, 1,
                                   &Execution->CommandBuffer.get()),
                    vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 0, nullptr)};

  _pi_timeline_event::setWaitingSemaphores(num_events_in_wait_list,
                                           event_wait_list, SubmitInfo);

  if (event) {
    Execution->Semaphore =
        _pi_semaphore::createNew(*command_queue->Context_->Device);
    SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setSignalSemaphoreValueCount(1);
    SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setPSignalSemaphoreValues(&Counter);

    SubmitInfo.get<vk::SubmitInfo>().setSignalSemaphoreCount(1);
    SubmitInfo.get<vk::SubmitInfo>().setPSignalSemaphores(
        &Execution->Semaphore->Semaphore.get());
  }
  Execution->addAllEventsDependencies(num_events_in_wait_list, event_wait_list);

  Execution->Fence = command_queue->Context_->Device->createFenceUnique(
      vk::FenceCreateInfo(vk::FenceCreateFlags()));
  command_queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                              Execution->Fence.get());
  auto Event = std::make_unique<_pi_timeline_event>(
      command_queue->Context_, std::move(Execution), Counter);
  dst_buffer->DeviceDirty = true;
  if (event) {
    *event = Event.release();
  } else {
    command_queue->StoredExecutions.push_back(std::move(Event->Execution));
  }
  return PI_SUCCESS;
}

pi_result VLK(piEnqueueMemBufferCopyRect)(
    pi_queue command_queue, pi_mem src_buffer, pi_mem dst_buffer,
    const size_t *src_origin, const size_t *dst_origin, const size_t *region,
    size_t src_row_pitch, size_t src_slice_pitch, size_t dst_row_pitch,
    size_t dst_slice_pitch, pi_uint32 num_events_in_wait_list,
    const pi_event *event_wait_list, pi_event *event) {

  // Due to openCL specification
  // https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueCopyBufferRect.html
  if (src_row_pitch == 0)
    src_row_pitch = region[0];
  if (src_slice_pitch == 0)
    src_slice_pitch = region[1] * src_row_pitch;
  if (dst_row_pitch == 0)
    dst_row_pitch = region[0];
  if (dst_slice_pitch == 0)
    dst_slice_pitch = region[1] * dst_row_pitch;

  std::vector<vk::BufferCopy> Range;

  for (size_t Slice = 0; Slice < region[2]; Slice++) {
    for (size_t Row = 0; Row < region[1]; Row++) {
      Range.emplace_back(
          (src_origin[2] + Slice) * src_slice_pitch +
              (src_origin[1] + Row) * src_row_pitch + src_origin[0],
          (dst_origin[2] + Slice) * dst_slice_pitch +
              (dst_origin[1] + Row) * dst_row_pitch + dst_origin[0],
          region[0]);
    }
  }

  auto Execution = std::make_unique<_pi_execution>(command_queue->Context_);
  Execution->CommandBuffer = command_queue->createCommandBuffer();

  Execution->CommandBuffer->begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eSimultaneousUse));
  Execution->QueryPool =
      enableProfiling(command_queue, Execution->CommandBuffer.get());
  Execution->CommandBuffer->copyBuffer(src_buffer->DeviceBuffer,
                                       dst_buffer->DeviceBuffer, Range);
  writeFinishTimestamp(command_queue, Execution->CommandBuffer.get(),
                       Execution->QueryPool.get());
  Execution->CommandBuffer->end();
  Execution->MemoryReferences.emplace_back(dst_buffer);
  Execution->MemoryReferences.emplace_back(src_buffer);

  // This Execution must be released before the pi_queue
  Execution->Queue.reset(command_queue);
  VLK(piQueueRetain)(command_queue);

  //_pi_semaphore::sptr_t Semaphore(nullptr);
  const uint64_t Counter = 1;

  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
      SubmitInfo = {vk::SubmitInfo(0, nullptr, nullptr, 1,
                                   &Execution->CommandBuffer.get()),
                    vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 0, nullptr)};

  _pi_timeline_event::setWaitingSemaphores(num_events_in_wait_list,
                                           event_wait_list, SubmitInfo);

  // Set Resulting Semaphore for if event is aquired
  if (event) {
    Execution->Semaphore =
        _pi_semaphore::createNew(*command_queue->Context_->Device);
    SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setSignalSemaphoreValueCount(1);
    SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
        .setPSignalSemaphoreValues(&Counter);

    SubmitInfo.get<vk::SubmitInfo>().setSignalSemaphoreCount(1);
    SubmitInfo.get<vk::SubmitInfo>().setPSignalSemaphores(
        &Execution->Semaphore->Semaphore.get());
  }
  Execution->addAllEventsDependencies(num_events_in_wait_list, event_wait_list);

  Execution->Fence = command_queue->Context_->Device->createFenceUnique(
      vk::FenceCreateInfo(vk::FenceCreateFlags()));
  command_queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                              Execution->Fence.get());

  auto Event = std::make_unique<_pi_timeline_event>(
      command_queue->Context_, std::move(Execution), Counter);
  if (event) {
    *event = Event.release();
  } else {
    command_queue->StoredExecutions.push_back(std::move(Event->Execution));
  }
  return PI_SUCCESS;
}

pi_result VLK(piEnqueueMemBufferFill)(pi_queue command_queue, pi_mem buffer,
                                      const void *pattern, size_t pattern_size,
                                      size_t offset, size_t size,
                                      pi_uint32 num_events_in_wait_list,
                                      const pi_event *event_wait_list,
                                      pi_event *event) {
  // size must be a multiple of 4
  // dstOffset must be a multiple of 4
  assert(offset % 4 == 0);
  assert(size % 4 == 0);
  // data is the 4-byte word written repeatedly to the buffer to fill size
  // bytes of data.

  // Vulkan only supports direct buffer fill for a 4 Byte Size element
  // use native function if patternsize is 4 Byte
  if (pattern_size == 4) {
    auto Execution = std::make_unique<_pi_execution>(command_queue->Context_);
    Execution->CommandBuffer = command_queue->createCommandBuffer();

    Execution->CommandBuffer->begin(vk::CommandBufferBeginInfo(
        vk::CommandBufferUsageFlagBits::eSimultaneousUse));
    Execution->QueryPool =
        enableProfiling(command_queue, Execution->CommandBuffer.get());
    Execution->CommandBuffer->fillBuffer(
        buffer->DeviceBuffer, offset, size,
        *reinterpret_cast<const uint32_t *>(pattern));
    writeFinishTimestamp(command_queue, Execution->CommandBuffer.get(),
                         Execution->QueryPool.get());
    Execution->CommandBuffer->end();
    Execution->MemoryReferences.emplace_back(buffer);

    // This Execution must be released before the pi_queue
    Execution->Queue.reset(command_queue);
    VLK(piQueueRetain)(command_queue);

    //_pi_semaphore::sptr_t Semaphore(nullptr);
    const uint64_t Counter = 1;

    vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfoKHR>
        SubmitInfo = {
            vk::SubmitInfo(0, nullptr, nullptr, 1,
                           &Execution->CommandBuffer.get()),
            vk::TimelineSemaphoreSubmitInfoKHR(0, nullptr, 0, nullptr)};

    _pi_timeline_event::setWaitingSemaphores(num_events_in_wait_list,
                                             event_wait_list, SubmitInfo);

    // Set Resulting Semaphore for if event is aquired
    if (event) {
      Execution->Semaphore =
          _pi_semaphore::createNew(*command_queue->Context_->Device);
      SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
          .setSignalSemaphoreValueCount(1);
      SubmitInfo.get<vk::TimelineSemaphoreSubmitInfoKHR>()
          .setPSignalSemaphoreValues(&Counter);

      SubmitInfo.get<vk::SubmitInfo>().setSignalSemaphoreCount(1);
      SubmitInfo.get<vk::SubmitInfo>().setPSignalSemaphores(
          &Execution->Semaphore->Semaphore.get());
    }
    Execution->addAllEventsDependencies(num_events_in_wait_list,
                                        event_wait_list);

    Execution->Fence = command_queue->Context_->Device->createFenceUnique(
        vk::FenceCreateInfo(vk::FenceCreateFlags()));
    command_queue->Queue.submit(1, &SubmitInfo.get<vk::SubmitInfo>(),
                                Execution->Fence.get());

    auto Event = std::make_unique<_pi_timeline_event>(
        command_queue->Context_, std::move(Execution), Counter);
    if (event) {
      *event = Event.release();
    } else {
      command_queue->StoredExecutions.push_back(std::move(Event->Execution));
    }

    return PI_SUCCESS;
  } else {
    assert(size % pattern_size == 0);

    // For arbitrary pattern sizes map memory
    // and do manual buffer fill using memcpy
    // and move to device through unmap
    // TODO: Think about offset and size - for now it is complete overwrite
    char *MapPtr = nullptr;
    pi_result Result = VLK(piEnqueueMemBufferMap)(
        command_queue, buffer, false, CL_MAP_WRITE_INVALIDATE_REGION, offset,
        size, num_events_in_wait_list, event_wait_list, nullptr,
        (reinterpret_cast<void **>(&MapPtr)));
    if (Result != PI_SUCCESS) {
      return Result;
    }
    for (uint32_t i = 0; i < size; i += pattern_size) {
      memcpy(MapPtr + i, pattern, pattern_size);
    }
    return VLK(piEnqueueMemUnmap)(command_queue, buffer, MapPtr, 0, nullptr,
                                  event);
  }
}

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

  auto TransferBackNeeded = false;
  // Only transfer data back from device if there are dependencies
  // (event dependency or blocking write) otherwise
  // there should not have been any changes to the buffer on device
  if ((num_events_in_wait_list > 0 || memobj->DeviceDirty) &&
      ((map_flags & CL_MAP_READ) || (map_flags & CL_MAP_WRITE))) {
    TransferBackNeeded = true;
  }

  memobj->LastMapFlags = map_flags;
  memobj->LastMapBlocking = blocking_map;
  if (memobj->HostPtr) {
    *ret_map = memobj->HostPtr;
    if (TransferBackNeeded) {
      return VLK(piEnqueueMemBufferRead)(
          command_queue, memobj, blocking_map, offset, size, memobj->HostPtr,
          num_events_in_wait_list, event_wait_list, event);
    }
    if (event)
      *event = new _pi_empty_event();
    return PI_SUCCESS;
  } else {
    if (TransferBackNeeded) {
      if (blocking_map) {
        memobj->copyDtoHblocking(num_events_in_wait_list, event_wait_list);
      } else {
        _pi_execution::uptr_t Execution =
            memobj->copyDtoH(num_events_in_wait_list, event_wait_list);
        auto Event = std::make_unique<_pi_timeline_event>(
            command_queue->Context_, std::move(Execution), 1);
        if (event)
          *event = Event.release();
      }
    }
    vk::Result Error = memobj->Context_->Device->mapMemory(
        memobj->HostMemory, offset, size, vk::MemoryMapFlags(), ret_map);
    return mapVulkanErrToCLErr(Error);
  }
}

pi_result VLK(piEnqueueMemUnmap)(pi_queue command_queue, pi_mem memobj,
                                 void *mapped_ptr,
                                 pi_uint32 num_events_in_wait_list,
                                 const pi_event *event_wait_list,
                                 pi_event *event) {

  // Only unmap the pointer if its a fake host pointer
  if (!memobj->HostPtr)
    memobj->Context_->Device->unmapMemory(memobj->HostMemory);

  auto LastMapFlags = memobj->LastMapFlags;
  auto LastMapBlocking = memobj->LastMapBlocking;
  memobj->LastMapFlags = 0ul;
  memobj->LastMapBlocking = false;

  if (LastMapFlags & CL_MAP_WRITE ||
      LastMapFlags & CL_MAP_WRITE_INVALIDATE_REGION) {

    if (memobj->HostPtr) {
      // If it is a "fake" host pointer, do writing into device buffer directly
      // alternative would be FakeHostPtr -> HostMem -> DeviceMem
      // but this saves the additional copy
      // TODO: Think about offset and size (maybe remember from MAP)
      return VLK(piEnqueueMemBufferWrite)(
          command_queue, memobj, LastMapBlocking, 0, memobj->TotalMemorySize,
          mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    }
    // In this case the memory is already written in Host Memory
    // simple copy to device
    if (LastMapBlocking) {
      memobj->copyHtoDblocking(num_events_in_wait_list, event_wait_list);
    } else {
      _pi_execution::uptr_t Execution =
          memobj->copyHtoD(num_events_in_wait_list, event_wait_list);

      auto Event = std::make_unique<_pi_timeline_event>(
          command_queue->Context_, std::move(Execution), 1);
      if (event)
        *event = Event.release();
    }
  } else {

    if (event)
      *event = new _pi_empty_event();
  }

  return PI_SUCCESS;
}

pi_result VLK(piextKernelSetArgMemObj)(pi_kernel kernel, pi_uint32 arg_index,
                                       const pi_mem *arg_value) {
  return kernel->addArgument(arg_index, arg_value);
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

__SYCL_EXPORT pi_result piPluginInit(pi_plugin *PluginInit) {
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
  _PI_CL(piProgramCreateWithBinary, VLK(piProgramCreateWithBinary))
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
