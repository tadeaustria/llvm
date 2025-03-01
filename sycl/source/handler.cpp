//==-------- handler.cpp --- SYCL command group handler --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/kernel_desc.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/info/info_desc.hpp>
#include <detail/kernel_impl.hpp>
#include <detail/queue_impl.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

event handler::finalize() {
  // This block of code is needed only for reduction implementation.
  // It is harmless (does nothing) for everything else.
  if (MIsFinalized)
    return MLastEvent;
  MIsFinalized = true;

  unique_ptr_class<detail::CG> CommandGroup;
  switch (MCGType) {
  case detail::CG::KERNEL:
  case detail::CG::RUN_ON_HOST_INTEL: {
    CommandGroup.reset(new detail::CGExecKernel(
        std::move(MNDRDesc), std::move(MHostKernel), std::move(MKernel),
        std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), std::move(MArgs), std::move(MKernelName),
        std::move(MOSModuleHandle), std::move(MStreamStorage), MCGType,
        MCodeLoc));
    break;
  }
  case detail::CG::CODEPLAY_INTEROP_TASK:
    CommandGroup.reset(new detail::CGInteropTask(
        std::move(MInteropTask), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::COPY_ACC_TO_PTR:
  case detail::CG::COPY_PTR_TO_ACC:
  case detail::CG::COPY_ACC_TO_ACC:
    CommandGroup.reset(new detail::CGCopy(
        MCGType, MSrcPtr, MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::FILL:
    CommandGroup.reset(new detail::CGFill(
        std::move(MPattern), MDstPtr, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::UPDATE_HOST:
    CommandGroup.reset(new detail::CGUpdateHost(
        MDstPtr, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::COPY_USM:
    CommandGroup.reset(new detail::CGCopyUSM(
        MSrcPtr, MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::FILL_USM:
    CommandGroup.reset(new detail::CGFillUSM(
        std::move(MPattern), MDstPtr, MLength, std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::PREFETCH_USM:
    CommandGroup.reset(new detail::CGPrefetchUSM(
        MDstPtr, MLength, std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCodeLoc));
    break;
  case detail::CG::CODEPLAY_HOST_TASK:
    CommandGroup.reset(new detail::CGHostTask(
        std::move(MHostTask), MQueue, MQueue->getContextImplPtr(),
        std::move(MArgs), std::move(MArgsStorage), std::move(MAccStorage),
        std::move(MSharedPtrStorage), std::move(MRequirements),
        std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::BARRIER:
  case detail::CG::BARRIER_WAITLIST:
    CommandGroup.reset(new detail::CGBarrier(
        std::move(MEventsWaitWithBarrier), std::move(MArgsStorage),
        std::move(MAccStorage), std::move(MSharedPtrStorage),
        std::move(MRequirements), std::move(MEvents), MCGType, MCodeLoc));
    break;
  case detail::CG::NONE:
    throw runtime_error("Command group submitted without a kernel or a "
                        "explicit memory operation.",
                        PI_INVALID_OPERATION);
  default:
    throw runtime_error("Unhandled type of command group",
                        PI_INVALID_OPERATION);
  }

  detail::EventImplPtr Event = detail::Scheduler::getInstance().addCG(
      std::move(CommandGroup), std::move(MQueue));

  MLastEvent = detail::createSyclObjFromImpl<event>(Event);
  return MLastEvent;
}

void handler::associateWithHandler(detail::AccessorBaseHost *AccBase,
                                   access::target AccTarget) {
  detail::AccessorImplPtr AccImpl = detail::getSyclObjImpl(*AccBase);
  detail::Requirement *Req = AccImpl.get();
  // Add accessor to the list of requirements.
  MRequirements.push_back(Req);
  // Store copy of the accessor.
  MAccStorage.push_back(std::move(AccImpl));
  // Add an accessor to the handler list of associated accessors.
  // For associated accessors index does not means nothing.
  MAssociatedAccesors.emplace_back(detail::kernel_param_kind_t::kind_accessor,
                                   Req, static_cast<int>(AccTarget),
                                   /*index*/ 0);
}

// TODO remove this one once ABI breaking changes are allowed.
void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource) {
  processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource,
             false);
}

void handler::processArg(void *Ptr, const detail::kernel_param_kind_t &Kind,
                         const int Size, const size_t Index, size_t &IndexShift,
                         bool IsKernelCreatedFromSource, bool IsESIMD) {
  using detail::kernel_param_kind_t;

  switch (Kind) {
  case kernel_param_kind_t::kind_std_layout:
  case kernel_param_kind_t::kind_pointer: {
    MArgs.emplace_back(Kind, Ptr, Size, Index + IndexShift);
    break;
  }
  case kernel_param_kind_t::kind_accessor: {
    // For args kind of accessor Size is information about accessor.
    // The first 11 bits of Size encodes the accessor target.
    const access::target AccTarget = static_cast<access::target>(Size & 0x7ff);
    switch (AccTarget) {
    case access::target::global_buffer:
    case access::target::constant_buffer: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);

      // Stream implementation creates an accessor with initial size for
      // work item. Number of work items is not available during
      // stream construction, that is why size of the accessor is updated here
      // using information about number of work items.
      if (AccImpl->PerWI) {
        AccImpl->resize(MNDRDesc.GlobalSize.size());
      }
      MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);

      // TODO ESIMD currently does not suport offset, memory and access ranges -
      // accessor::init for ESIMD-mode accessor has a single field, translated
      // to a single kernel argument set above.
      if (!IsKernelCreatedFromSource && !IsESIMD) {
        // Dimensionality of the buffer is 1 when dimensionality of the
        // accessor is 0.
        const size_t SizeAccField =
            sizeof(size_t) * (AccImpl->MDims == 0 ? 1 : AccImpl->MDims);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout,
                           &AccImpl->MAccessRange[0], SizeAccField,
                           Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout,
                           &AccImpl->MMemoryRange[0], SizeAccField,
                           Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout,
                           &AccImpl->MOffset[0], SizeAccField,
                           Index + IndexShift);
      }
      break;
    }
    case access::target::local: {
      detail::LocalAccessorImplHost *LAcc =
          static_cast<detail::LocalAccessorImplHost *>(Ptr);

      range<3> &Size = LAcc->MSize;
      const int Dims = LAcc->MDims;
      int SizeInBytes = LAcc->MElemSize;
      for (int I = 0; I < Dims; ++I)
        SizeInBytes *= Size[I];
      MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, nullptr,
                         SizeInBytes, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        ++IndexShift;
        const size_t SizeAccField = Dims * sizeof(Size[0]);
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
        ++IndexShift;
        MArgs.emplace_back(kernel_param_kind_t::kind_std_layout, &Size,
                           SizeAccField, Index + IndexShift);
      }
      break;
    }
    case access::target::image:
    case access::target::image_array: {
      detail::Requirement *AccImpl = static_cast<detail::Requirement *>(Ptr);
      MArgs.emplace_back(Kind, AccImpl, Size, Index + IndexShift);
      if (!IsKernelCreatedFromSource) {
        // TODO Handle additional kernel arguments for image class
        // if the compiler front-end adds them.
      }
      break;
    }
    case access::target::host_image:
    case access::target::host_buffer: {
      throw cl::sycl::invalid_parameter_error(
          "Unsupported accessor target case.", PI_INVALID_OPERATION);
      break;
    }
    }
    break;
  }
  case kernel_param_kind_t::kind_sampler: {
    MArgs.emplace_back(kernel_param_kind_t::kind_sampler, Ptr, sizeof(sampler),
                       Index + IndexShift);
    break;
  }
  }
}

void handler::extractArgsAndReqs() {
  assert(MKernel && "MKernel is not initialized");
  std::vector<detail::ArgDesc> UnPreparedArgs = std::move(MArgs);
  MArgs.clear();

  std::sort(
      UnPreparedArgs.begin(), UnPreparedArgs.end(),
      [](const detail::ArgDesc &first, const detail::ArgDesc &second) -> bool {
        return (first.MIndex < second.MIndex);
      });

  const bool IsKernelCreatedFromSource = MKernel->isCreatedFromSource();

  size_t IndexShift = 0;
  for (size_t I = 0; I < UnPreparedArgs.size(); ++I) {
    void *Ptr = UnPreparedArgs[I].MPtr;
    const detail::kernel_param_kind_t &Kind = UnPreparedArgs[I].MType;
    const int &Size = UnPreparedArgs[I].MSize;
    const int Index = UnPreparedArgs[I].MIndex;
    processArg(Ptr, Kind, Size, Index, IndexShift, IsKernelCreatedFromSource,
               false);
  }
}

// TODO remove once ABI breaking changes are allowed
void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs, size_t IndexShift) {
  extractArgsAndReqsFromLambda(LambdaPtr, KernelArgsNum, KernelArgs, false, IndexShift);
}

void handler::extractArgsAndReqsFromLambda(
    char *LambdaPtr, size_t KernelArgsNum,
    const detail::kernel_param_desc_t *KernelArgs, bool IsESIMD, size_t IndexShift) {
  const bool IsKernelCreatedFromSource = false;
  //size_t IndexShift = 0;
  for (size_t I = 0; I < KernelArgsNum; ++I) {
    void *Ptr = LambdaPtr + KernelArgs[I].offset;
    const detail::kernel_param_kind_t &Kind = KernelArgs[I].kind;
    const int &Size = KernelArgs[I].info;
    if (Kind == detail::kernel_param_kind_t::kind_accessor) {
      // For args kind of accessor Size is information about accessor.
      // The first 11 bits of Size encodes the accessor target.
      const access::target AccTarget =
          static_cast<access::target>(Size & 0x7ff);
      if ((AccTarget == access::target::global_buffer ||
           AccTarget == access::target::constant_buffer) ||
          (AccTarget == access::target::image ||
           AccTarget == access::target::image_array)) {
        detail::AccessorBaseHost *AccBase =
            static_cast<detail::AccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*AccBase).get();
      } else if (AccTarget == access::target::local) {
        detail::LocalAccessorBaseHost *LocalAccBase =
            static_cast<detail::LocalAccessorBaseHost *>(Ptr);
        Ptr = detail::getSyclObjImpl(*LocalAccBase).get();
      }
    }
    processArg(Ptr, Kind, Size, I, IndexShift, IsKernelCreatedFromSource,
               IsESIMD);
  }
}

// Calling methods of kernel_impl requires knowledge of class layout.
// As this is impossible in header, there's a function that calls necessary
// method inside the library and returns the result.
string_class handler::getKernelName() {
  return MKernel->get_info<info::kernel::function_name>();
}

void handler::barrier(const vector_class<event> &WaitList) {
  throwIfActionIsCreated();
  MCGType = detail::CG::BARRIER_WAITLIST;
  MEventsWaitWithBarrier.resize(WaitList.size());
  std::transform(
      WaitList.begin(), WaitList.end(), MEventsWaitWithBarrier.begin(),
      [](const event &Event) { return detail::getSyclObjImpl(Event); });
}

void handler::memcpy(void *Dest, const void *Src, size_t Count) {
  throwIfActionIsCreated();
  MSrcPtr = const_cast<void *>(Src);
  MDstPtr = Dest;
  MLength = Count;
  MCGType = detail::CG::COPY_USM;
}

void handler::memset(void *Dest, int Value, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = Dest;
  MPattern.push_back(static_cast<char>(Value));
  MLength = Count;
  MCGType = detail::CG::FILL_USM;
}

void handler::prefetch(const void *Ptr, size_t Count) {
  throwIfActionIsCreated();
  MDstPtr = const_cast<void *>(Ptr);
  MLength = Count;
  MCGType = detail::CG::PREFETCH_USM;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
