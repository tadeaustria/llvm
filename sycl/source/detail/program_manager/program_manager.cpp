//==------ program_manager.cpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/ONEAPI/experimental/spec_constant.hpp>
#include <CL/sycl/backend_types.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/config.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/program_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <detail/spec_constant_impl.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using ContextImplPtr = std::shared_ptr<cl::sycl::detail::context_impl>;

static constexpr int DbgProgMgr = 0;

enum BuildState { BS_InProgress, BS_Done, BS_Failed };

static constexpr char UseSpvEnv[]("SYCL_USE_KERNEL_SPV");

ProgramManager &ProgramManager::getInstance() {
  return GlobalHandler::instance().getProgramManager();
}

static RT::PiProgram createBinaryProgram(const ContextImplPtr Context,
                                         const device &Device,
                                         const unsigned char *Data,
                                         size_t DataLen) {
  const detail::plugin &Plugin = Context->getPlugin();
#ifndef _NDEBUG
  pi_uint32 NumDevices = 0;
  Plugin.call<PiApiKind::piContextGetInfo>(Context->getHandleRef(),
                                           PI_CONTEXT_INFO_NUM_DEVICES,
                                           sizeof(NumDevices), &NumDevices,
                                           /*param_value_size_ret=*/nullptr);
  assert(NumDevices > 0 &&
         "Only a single device is supported for AOT compilation");
#endif

  RT::PiProgram Program;
  const RT::PiDevice PiDevice = getSyclObjImpl(Device)->getHandleRef();
  pi_int32 BinaryStatus = CL_SUCCESS;
  Plugin.call<PiApiKind::piProgramCreateWithBinary>(
      Context->getHandleRef(), 1 /*one binary*/, &PiDevice, &DataLen, &Data,
      &BinaryStatus, &Program);

  if (BinaryStatus != CL_SUCCESS) {
    throw runtime_error("Creating program with binary failed.", BinaryStatus);
  }

  return Program;
}

static RT::PiProgram createSpirvProgram(const ContextImplPtr Context,
                                        const unsigned char *Data,
                                        size_t DataLen) {
  RT::PiProgram Program = nullptr;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piProgramCreate>(Context->getHandleRef(), Data,
                                          DataLen, &Program);
  return Program;
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(OSModuleHandle M, const string_class &KernelName,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \""
              << KernelName << "\", " << getRawSyclObjImpl(Context) << ", "
              << getRawSyclObjImpl(Device) << ", " << JITCompilationIsRequired
              << ")\n";

  KernelSetId KSId = getKernelSetId(M, KernelName);
  return getDeviceImage(M, KSId, Context, Device, JITCompilationIsRequired);
}

template <typename ExceptionT, typename RetT>
RetT *waitUntilBuilt(KernelProgramCache &Cache,
                     KernelProgramCache::BuildResult<RetT> *BuildResult) {
  // any thread which will find nullptr in cache will wait until the pointer
  // is not null anymore
  Cache.waitUntilBuilt(*BuildResult, [BuildResult]() {
    int State = BuildResult->State.load();

    return State == BS_Done || State == BS_Failed;
  });

  if (BuildResult->Error.isFilledIn()) {
    const KernelProgramCache::BuildError &Error = BuildResult->Error;
    throw ExceptionT(Error.Msg, Error.Code);
  }

  RetT *Result = BuildResult->Ptr.load();

  return Result;
}

/// Try to fetch entity (kernel or program) from cache. If there is no such
/// entity try to build it. Throw any exception build process may throw.
/// This method eliminates unwanted builds by employing atomic variable with
/// build state and waiting until the entity is built in another thread.
/// If the building thread has failed the awaiting thread will fail either.
/// Exception thrown by build procedure are rethrown.
///
/// \tparam RetT type of entity to get
/// \tparam ExceptionT type of exception to throw on awaiting thread if the
///         building thread fails build step.
/// \tparam KeyT key (in cache) to fetch built entity with
/// \tparam AcquireFT type of function which will acquire the locked version of
///         the cache. Accept reference to KernelProgramCache.
/// \tparam GetCacheFT type of function which will fetch proper cache from
///         locked version. Accepts reference to locked version of cache.
/// \tparam BuildFT type of function which will build the entity if it is not in
///         cache. Accepts nothing. Return pointer to built entity.
template <typename RetT, typename ExceptionT, typename KeyT, typename AcquireFT,
          typename GetCacheFT, typename BuildFT>
KernelProgramCache::BuildResult<RetT> *
getOrBuild(KernelProgramCache &KPCache, KeyT &&CacheKey, AcquireFT &&Acquire,
           GetCacheFT &&GetCache, BuildFT &&Build) {
  bool InsertionTookPlace;
  KernelProgramCache::BuildResult<RetT> *BuildResult;

  {
    auto LockedCache = Acquire(KPCache);
    auto &Cache = GetCache(LockedCache);
    auto Inserted =
        Cache.emplace(std::piecewise_construct, std::forward_as_tuple(CacheKey),
                      std::forward_as_tuple(nullptr, BS_InProgress));

    InsertionTookPlace = Inserted.second;
    BuildResult = &Inserted.first->second;
  }

  // no insertion took place, thus some other thread has already inserted smth
  // in the cache
  if (!InsertionTookPlace) {
    for (;;) {
      RetT *Result = waitUntilBuilt<ExceptionT>(KPCache, BuildResult);

      if (Result)
        return BuildResult;

      // Previous build is failed. There was no SYCL exception though.
      // We might try to build once more.
      int Expected = BS_Failed;
      int Desired = BS_InProgress;

      if (BuildResult->State.compare_exchange_strong(Expected, Desired))
        break; // this thread is the building thread now
    }
  }

  // only the building thread will run this
  try {
    RetT *Desired = Build();

#ifndef NDEBUG
    RetT *Expected = nullptr;

    if (!BuildResult->Ptr.compare_exchange_strong(Expected, Desired))
      // We've got a funny story here
      assert(false && "We've build an entity that is already have been built.");
#else
    BuildResult->Ptr.store(Desired);
#endif

    {
      // Even if shared variable is atomic, it must be modified under the mutex
      // in order to correctly publish the modification to the waiting thread
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BS_Done);
    }

    KPCache.notifyAllBuild(*BuildResult);

    return BuildResult;
  } catch (const exception &Ex) {
    BuildResult->Error.Msg = Ex.what();
    BuildResult->Error.Code = Ex.get_cl_code();

    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BS_Failed);
    }

    KPCache.notifyAllBuild(*BuildResult);

    std::rethrow_exception(std::current_exception());
  } catch (...) {
    {
      std::lock_guard<std::mutex> Lock(BuildResult->MBuildResultMutex);
      BuildResult->State.store(BS_Failed);
    }

    KPCache.notifyAllBuild(*BuildResult);

    std::rethrow_exception(std::current_exception());
  }
}

// TODO replace this with a new PI API function
static bool isDeviceBinaryTypeSupported(const context &C,
                                        RT::PiDeviceBinaryType Format) {
  // All formats except PI_DEVICE_BINARY_TYPE_SPIRV are supported.
  if (Format != PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  const backend ContextBackend =
      detail::getSyclObjImpl(C)->getPlugin().getBackend();

  // The CUDA backend cannot use SPIR-V
  if (ContextBackend == backend::cuda)
    return false;

  // The Vulkan backend supports only SPIRV
  if (ContextBackend == backend::vulkan && Format == PI_DEVICE_BINARY_TYPE_SPIRV)
    return true;

  vector_class<device> Devices = C.get_devices();

  // Program type is SPIR-V, so we need a device compiler to do JIT.
  for (const device &D : Devices) {
    if (!D.get_info<info::device::is_compiler_available>())
      return false;
  }

  // OpenCL 2.1 and greater require clCreateProgramWithIL
  if (ContextBackend == backend::opencl) {
    std::string ver = C.get_platform().get_info<info::platform::version>();
    if (ver.find("OpenCL 1.0") == std::string::npos &&
        ver.find("OpenCL 1.1") == std::string::npos &&
        ver.find("OpenCL 1.2") == std::string::npos &&
        ver.find("OpenCL 2.0") == std::string::npos)
      return true;
  }

  for (const device &D : Devices) {
    // We need cl_khr_il_program extension to be present
    // and we can call clCreateProgramWithILKHR using the extension
    vector_class<string_class> Extensions =
        D.get_info<info::device::extensions>();
    if (Extensions.end() ==
        std::find(Extensions.begin(), Extensions.end(), "cl_khr_il_program"))
      return false;
  }

  return true;
}

static const char *getFormatStr(RT::PiDeviceBinaryType Format) {
  switch (Format) {
  case PI_DEVICE_BINARY_TYPE_NONE:
    return "none";
  case PI_DEVICE_BINARY_TYPE_NATIVE:
    return "native";
  case PI_DEVICE_BINARY_TYPE_SPIRV:
    return "SPIR-V";
  case PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE:
    return "LLVM IR";
  }
  assert(false && "Unknown device image format");
  return "unknown";
}

RT::PiProgram ProgramManager::createPIProgram(const RTDeviceBinaryImage &Img,
                                              const context &Context,
                                              const device &Device) {
  if (DbgProgMgr > 0)
    std::cerr << ">>> ProgramManager::createPIProgram(" << &Img << ", "
              << getRawSyclObjImpl(Context) << ", " << getRawSyclObjImpl(Device)
              << ")\n";
  const pi_device_binary_struct &RawImg = Img.getRawData();

  // perform minimal sanity checks on the device image and the descriptor
  if (RawImg.BinaryEnd < RawImg.BinaryStart) {
    throw runtime_error("Malformed device program image descriptor",
                        PI_INVALID_VALUE);
  }
  if (RawImg.BinaryEnd == RawImg.BinaryStart) {
    throw runtime_error("Invalid device program image: size is zero",
                        PI_INVALID_VALUE);
  }
  size_t ImgSize = Img.getSize();

  // TODO if the binary image is a part of the fat binary, the clang
  //   driver should have set proper format option to the
  //   clang-offload-wrapper. The fix depends on AOT compilation
  //   implementation, so will be implemented together with it.
  //   Img->Format can't be updated as it is inside of the in-memory
  //   OS module binary.
  RT::PiDeviceBinaryType Format = Img.getFormat();

  if (Format == PI_DEVICE_BINARY_TYPE_NONE)
    Format = pi::getBinaryImageFormat(RawImg.BinaryStart, ImgSize);
  // RT::PiDeviceBinaryType Format = Img->Format;
  // assert(Format != PI_DEVICE_BINARY_TYPE_NONE && "Image format not set");

  if (!isDeviceBinaryTypeSupported(Context, Format))
    throw feature_not_supported(
        "SPIR-V online compilation is not supported in this context",
        PI_INVALID_OPERATION);

  // Load the image
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  RT::PiProgram Res =
      Format == PI_DEVICE_BINARY_TYPE_SPIRV
          ? createSpirvProgram(Ctx, RawImg.BinaryStart, ImgSize)
          : createBinaryProgram(Ctx, Device, RawImg.BinaryStart, ImgSize);

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    // associate the PI program with the image it was created for
    NativePrograms[Res] = &Img;
  }

  if (DbgProgMgr > 1)
    std::cerr << "created program: " << Res
              << "; image format: " << getFormatStr(Format) << "\n";

  return Res;
}

RT::PiProgram ProgramManager::getBuiltPIProgram(OSModuleHandle M,
                                                const context &Context,
                                                const device &Device,
                                                const string_class &KernelName,
                                                const program_impl *Prg,
                                                bool JITCompilationIsRequired) {
  KernelSetId KSId = getKernelSetId(M, KernelName);

  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  using PiProgramT = KernelProgramCache::PiProgramT;
  using ProgramCacheT = KernelProgramCache::ProgramCacheT;

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireCachedPrograms();
  };
  auto GetF = [](const Locked<ProgramCacheT> &LockedCache) -> ProgramCacheT & {
    return LockedCache.get();
  };
  auto BuildF = [this, &M, &KSId, &Context, &Device, Prg,
                 &JITCompilationIsRequired] {
    const RTDeviceBinaryImage &Img =
        getDeviceImage(M, KSId, Context, Device, JITCompilationIsRequired);

    ContextImplPtr ContextImpl = getSyclObjImpl(Context);
    const detail::plugin &Plugin = ContextImpl->getPlugin();
    RT::PiProgram NativePrg = createPIProgram(Img, Context, Device);
    if (Prg)
      flushSpecConstants(*Prg, getSyclObjImpl(Device)->getHandleRef(),
                         NativePrg, &Img);
    ProgramPtr ProgramManaged(
        NativePrg, Plugin.getPiPlugin().PiFunctionTable.piProgramRelease);

    // Link a fallback implementation of device libraries if they are not
    // supported by a device compiler.
    // Pre-compiled programs are supposed to be already linked.
    // If device image is not SPIR-V, DeviceLibReqMask will be 0 which means
    // no fallback device library will be linked.
    uint32_t DeviceLibReqMask = 0;
    if (Img.getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV &&
        !SYCLConfig<SYCL_DEVICELIB_NO_FALLBACK>::get())
      DeviceLibReqMask = getDeviceLibReqMask(Img);

    ProgramPtr BuiltProgram =
        build(std::move(ProgramManaged), ContextImpl, Img.getCompileOptions(),
              Img.getLinkOptions(), getRawSyclObjImpl(Device)->getHandleRef(),
              ContextImpl->getCachedLibPrograms(), DeviceLibReqMask);

    {
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      NativePrograms[BuiltProgram.get()] = &Img;
    }
    return BuiltProgram.release();
  };

  SerializedObj SpecConsts;
  if (Prg)
    Prg->stableSerializeSpecConstRegistry(SpecConsts);

  const RT::PiDevice PiDevice = getRawSyclObjImpl(Device)->getHandleRef();
  auto BuildResult = getOrBuild<PiProgramT, compile_program_error>(
      Cache,
      std::make_pair(std::make_pair(std::move(SpecConsts), KSId), PiDevice),
      AcquireF, GetF, BuildF);
  return BuildResult->Ptr.load();
}

std::pair<RT::PiKernel, std::mutex *> ProgramManager::getOrCreateKernel(
    OSModuleHandle M, const context &Context, const device &Device,
    const string_class &KernelName, const program_impl *Prg) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getOrCreateKernel(" << M << ", "
              << getRawSyclObjImpl(Context) << ", " << getRawSyclObjImpl(Device)
              << ", " << KernelName << ")\n";
  }

  RT::PiProgram Program =
      getBuiltPIProgram(M, Context, Device, KernelName, Prg);
  const ContextImplPtr Ctx = getSyclObjImpl(Context);

  using PiKernelT = KernelProgramCache::PiKernelT;
  using KernelCacheT = KernelProgramCache::KernelCacheT;
  using KernelByNameT = KernelProgramCache::KernelByNameT;

  KernelProgramCache &Cache = Ctx->getKernelProgramCache();

  auto AcquireF = [](KernelProgramCache &Cache) {
    return Cache.acquireKernelsPerProgramCache();
  };
  auto GetF =
      [&Program](const Locked<KernelCacheT> &LockedCache) -> KernelByNameT & {
    return LockedCache.get()[Program];
  };
  auto BuildF = [&Program, &KernelName, &Ctx] {
    PiKernelT *Result = nullptr;

    // TODO need some user-friendly error/exception
    // instead of currently obscure one
    const detail::plugin &Plugin = Ctx->getPlugin();
    Plugin.call<PiApiKind::piKernelCreate>(Program, KernelName.c_str(),
                                           &Result);

    // Some PI Plugins (like OpenCL) require this call to enable USM
    // For others, PI will turn this into a NOP.
    Plugin.call<PiApiKind::piKernelSetExecInfo>(Result, PI_USM_INDIRECT_ACCESS,
                                                sizeof(pi_bool), &PI_TRUE);

    return Result;
  };

  const RT::PiDevice PiDevice = getRawSyclObjImpl(Device)->getHandleRef();
  auto BuildResult = getOrBuild<PiKernelT, invalid_object_error>(
      Cache, std::make_pair(KernelName, PiDevice), AcquireF, GetF, BuildF);
  return std::make_pair(BuildResult->Ptr.load(),
                        &(BuildResult->MBuildResultMutex));
}

RT::PiProgram
ProgramManager::getPiProgramFromPiKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context) {
  RT::PiProgram Program;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piKernelGetInfo>(
      Kernel, PI_KERNEL_INFO_PROGRAM, sizeof(RT::PiProgram), &Program, nullptr);
  return Program;
}

string_class ProgramManager::getProgramBuildLog(const RT::PiProgram &Program,
                                                const ContextImplPtr Context) {
  size_t PIDevicesSize = 0;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES, 0,
                                           nullptr, &PIDevicesSize);
  vector_class<RT::PiDevice> PIDevices(PIDevicesSize / sizeof(RT::PiDevice));
  Plugin.call<PiApiKind::piProgramGetInfo>(Program, PI_PROGRAM_INFO_DEVICES,
                                           PIDevicesSize, PIDevices.data(),
                                           nullptr);
  string_class Log = "The program was built for " +
                     std::to_string(PIDevices.size()) + " devices";
  for (RT::PiDevice &Device : PIDevices) {
    std::string DeviceBuildInfoString;
    size_t DeviceBuildInfoStrSize = 0;
    Plugin.call<PiApiKind::piProgramGetBuildInfo>(
        Program, Device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
        &DeviceBuildInfoStrSize);
    if (DeviceBuildInfoStrSize > 0) {
      vector_class<char> DeviceBuildInfo(DeviceBuildInfoStrSize);
      Plugin.call<PiApiKind::piProgramGetBuildInfo>(
          Program, Device, CL_PROGRAM_BUILD_LOG, DeviceBuildInfoStrSize,
          DeviceBuildInfo.data(), nullptr);
      DeviceBuildInfoString = std::string(DeviceBuildInfo.data());
    }

    std::string DeviceNameString;
    size_t DeviceNameStrSize = 0;
    Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME, 0,
                                            nullptr, &DeviceNameStrSize);
    if (DeviceNameStrSize > 0) {
      vector_class<char> DeviceName(DeviceNameStrSize);
      Plugin.call<PiApiKind::piDeviceGetInfo>(Device, PI_DEVICE_INFO_NAME,
                                              DeviceNameStrSize,
                                              DeviceName.data(), nullptr);
      DeviceNameString = std::string(DeviceName.data());
    }
    Log += "\nBuild program log for '" + DeviceNameString + "':\n" +
           DeviceBuildInfoString;
  }
  return Log;
}

// TODO device libraries may use scpecialization constants, manifest files, etc.
// To support that they need to be delivered in a different container - so that
// pi_device_binary_struct can be created for each of them.
static bool loadDeviceLib(const ContextImplPtr Context, const char *Name,
                          RT::PiProgram &Prog) {
  std::string LibSyclDir = OSUtil::getCurrentDSODir();
  std::ifstream File(LibSyclDir + OSUtil::DirSep + Name,
                     std::ifstream::in | std::ifstream::binary);
  if (!File.good()) {
    return false;
  }

  File.seekg(0, std::ios::end);
  size_t FileSize = File.tellg();
  File.seekg(0, std::ios::beg);
  std::vector<char> FileContent(FileSize);
  File.read(&FileContent[0], FileSize);
  File.close();

  Prog =
      createSpirvProgram(Context, (unsigned char *)&FileContent[0], FileSize);
  return Prog != nullptr;
}

static const char *getDeviceLibFilename(DeviceLibExt Extension) {
  switch (Extension) {
  case DeviceLibExt::cl_intel_devicelib_assert:
    return "libsycl-fallback-cassert.spv";
  case DeviceLibExt::cl_intel_devicelib_math:
    return "libsycl-fallback-cmath.spv";
  case DeviceLibExt::cl_intel_devicelib_math_fp64:
    return "libsycl-fallback-cmath-fp64.spv";
  case DeviceLibExt::cl_intel_devicelib_complex:
    return "libsycl-fallback-complex.spv";
  case DeviceLibExt::cl_intel_devicelib_complex_fp64:
    return "libsycl-fallback-complex-fp64.spv";
  }
  throw compile_program_error("Unhandled (new?) device library extension",
                              PI_INVALID_OPERATION);
}

static const char *getDeviceLibExtensionStr(DeviceLibExt Extension) {
  switch (Extension) {
  case DeviceLibExt::cl_intel_devicelib_assert:
    return "cl_intel_devicelib_assert";
  case DeviceLibExt::cl_intel_devicelib_math:
    return "cl_intel_devicelib_math";
  case DeviceLibExt::cl_intel_devicelib_math_fp64:
    return "cl_intel_devicelib_math_fp64";
  case DeviceLibExt::cl_intel_devicelib_complex:
    return "cl_intel_devicelib_complex";
  case DeviceLibExt::cl_intel_devicelib_complex_fp64:
    return "cl_intel_devicelib_complex_fp64";
  }
  throw compile_program_error("Unhandled (new?) device library extension",
                              PI_INVALID_OPERATION);
}

static RT::PiProgram loadDeviceLibFallback(
    const ContextImplPtr Context, DeviceLibExt Extension,
    const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms) {

  const char *LibFileName = getDeviceLibFilename(Extension);
  auto CacheResult = CachedLibPrograms.emplace(
      std::make_pair(std::make_pair(Extension, Device), nullptr));
  bool Cached = !CacheResult.second;
  auto LibProgIt = CacheResult.first;
  RT::PiProgram &LibProg = LibProgIt->second;

  if (Cached)
    return LibProg;

  if (!loadDeviceLib(Context, LibFileName, LibProg)) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(std::string("Failed to load ") + LibFileName,
                                PI_INVALID_VALUE);
  }

  const detail::plugin &Plugin = Context->getPlugin();
  // TODO no spec constants are used in the std libraries, support in the future
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramCompile>(
      LibProg,
      /*num devices = */ 1, &Device,
      // Do not use compile options for library programs: it is not clear
      // if user options (image options) are supposed to be applied to
      // library program as well, and what actually happens to a SPIR-V
      // program if we apply them.
      "", 0, nullptr, nullptr, nullptr, nullptr);
  if (Error != PI_SUCCESS) {
    CachedLibPrograms.erase(LibProgIt);
    throw compile_program_error(
        ProgramManager::getProgramBuildLog(LibProg, Context), Error);
  }

  return LibProg;
}

ProgramManager::ProgramManager() {
  const char *SpvFile = std::getenv(UseSpvEnv);
  // If a SPIR-V file is specified with an environment variable,
  // register the corresponding image
  if (SpvFile) {
    m_UseSpvFile = true;
    // The env var requests that the program is loaded from a SPIR-V file on
    // disk
    std::ifstream File(SpvFile, std::ios::binary);

    if (!File.is_open())
      throw runtime_error(std::string("Can't open file specified via ") +
                              UseSpvEnv + ": " + SpvFile,
                          PI_INVALID_VALUE);
    File.seekg(0, std::ios::end);
    size_t Size = File.tellg();
    std::unique_ptr<char[]> Data(new char[Size]);
    File.seekg(0);
    File.read(Data.get(), Size);
    File.close();
    if (!File.good())
      throw runtime_error(std::string("read from ") + SpvFile +
                              std::string(" failed"),
                          PI_INVALID_VALUE);
    auto ImgPtr = make_unique_ptr<DynRTDeviceBinaryImage>(
        std::move(Data), Size, OSUtil::DummyModuleHandle);

    if (DbgProgMgr > 0) {
      std::cerr << "loaded device image binary from " << SpvFile << "\n";
      std::cerr << "format: " << getFormatStr(ImgPtr->getFormat()) << "\n";
    }
    // No need for a mutex here since all access to these private fields is
    // blocked until the construction of the ProgramManager singleton is
    // finished.
    m_DeviceImages[SpvFileKSId].reset(
        new std::vector<RTDeviceBinaryImageUPtr>());
    m_DeviceImages[SpvFileKSId]->push_back(std::move(ImgPtr));
  }
}

RTDeviceBinaryImage &
ProgramManager::getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                               const context &Context, const device &Device,
                               bool JITCompilationIsRequired) {
  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::getDeviceImage(" << M << ", \"" << KSId
              << "\", " << getRawSyclObjImpl(Context) << ", "
              << getRawSyclObjImpl(Device) << ", " << JITCompilationIsRequired
              << ")\n";

    std::cerr << "available device images:\n";
    debugPrintBinaryImages();
  }
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  std::vector<RTDeviceBinaryImageUPtr> &Imgs = *m_DeviceImages[KSId];
  const ContextImplPtr Ctx = getSyclObjImpl(Context);
  pi_uint32 ImgInd = 0;
  RTDeviceBinaryImage *Img = nullptr;

  // TODO: There may be cases with cl::sycl::program class usage in source code
  // that will result in a multi-device context. This case needs to be handled
  // here or at the program_impl class level

  // Ask the native runtime under the given context to choose the device image
  // it prefers.
  std::vector<pi_device_binary> RawImgs(Imgs.size());
  for (unsigned I = 0; I < Imgs.size(); I++)
    RawImgs[I] = const_cast<pi_device_binary>(&Imgs[I]->getRawData());

  Ctx->getPlugin().call<PiApiKind::piextDeviceSelectBinary>(
      getSyclObjImpl(Device)->getHandleRef(), RawImgs.data(),
      (cl_uint)RawImgs.size(), &ImgInd);

  if (JITCompilationIsRequired) {
    // If the image is already compiled with AOT, throw an exception.
    const pi_device_binary_struct &RawImg = Imgs[ImgInd]->getRawData();
    if ((strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_X86_64) == 0) ||
        (strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_GEN) == 0) ||
        (strcmp(RawImg.DeviceTargetSpec,
                __SYCL_PI_DEVICE_BINARY_TARGET_SPIRV64_FPGA) == 0)) {
      throw feature_not_supported("Recompiling AOT image is not supported",
                                  PI_INVALID_OPERATION);
    }
  }

  Img = Imgs[ImgInd].get();

  if (DbgProgMgr > 0) {
    std::cerr << "selected device image: " << &Img->getRawData() << "\n";
    Img->print();
  }

  if (std::getenv("SYCL_DUMP_IMAGES") && !m_UseSpvFile)
    dumpImage(*Img, KSId);
  return *Img;
}

static bool isDeviceLibRequired(DeviceLibExt Ext, uint32_t DeviceLibReqMask) {
  uint32_t Mask =
      0x1 << (static_cast<uint32_t>(Ext) -
              static_cast<uint32_t>(DeviceLibExt::cl_intel_devicelib_assert));
  return ((DeviceLibReqMask & Mask) == Mask);
}

static std::vector<RT::PiProgram> getDeviceLibPrograms(
    const ContextImplPtr Context, const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms,
    uint32_t DeviceLibReqMask) {
  std::vector<RT::PiProgram> Programs;

  std::pair<DeviceLibExt, bool> RequiredDeviceLibExt[] = {
      {DeviceLibExt::cl_intel_devicelib_assert,
       /* is fallback loaded? */ false},
      {DeviceLibExt::cl_intel_devicelib_math, false},
      {DeviceLibExt::cl_intel_devicelib_math_fp64, false},
      {DeviceLibExt::cl_intel_devicelib_complex, false},
      {DeviceLibExt::cl_intel_devicelib_complex_fp64, false}};

  // Disable all devicelib extensions requiring fp64 support if at least
  // one underlying device doesn't support cl_khr_fp64.
  std::string DevExtList =
      get_device_info<std::string, info::device::extensions>::get(
          Device, Context->getPlugin());
  const bool fp64Support = (DevExtList.npos != DevExtList.find("cl_khr_fp64"));

  // Load a fallback library for an extension if the device does not
  // support it.
  for (auto &Pair : RequiredDeviceLibExt) {
    DeviceLibExt Ext = Pair.first;
    bool &FallbackIsLoaded = Pair.second;

    if (FallbackIsLoaded) {
      continue;
    }

    if (!isDeviceLibRequired(Ext, DeviceLibReqMask)) {
      continue;
    }
    if ((Ext == DeviceLibExt::cl_intel_devicelib_math_fp64 ||
         Ext == DeviceLibExt::cl_intel_devicelib_complex_fp64) &&
        !fp64Support) {
      continue;
    }

    const char *ExtStr = getDeviceLibExtensionStr(Ext);

    bool InhibitNativeImpl = false;
    if (const char *Env = getenv("SYCL_DEVICELIB_INHIBIT_NATIVE")) {
      InhibitNativeImpl = strstr(Env, ExtStr) != nullptr;
    }

    bool DeviceSupports = DevExtList.npos != DevExtList.find(ExtStr);

    if (!DeviceSupports || InhibitNativeImpl) {
      Programs.push_back(
          loadDeviceLibFallback(Context, Ext, Device, CachedLibPrograms));
      FallbackIsLoaded = true;
    }
  }
  return Programs;
}

ProgramManager::ProgramPtr ProgramManager::build(
    ProgramPtr Program, const ContextImplPtr Context,
    const string_class &CompileOptions, const string_class &LinkOptions,
    const RT::PiDevice &Device,
    std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>
        &CachedLibPrograms,
    uint32_t DeviceLibReqMask) {

  if (DbgProgMgr > 0) {
    std::cerr << ">>> ProgramManager::build(" << Program.get() << ", "
              << CompileOptions << ", " << LinkOptions << ", ... " << Device
              << ")\n";
  }

  bool LinkDeviceLibs = (DeviceLibReqMask != 0);
  const char *CompileOpts = std::getenv("SYCL_PROGRAM_COMPILE_OPTIONS");
  if (!CompileOpts) {
    CompileOpts = CompileOptions.c_str();
  }
  const char *LinkOpts = std::getenv("SYCL_PROGRAM_LINK_OPTIONS");
  if (!LinkOpts) {
    LinkOpts = LinkOptions.c_str();
  }

  // TODO: Currently, online linking isn't implemented yet on Level Zero.
  // To enable device libraries and unify the behaviors on all backends,
  // online linking is disabled temporarily, all fallback device libraries
  // will be linked offline. When Level Zero supports online linking, we need
  // to remove the line of code below and switch back to online linking.
  LinkDeviceLibs = false;

  // TODO: this is a temporary workaround for GPU tests for ESIMD compiler.
  // We do not link with other device libraries, because it may fail
  // due to unrecognized SPIR-V format of those libraries.
  if (std::string(CompileOpts).find(std::string("-cmc")) != std::string::npos ||
      std::string(CompileOpts).find(std::string("-vc-codegen")) !=
          std::string::npos)
    LinkDeviceLibs = false;

  std::vector<RT::PiProgram> LinkPrograms;
  if (LinkDeviceLibs) {
    LinkPrograms = getDeviceLibPrograms(Context, Device, CachedLibPrograms,
                                        DeviceLibReqMask);
  }

  const detail::plugin &Plugin = Context->getPlugin();
  if (LinkPrograms.empty()) {
    std::string Opts(CompileOpts);

    RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramBuild>(
        Program.get(), /*num devices =*/1, &Device, Opts.c_str(), nullptr,
        nullptr);
    if (Error != PI_SUCCESS)
      throw compile_program_error(getProgramBuildLog(Program.get(), Context),
                                  Error);
    return Program;
  }

  // Include the main program and compile/link everything together
  Plugin.call<PiApiKind::piProgramCompile>(Program.get(), /*num devices =*/1,
                                           &Device, CompileOpts, 0, nullptr,
                                           nullptr, nullptr, nullptr);
  LinkPrograms.push_back(Program.get());

  RT::PiProgram LinkedProg = nullptr;
  RT::PiResult Error = Plugin.call_nocheck<PiApiKind::piProgramLink>(
      Context->getHandleRef(), /*num devices =*/1, &Device, LinkOpts,
      LinkPrograms.size(), LinkPrograms.data(), nullptr, nullptr, &LinkedProg);

  // Link program call returns a new program object if all parameters are valid,
  // or NULL otherwise. Release the original (user) program.
  Program.reset(LinkedProg);
  if (Error != PI_SUCCESS) {
    if (LinkedProg) {
      // A non-trivial error occurred during linkage: get a build log, release
      // an incomplete (but valid) LinkedProg, and throw.
      throw compile_program_error(getProgramBuildLog(LinkedProg, Context),
                                  Error);
    }
    Plugin.checkPiResult(Error);
  }
  return Program;
}

static ProgramManager::KernelArgMask
createKernelArgMask(const pi::ByteArray &Bytes) {
  const int NBytesForSize = 8;
  const int NBitsInElement = 8;
  std::uint64_t SizeInBits = 0;
  for (int I = 0; I < NBytesForSize; ++I)
    SizeInBits |= static_cast<std::uint64_t>(Bytes[I]) << I * NBitsInElement;

  ProgramManager::KernelArgMask Result;
  for (std::uint64_t I = 0; I < SizeInBits; ++I) {
    std::uint8_t Byte = Bytes[NBytesForSize + (I / NBitsInElement)];
    Result.push_back(Byte & (1 << (I % NBitsInElement)));
  }

  return Result;
}

void ProgramManager::addImages(pi_device_binaries DeviceBinary) {
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());

  for (int I = 0; I < DeviceBinary->NumDeviceBinaries; I++) {
    pi_device_binary RawImg = &(DeviceBinary->DeviceBinaries[I]);
    OSModuleHandle M = OSUtil::getOSModuleHandle(RawImg);
    const _pi_offload_entry EntriesB = RawImg->EntriesBegin;
    const _pi_offload_entry EntriesE = RawImg->EntriesEnd;
    auto Img = make_unique_ptr<RTDeviceBinaryImage>(RawImg, M);

    // Fill the kernel argument mask map
    const pi::DeviceBinaryImage::PropertyRange &KPOIRange =
        Img->getKernelParamOptInfo();
    if (KPOIRange.isAvailable()) {
      KernelNameToArgMaskMap &ArgMaskMap =
          m_EliminatedKernelArgMasks[Img.get()];
      for (const auto &Info : KPOIRange)
        ArgMaskMap[Info->Name] =
            createKernelArgMask(pi::DeviceBinaryProperty(Info).asByteArray());
    }
    // Use the entry information if it's available
    if (EntriesB != EntriesE) {
      // The kernel sets for any pair of images are either disjoint or
      // identical, look up the kernel set using the first kernel name...
      StrToKSIdMap &KSIdMap = m_KernelSets[M];
      auto KSIdIt = KSIdMap.find(EntriesB->name);
      if (KSIdIt != KSIdMap.end()) {
        for (_pi_offload_entry EntriesIt = EntriesB + 1; EntriesIt != EntriesE;
             ++EntriesIt)
          assert(KSIdMap[EntriesIt->name] == KSIdIt->second &&
                 "Kernel sets are not disjoint");
        auto &Imgs = m_DeviceImages[KSIdIt->second];
        assert(Imgs && "Device image vector should have been already created");
        Imgs->push_back(std::move(Img));
        continue;
      }
      // ... or create the set first if it hasn't been
      KernelSetId KSId = getNextKernelSetId();
      for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE;
           ++EntriesIt) {
        auto Result = KSIdMap.insert(std::make_pair(EntriesIt->name, KSId));
        (void)Result;
        assert(Result.second && "Kernel sets are not disjoint");
      }
      m_DeviceImages[KSId].reset(new std::vector<RTDeviceBinaryImageUPtr>());
      m_DeviceImages[KSId]->push_back(std::move(Img));
      continue;
    }
    // Otherwise assume that the image contains all kernels associated with the
    // module
    KernelSetId &KSId = m_OSModuleKernelSets[M];
    if (KSId == 0)
      KSId = getNextKernelSetId();

    auto &Imgs = m_DeviceImages[KSId];
    if (!Imgs)
      Imgs.reset(new std::vector<RTDeviceBinaryImageUPtr>());
    Imgs->push_back(std::move(Img));
  }
}

void ProgramManager::debugPrintBinaryImages() const {
  for (const auto &ImgVecIt : m_DeviceImages) {
    std::cerr << "  ++++++ Kernel set: " << ImgVecIt.first << "\n";
    for (const auto &Img : *ImgVecIt.second)
      Img.get()->print();
  }
}

KernelSetId ProgramManager::getNextKernelSetId() const {
  // No need for atomic, should be guarded by the caller
  static KernelSetId Result = LastKSId;
  return ++Result;
}

KernelSetId
ProgramManager::getKernelSetId(OSModuleHandle M,
                               const string_class &KernelName) const {
  // If the env var instructs to use image from a file,
  // return the kernel set associated with it
  if (m_UseSpvFile && M == OSUtil::ExeModuleHandle)
    return SpvFileKSId;
  std::lock_guard<std::mutex> Guard(Sync::getGlobalLock());
  auto KSIdMapIt = m_KernelSets.find(M);
  if (KSIdMapIt != m_KernelSets.end()) {
    const StrToKSIdMap &KSIdMap = KSIdMapIt->second;
    auto KSIdIt = KSIdMap.find(KernelName);
    // If the kernel has been assigned to a kernel set, return it
    if (KSIdIt != KSIdMap.end())
      return KSIdIt->second;
  }
  // If no kernel set was found check if there is a kernel set containing
  // all kernels in the given module
  auto ModuleKSIdIt = m_OSModuleKernelSets.find(M);
  if (ModuleKSIdIt != m_OSModuleKernelSets.end())
    return ModuleKSIdIt->second;

  throw runtime_error("No kernel named " + KernelName + " was found",
                      PI_INVALID_KERNEL_NAME);
}

void ProgramManager::dumpImage(const RTDeviceBinaryImage &Img,
                               KernelSetId KSId) const {
  std::string Fname("sycl_");
  const pi_device_binary_struct &RawImg = Img.getRawData();
  Fname += RawImg.DeviceTargetSpec;
  Fname += std::to_string(KSId);
  std::string Ext;

  RT::PiDeviceBinaryType Format = Img.getFormat();
  if (Format == PI_DEVICE_BINARY_TYPE_SPIRV)
    Ext = ".spv";
  else if (Format == PI_DEVICE_BINARY_TYPE_LLVMIR_BITCODE)
    Ext = ".bc";
  else
    Ext = ".bin";
  Fname += Ext;

  std::ofstream F(Fname, std::ios::binary);

  if (!F.is_open()) {
    throw runtime_error("Can not write " + Fname, PI_ERROR_UNKNOWN);
  }
  Img.dump(F);
  F.close();
}

void ProgramManager::flushSpecConstants(const program_impl &Prg,
                                        RT::PiDevice Device,
                                        RT::PiProgram NativePrg,
                                        const RTDeviceBinaryImage *Img) {
  if (DbgProgMgr > 2) {
    std::cerr << ">>> ProgramManager::flushSpecConstants(" << Prg.get()
              << ",...)\n";
  }
  if (!Prg.hasSetSpecConstants())
    return; // nothing to do
  pi::PiProgram PrgHandle = Prg.getHandleRef();
  // program_impl can't correspond to two different native programs
  assert(!NativePrg || !PrgHandle || (NativePrg == PrgHandle));
  NativePrg = NativePrg ? NativePrg : PrgHandle;

  if (!Img) {
    // caller hasn't provided the image object - find it
    { // make sure NativePrograms map access is synchronized
      std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
      auto It = NativePrograms.find(NativePrg);
      if (It == NativePrograms.end())
        throw sycl::ONEAPI::experimental::spec_const_error(
            "spec constant is set in a program w/o a binary image",
            PI_INVALID_OPERATION);
      Img = It->second;
    }
    if (!Img->supportsSpecConstants()) {
      if (DbgProgMgr > 0)
        std::cerr << ">>> ProgramManager::flushSpecConstants: binary image "
                  << &Img->getRawData() << " doesn't support spec constants\n";
      // This device binary image does not support runtime setting of
      // specialization constants; compiler must have generated default values.
      // NOTE: Can't throw here, as it would always take place with AOT
      //-compiled code. New Khronos 2020 spec should fix this inconsistency.
      return;
    }
  }
  Prg.flush_spec_constants(*Img, NativePrg);
}

// If the kernel is loaded from spv file, it may not include DeviceLib require
// mask, sycl runtime won't know which fallback device libraries are needed. In
// such case, the safest way is to load all fallback device libraries.
uint32_t ProgramManager::getDeviceLibReqMask(const RTDeviceBinaryImage &Img) {
  const pi::DeviceBinaryImage::PropertyRange &DLMRange =
      Img.getDeviceLibReqMask();
  if (DLMRange.isAvailable())
    return pi::DeviceBinaryProperty(*(DLMRange.begin())).asUint32();
  else
    return 0xFFFFFFFF;
}

// TODO consider another approach with storing the masks in the integration
// header instead.
ProgramManager::KernelArgMask ProgramManager::getEliminatedKernelArgMask(
    OSModuleHandle M, const context &Context, const device &Device,
    pi::PiProgram NativePrg, const string_class &KernelName,
    bool KnownProgram) {
  // If instructed to use a spv file, assume no eliminated arguments.
  if (m_UseSpvFile && M == OSUtil::ExeModuleHandle)
    return {};

  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    auto ImgIt = NativePrograms.find(NativePrg);
    if (ImgIt != NativePrograms.end()) {
      auto MapIt = m_EliminatedKernelArgMasks.find(ImgIt->second);
      if (MapIt != m_EliminatedKernelArgMasks.end())
        return MapIt->second[KernelName];
      return {};
    }
  }

  if (KnownProgram)
    throw runtime_error("Program is not associated with a binary image",
                        PI_INVALID_VALUE);

  // If not sure whether the program was built with one of the images, try
  // finding the binary.
  // TODO this can backfire in some extreme edge cases where there's a kernel
  // name collision between our binaries and user-created native programs.
  KernelSetId KSId;
  try {
    KSId = getKernelSetId(M, KernelName);
  } catch (sycl::runtime_error &e) {
    // If the kernel name wasn't found, assume that the program wasn't created
    // from one of our device binary images.
    if (e.get_cl_code() == PI_INVALID_KERNEL_NAME)
      return {};
    std::rethrow_exception(std::current_exception());
  }
  RTDeviceBinaryImage &Img = getDeviceImage(M, KSId, Context, Device);
  {
    std::lock_guard<std::mutex> Lock(MNativeProgramsMutex);
    NativePrograms[NativePrg] = &Img;
  }
  auto MapIt = m_EliminatedKernelArgMasks.find(&Img);
  if (MapIt != m_EliminatedKernelArgMasks.end())
    return MapIt->second[KernelName];
  return {};
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

extern "C" void __sycl_register_lib(pi_device_binaries desc) {
  cl::sycl::detail::ProgramManager::getInstance().addImages(desc);
}

// Executed as a part of current module's (.exe, .dll) static initialization
extern "C" void __sycl_unregister_lib(pi_device_binaries desc) {
  // TODO implement the function
}
