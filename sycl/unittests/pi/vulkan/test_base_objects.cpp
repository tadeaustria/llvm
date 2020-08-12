//==---- test_base_objects.cpp --- PI unit tests ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/cuda_definitions.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_vulkan.hpp>
#include <thread>

const unsigned int LATEST_KNOWN_CUDA_DRIVER_API_VERSION = 3020u;

using namespace cl::sycl;

class VulkanBaseObjectsTest : public ::testing::Test {
protected:
  detail::plugin plugin = pi::initializeAndGet(backend::vulkan);

  VulkanBaseObjectsTest() = default;

  ~VulkanBaseObjectsTest() = default;
};

TEST_F(VulkanBaseObjectsTest, piContextCreate) {
  pi_uint32 numPlatforms = 0;
  pi_platform platform = nullptr;
  pi_device device;
  ASSERT_EQ(plugin.getBackend(), backend::vulkan);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                0, nullptr, &numPlatforms)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                numPlatforms, &platform, nullptr)),
            PI_SUCCESS)
      << "piPlatformsGet failed.\n";

  ASSERT_GE(numPlatforms, 1u);
  ASSERT_NE(platform, nullptr);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piDevicesGet>(
                platform, PI_DEVICE_TYPE_GPU, 1, &device, nullptr)),
            PI_SUCCESS)
      << "piDevicesGet failed.\n";

  pi_context ctxt = nullptr;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piContextCreate>(
                nullptr, 1, &device, nullptr, nullptr, &ctxt)),
            PI_SUCCESS)
      << "piContextCreate failed.\n";

  EXPECT_NE(ctxt, nullptr);
  EXPECT_EQ(ctxt->PhDevice_, device);

  //// Retrieve the cuCtxt to check information is correct
  //CUcontext cudaContext = ctxt->;
  //unsigned int version = 0;
  //cuCtxGetApiVersion(cudaContext, &version);
  //EXPECT_EQ(version, LATEST_KNOWN_CUDA_DRIVER_API_VERSION);

  //CUresult cuErr = cuCtxDestroy(cudaContext);
  //ASSERT_EQ(cuErr, CUDA_SUCCESS);
}