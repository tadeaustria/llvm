//==---- test_queue.cpp --- PI unit tests ----------------------------------==//
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

using namespace cl::sycl;

struct VulkanTestQueue : public ::testing::Test {

protected:
  detail::plugin plugin = pi::initializeAndGet(backend::vulkan);

  pi_platform platform_;
  pi_device device_;
  pi_context context_;

  void SetUp() override {
    pi_uint32 numPlatforms = 0;
    ASSERT_EQ(plugin.getBackend(), backend::vulkan);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  0, nullptr, &numPlatforms)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  numPlatforms, &platform_, nullptr)),
              PI_SUCCESS)
        << "piPlatformsGet failed.\n";

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piDevicesGet>(
                  platform_, PI_DEVICE_TYPE_GPU, 1, &device_, nullptr)),
              PI_SUCCESS);
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piContextCreate>(
                  nullptr, 1, &device_, nullptr, nullptr, &context_)),
              PI_SUCCESS);
    EXPECT_NE(context_, nullptr);
  }

  void TearDown() override {
    plugin.call<detail::PiApiKind::piDeviceRelease>(device_);
    plugin.call<detail::PiApiKind::piContextRelease>(context_);
  }

  VulkanTestQueue() = default;

  ~VulkanTestQueue() = default;
};

TEST_F(VulkanTestQueue, PICreateQueueSimple) {
  pi_queue queue;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, 0, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->Context_, context_);

  ASSERT_EQ(queue->RefCounter_, 1u);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}

TEST_F(VulkanTestQueue, PICreateQueueSimpleProperties) {
  pi_queue queue;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueCreate>(
                context_, device_, CL_QUEUE_ON_DEVICE, &queue)),
            PI_SUCCESS);
  ASSERT_NE(queue, nullptr);
  EXPECT_EQ(queue->Context_, context_);

  // Although properties are not used in Vulkan yet, they should be
  // set and read properly
  pi_queue_properties propertyRead;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueGetInfo>(
                queue, PI_QUEUE_INFO_PROPERTIES, sizeof(pi_queue_properties),
                &propertyRead, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ(propertyRead, (pi_uint64)CL_QUEUE_ON_DEVICE);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueRelease>(queue)),
            PI_SUCCESS);
}