//==---- test_commands.cpp --- PI unit tests -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "TestGetPlugin.hpp"
#include <CL/sycl.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <pi_vulkan.hpp>

using namespace cl::sycl;

struct VulkanCommandsTest : public ::testing::Test {

protected:
  detail::plugin plugin = pi::initializeAndGet(backend::vulkan);

  pi_platform platform_;
  pi_device device_;
  pi_context context_;
  pi_queue queue_;

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
    ASSERT_NE(context_, nullptr);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueCreate>(
                  context_, device_, 0, &queue_)),
              PI_SUCCESS);
    ASSERT_NE(queue_, nullptr);
    ASSERT_EQ(queue_->Context_, context_);
  }

  void TearDown() override {
    plugin.call<detail::PiApiKind::piQueueRelease>(queue_);
    plugin.call<detail::PiApiKind::piContextRelease>(context_);
  }

  VulkanCommandsTest() = default;

  ~VulkanCommandsTest() = default;
};

TEST_F(VulkanCommandsTest, PIEnqueueReadBufferBlocking) {
  constexpr const size_t memSize = 10u;
  constexpr const size_t bytes = memSize * sizeof(int);
  const int data[memSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int output[memSize] = {};

  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, bytes, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                queue_, memObj, true, 0, bytes, data, 0, nullptr, nullptr)),
            PI_SUCCESS);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                queue_, memObj, true, 0, bytes, output, 0, nullptr, nullptr)),
            PI_SUCCESS);

  bool isSame =
      std::equal(std::begin(output), std::end(output), std::begin(data));
  EXPECT_TRUE(isSame);
  if (!isSame) {
    std::for_each(std::begin(output), std::end(output),
                  [](int &elem) { std::cout << elem << ","; });
    std::cout << std::endl;
  }
}

TEST_F(VulkanCommandsTest, PIEnqueueReadBufferNonBlocking) {
  constexpr const size_t memSize = 10u;
  constexpr const size_t bytes = memSize * sizeof(int);
  const int data[memSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int output[memSize] = {};

  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, bytes, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  pi_event cpIn, cpOut;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                queue_, memObj, false, 0, bytes, data, 0, nullptr, &cpIn)),
            PI_SUCCESS);
  ASSERT_NE(cpIn, nullptr);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                queue_, memObj, false, 0, bytes, output, 1, &cpIn, &cpOut)),
            PI_SUCCESS);
  ASSERT_NE(cpOut, nullptr);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEventsWait>(1, &cpOut)),
            PI_SUCCESS);

  bool isSame =
      std::equal(std::begin(output), std::end(output), std::begin(data));
  EXPECT_TRUE(isSame);
  if (!isSame) {
    std::for_each(std::begin(output), std::end(output),
                  [](int &elem) { std::cout << elem << ","; });
    std::cout << std::endl;
  }
}

TEST_F(VulkanCommandsTest, PIEnqueueReadBufferNonBlockingLarge) {
  constexpr const size_t memSize = 8192u;
  constexpr const size_t bytes = memSize * sizeof(float);
  float data[memSize];
  float output[memSize] = {};

  for (size_t i = 0; i < memSize; i++) {
    data[i] = i;
  }

  pi_mem memObj;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                context_, PI_MEM_FLAGS_ACCESS_RW, bytes, nullptr, &memObj, nullptr)),
            PI_SUCCESS);

  pi_event cpIn, cpOut;
  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                queue_, memObj, false, 0, bytes, data, 0, nullptr, &cpIn)),
            PI_SUCCESS);
  ASSERT_NE(cpIn, nullptr);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                queue_, memObj, false, 0, bytes, output, 1, &cpIn, &cpOut)),
            PI_SUCCESS);
  ASSERT_NE(cpOut, nullptr);

  ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEventsWait>(1, &cpOut)),
            PI_SUCCESS);

  bool isSame =
      std::equal(std::begin(output), std::end(output), std::begin(data));
  EXPECT_TRUE(isSame);
  if (!isSame) {
    std::for_each(std::begin(output), std::end(output),
                  [](float &elem) { std::cout << elem << ","; });
    std::cout << std::endl;
  }
}
