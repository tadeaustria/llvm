// REQUIRES: gpu, vulkan

// RUN: env SYCL_BE=PI_VULKAN sycl-ls --verbose >%t.vulkan.out
// RUN: FileCheck %s --check-prefixes=CHECK-BUILTIN-GPU-VULKAN,CHECK-CUSTOM-GPU-VULKAN --input-file %t.vulkan.out

// CHECK-BUILTIN-GPU-VULKAN: gpu_selector(){{.*}}GPU :{{.*}}Vulkan
// CHECK-CUSTOM-GPU-VULKAN: custom_selector(gpu){{.*}}GPU :{{.*}}Vulkan

//==-- sycl-ls-gpu-vulkan.cpp - SYCL test for discovered/selected devices --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
