// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--------------- branches.cpp - SYCL branching test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  // Id indexer
  {
    vector_class<int> data(20, -1);
    const range<1> globalRange(20);
    {
      buffer<int, 1> b(data.data(), range<1>(20),
                       {property::buffer::use_host_ptr()});
      queue myQueue;
      myQueue.submit([&](handler &cgh) {
        auto B = b.get_access<access::mode::read_write>(cgh);
        cgh.parallel_for<class id1>(globalRange, [=](id<1> index) {
          if (index[0] < 10) {
            if (index[0] < 5) {
              B[index] = 5;
            } else {
              B[index] = -5;
            }
          } else {
            if (index[0] < 15) {
              B[index] = 15;
            } else {
              B[index] = -15;
            }
          }
        });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i];
      if (i < 5) {
        assert(id == 5);
      } else if (i < 10) {
        assert(id == -5);
      } else if (i < 15) {
        assert(id == 15);
      } else {
        assert(id == -15);
      }
    }
  }
}