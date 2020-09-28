// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------------- branchesLoops.cpp - SYCL branching test ------------------==//
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
          B[index] = 0;
          if (index[0] < 10) {
            for (int i = 0; i < index[0]; i++) {
              B[index] += i;
            }
          } else {
            for (int i = 0; i < index[0]; i++) {
              B[index] += i + 10;
            }
          }
        });
      });
    }
    for (int i = 0; i < data.size(); i++) {
      const int id = data[i];
      int j = i > 0 ? i - 1 : 0;
      if (i < 10) {
         //check with gaussian formular
         assert(id == (j*j+j)/2);
      } else {
        // check with gaussian formular but subtract gaussian until 9
         assert(id == ((j+10) * (j+10) + (j+10) - (9*9+9)) / 2 );
      }
    }
  }
}