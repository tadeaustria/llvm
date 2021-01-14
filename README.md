# Sylkan - A SYCL Platform targeting Vulkan compute

This project is a research prototype implementation of Sylkan, which allows developers to leverage the high level abstrations of SYCL while targeting a wealth of Vulkan compute devices. 

> :warning: This is an early research prototype. While simple SYCL programs should run without changes, various sophisticated features are not implemented. Sylkan should not be used for production purposes.

For further information including compilation instructions see [GettingStartedGuide](sycl/doc/GetStartedGuide.md).

Sylkan is based on Intel's open source DPC++ toolchain.

# Intel Project for LLVM\* technology

## Introduction

Intel staging area for llvm.org contribution.
Home for Intel LLVM-based projects:

* oneAPI Data Parallel C++ compiler - see **sycl** branch. More information on
   oneAPI and DPC++ is available at
[https://www.oneapi.com/](https://www.oneapi.com/)
  * [![Linux Post Commit Checks](https://github.com/intel/llvm/workflows/Linux%20Post%20Commit%20Checks/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Linux+Post+Commit+Checks%22)
    [![Generate Doxygen documentation](https://github.com/intel/llvm/workflows/Generate%20Doxygen%20documentation/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Generate+Doxygen+documentation%22)

## License

See [LICENSE.txt](sycl/LICENSE.TXT) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Sub-projects Documentation

* oneAPI Data Parallel C++ compiler - See
  [DPC++ Documentation](https://intel.github.io/llvm-docs/)

## DPC++ extensions

DPC++ is an open, cross-architecture language built upon the ISO C++ and Khronos
SYCL\* standards. DPC++ extends these standards with a number of extensions,
which can be found in [sycl/doc/extensions](sycl/doc/extensions) directory.

\*Other names and brands may be claimed as the property of others.
