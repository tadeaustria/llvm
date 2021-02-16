#TODO:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

# Plugin for Vulkan
# Create Shared library for libpi_vulkan.so.

find_package(Vulkan 1.2.141 REQUIRED)
find_package(Threads REQUIRED)
find_package(LLVM 12 REQUIRED NO_DEFAULT_PATH PATHS ${CMAKE_BINARY_DIR})

add_library(pi_vulkan SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
  "pi_vulkan.hpp"
  "pi_vulkan.cpp"
  )

add_dependencies(pi_vulkan
  ocl-headers
)

add_dependencies(sycl-toolchain pi_vulkan)

list(GET LLVM_INCLUDE_DIRS 0 LLVM_INCLUDE_DIR)

target_include_directories(pi_vulkan
  PRIVATE
    ${sycl_inc_dir}
  PUBLIC
    ${Vulkan_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIR}/../../llvm-spirv/include
    ${LLVM_INCLUDE_DIR}/../../llvm-spirv/lib/SPIRV/libSPIRV
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_vulkan PRIVATE __SYCL_BUILD_SYCL_DLL)
endif()

# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs SPIRVLib)

target_link_libraries(pi_vulkan 
	PRIVATE Vulkan::Vulkan
  PRIVATE OpenCL::Headers
  PRIVATE Threads::Threads
  PRIVATE ${OpenCL_LIBRARIES}
  PRIVATE ${llvm_libs}
  )
    

# add dynamic library functionality for RenderDoc
if(CMAKE_DL_LIBS)
    target_link_libraries(pi_vulkan PRIVATE ${CMAKE_DL_LIBS})
endif()

add_common_options(pi_vulkan)

install(TARGETS pi_vulkan
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_vulkan
    RUNTIME DESTINATION "bin" COMPONENT pi_vulkan)