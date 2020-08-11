#TODO:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

# Plugin for OpenCL
# Create Shared library for libpi_vulkan.so.
#TODO: remove dependency on pi.hpp in sycl project.
#TODO: Currently, the pi.hpp header is common between sycl and plugin library sources. 
#This can be changed by copying the pi.hpp file in the plugins project.

find_package(Vulkan 1.2.141 REQUIRED)

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

set_target_properties(pi_vulkan PROPERTIES 
                        LINKER_LANGUAGE CXX
                     )

target_include_directories(pi_vulkan
  PRIVATE
    ${sycl_inc_dir}
  PUBLIC
    ${Vulkan_INCLUDE_DIRS}
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_vulkan PRIVATE __SYCL_BUILD_SYCL_DLL)
endif()

target_link_libraries(pi_vulkan 
	PRIVATE Vulkan::Vulkan
  PRIVATE OpenCL::Headers
  PRIVATE ${OpenCL_LIBRARIES})
    

# add dynamic library functionality for RenderDoc
if(CMAKE_DL_LIBS)
    target_link_libraries(pi_vulkan PRIVATE ${CMAKE_DL_LIBS})
endif()

add_common_options(pi_vulkan)

install(TARGETS pi_vulkan
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_vulkan
    RUNTIME DESTINATION "bin" COMPONENT pi_vulkan)