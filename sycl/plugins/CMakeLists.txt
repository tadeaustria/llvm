set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(SYCL_BUILD_PI_CUDA)
 add_subdirectory(cuda)
endif()
if(SYCL_BUILD_PI_VULKAN)
 add_subdirectory(vulkan)
endif()

add_subdirectory(opencl)
add_subdirectory(level_zero)
