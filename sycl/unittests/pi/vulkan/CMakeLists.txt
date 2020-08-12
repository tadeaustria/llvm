set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(PiVulkanTests OBJECT
  test_base_objects.cpp
#  test_commands.cpp
  test_device.cpp
#  test_interop_get_native.cpp
#  test_kernels.cpp
  test_mem_obj.cpp
#  test_primary_context.cpp
  test_queue.cpp
#  test_sampler_properties.cpp
)

add_dependencies(PiVulkanTests sycl)

target_link_libraries(PiVulkanTests PRIVATE
  LLVMTestingSupport
  OpenCL-Headers)

target_compile_definitions(PiVulkanTests PRIVATE
  GTEST_HAS_COMBINE=1)

target_include_directories(
  PiVulkanTests PUBLIC 
  ${VULKAN_INCLUDE_DIRS} 
  "../"
  "${sycl_inc_dir}/CL/sycl/detail/"
  ${sycl_inc_dir}
  "${sycl_plugin_dir}/vulkan/"
)
