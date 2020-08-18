add_subdirectory(sycl-ls)

# TODO: move each tool in its own sub-directory
add_executable(get_device_count_by_type get_device_count_by_type.cpp)
add_dependencies(get_device_count_by_type ocl-headers ocl-icd level-zero-loader)

if(MSVC)
  set(LEVEL_ZERO_LIBRARY
    "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_STATIC_LIBRARY_PREFIX}ze_loader${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  set(LEVEL_ZERO_LIBRARY
    "${LLVM_LIBRARY_OUTPUT_INTDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}ze_loader${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

if(SYCL_BUILD_PI_VULKAN)
  find_package(Vulkan 1.2.141 REQUIRED)
  target_include_directories(get_device_count_by_type PRIVATE ${Vulkan_INCLUDE_DIRS})
endif()

target_link_libraries(get_device_count_by_type
  PRIVATE
    OpenCL::Headers
    LevelZeroLoader::Headers
    ${OpenCL_LIBRARIES}
    ${LEVEL_ZERO_LIBRARY}
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:cudadrv>
    $<$<BOOL:${SYCL_BUILD_PI_VULKAN}>:Vulkan::Vulkan>
)
target_compile_definitions(get_device_count_by_type
  PRIVATE
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
    $<$<BOOL:${SYCL_BUILD_PI_VULKAN}>:USE_PI_VULKAN>
)
