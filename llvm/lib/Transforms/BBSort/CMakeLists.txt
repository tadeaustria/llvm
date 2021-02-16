add_llvm_component_library(LLVMBBSort
  BBSort.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/BBSort

  DEPENDS
  intrinsics_gen
  )