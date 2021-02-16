#ifndef LLVM_TRANSFORMS_BBSORT_BBSORT_H
#define LLVM_TRANSFORMS_BBSORT_BBSORT_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class DominatorTree;

class BBSortPass : public PassInfoMixin<BBSortPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  bool runImpl(Function &F, DominatorTree &DT);
};


//===----------------------------------------------------------------------===//
//
// BBSort - Sort Basic Blocks in order of domination
//
///
///
FunctionPass *createBBSortPass();

} // namespace llvm

#endif // LLVM_TRANSFORMS_BBSORT_BBSORT_H
