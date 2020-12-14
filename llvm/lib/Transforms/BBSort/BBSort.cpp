#include "llvm/Transforms/BBSort/BBSort.h"

#include "llvm/IR/Dominators.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

namespace llvm {
  void initializeBBSortLegacyPassPass(PassRegistry &);
}

#define DEBUG_TYPE "BBSort"

namespace {

/// The constant hoisting pass.
class BBSortLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  BBSortLegacyPass() : FunctionPass(ID) {
    initializeBBSortLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &Fn) override;

  StringRef getPassName() const override { return "BBSort"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    //AU.setPreservesCFG();
    //if (ConstHoistWithBlockFrequency)
    //  AU.addRequired<BlockFrequencyInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    //AU.addRequired<ProfileSummaryInfoWrapperPass>();
    //AU.addRequired<TargetTransformInfoWrapperPass>();
  }

private:
  BBSortPass Impl;
};

} // namespace

char BBSortLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(BBSortLegacyPass, "BBSort",
                      "BBSort", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(BBSortLegacyPass, "BBSort",
                    "BBSort", false, false)

FunctionPass *llvm::createBBSortPass() { return new BBSortLegacyPass(); }

/// Perform the constant hoisting optimization for the given function.
bool BBSortLegacyPass::runOnFunction(Function &Fn) {
  if (skipFunction(Fn))
    return false;

  return Impl.runImpl(Fn, getAnalysis<DominatorTreeWrapperPass>().getDomTree());
}

void addInDomOrder(const llvm::DomTreeNode *N, std::vector<BasicBlock *> &vec) {
  vec.push_back(N->getBlock());
  for (auto I = N->begin(), E = N->end(); I != E; ++I) {
    addInDomOrder(*I, vec);
  }
}

bool BBSortPass::runImpl(Function &F, DominatorTree &DT) {

  //constexpr bool debug = false;

  auto &dom = DT;

  LLVM_DEBUG(dbgs() << "BBSortPass::run " << F.getName() << "\n");
  LLVM_DEBUG(dom.print(llvm::dbgs()));

  std::vector<BasicBlock *> vec;
  addInDomOrder(dom.getRootNode(), vec);
  auto &bbList = F.getBasicBlockList();

  LLVM_DEBUG(dbgs() << "Order Pre\n");
  LLVM_DEBUG(for (auto &bb : F) dbgs()
             << &bb << " ---------- " << bb.getName() << "\n");
  
  std::map<const BasicBlock *, unsigned int> ordering;
  for (unsigned int i = 0; i < vec.size(); ++i) {
    ordering[vec[i]] = i;
  }

  bbList.sort([&ordering](const BasicBlock &l, const BasicBlock &r) {
    return ordering[&l] < ordering[&r];
  });

  LLVM_DEBUG(dbgs() << "Order Post\n");
  LLVM_DEBUG(for (auto &bb : F) 
      dbgs() << &bb << " ---------- " << bb.getName() << "\n");
  
  return true;
}

PreservedAnalyses BBSortPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &dom = AM.getResult<DominatorTreeAnalysis>(F);

  runImpl(F, dom);

  return PreservedAnalyses::all();
}