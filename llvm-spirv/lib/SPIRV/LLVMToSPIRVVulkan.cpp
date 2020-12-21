//===- SPIRVWriter.cpp - Converts LLVM to SPIR-V ----------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
///
//===----------------------------------------------------------------------===//

#include "LLVMToSPIRVVulkan.h"
#include "LLVMToSPIRVDbgTran.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVExtInst.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "SPIRVMDWalker.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVUtil.h"
#include "SPIRVValue.h"
#include "SPIRVWriter.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils.h" // loop-simplify pass

#include <cstdlib>
#include <functional>
#include <memory>
#include <set>
#include <vector>

#define DEBUG_TYPE "spirv"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

/// This function replaces all blocks in a given phi instruction
/// due to the map
/// @param[inout] PhiInstruction is the instruction to be changed
/// @param[in] ReplaceMap map where the to changed Basic block is the key and it
/// is replaced by the value
void replaceBBsInPhi(
    SPIRVPhi *PhiInstruction,
    std::map<SPIRVBasicBlock *, SPIRVBasicBlock *> &ReplaceMap) {
  std::vector<SPIRVValue *> newPairs;
  bool ChangeDone = false;
  PhiInstruction->foreachPair([&newPairs, &ReplaceMap, &ChangeDone](
                                  SPIRVValue *Val, SPIRVBasicBlock *BB) {
    newPairs.push_back(Val);
    if (ReplaceMap.find(BB) != ReplaceMap.end()) {
      newPairs.push_back(ReplaceMap[BB]);
      ChangeDone = true;
    } else {
      newPairs.push_back(BB);
    }
  });
  if (ChangeDone)
    PhiInstruction->setPairs(newPairs);
}

LLVMToSPIRVVulkan::LLVMToSPIRVVulkan(SPIRVModule *SMod)
    : LLVMToSPIRV(SMod), RuntimeArrayArguments(), LoopInfoObj(),
      PredecessorBBChangeMap() {}

void LLVMToSPIRVVulkan::transFunction(Function *I) {
  SPIRVFunction *BF = transFunctionDecl(I);
  // Creating all basic blocks before creating any instruction.
  for (auto &FI : *I) {
    LLVMToSPIRV::transValue(&FI, nullptr);
  }
  for (auto &FI : *I) {
    SPIRVBasicBlock *BB =
        static_cast<SPIRVBasicBlock *>(LLVMToSPIRV::transValue(&FI, nullptr));
    for (auto &BI : FI) {
      LLVMToSPIRV::transValue(&BI, BB, false);
    }
  }

  // Enable FP contraction unless proven otherwise
  joinFPContract(I, FPContract::ENABLED);
  fpContractUpdateRecursive(I, getFPContract(I));

  bool IsKernelEntryPoint =
      BF->getModule()->isEntryPoint(spv::ExecutionModelGLCompute, BF->getId());

  if (IsKernelEntryPoint) {
    collectInputOutputVariables(BF, I);
  }
}

SPIRVFunction *LLVMToSPIRVVulkan::transFunctionDecl(Function *F) {
  if (auto BF = getTranslatedValue(F))
    return static_cast<SPIRVFunction *>(BF);

  if (F->isIntrinsic()) {
    // We should not translate LLVM intrinsics as a function
    assert(none_of(F->user_begin(), F->user_end(),
                   [this](User *U) { return getTranslatedValue(U); }) &&
           "LLVM intrinsics shouldn't be called in SPIRV");
    return nullptr;
  }

  // Using analysis pass
  FunctionAnalysisManager FAM;
  auto DomAnalysis = DominatorTreeAnalysis();
  auto PostDomAnalysis = PostDominatorTreeAnalysis();
  FAM.registerPass([&] { return DomAnalysis; });
  FAM.registerPass([&] { return PostDomAnalysis; });
  auto DominatorTree = DomAnalysis.run(*F, FAM);
  PDominatorTree = PostDomAnalysis.run(*F, FAM);
  // generate the LoopInfoBase for the current function
  LoopInfoObj.releaseMemory();
  LoopInfoObj.analyze(DominatorTree);

  // PDominatorTree.print(llvm::outs());

  SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
      transType(getAnalysis<OCLTypeToSPIRV>().getAdaptedType(F)));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName())
    BM->setName(BF, F->getName().str());
  if (isKernel(F))
    BM->addEntryPoint(ExecutionModelGLCompute, BF->getId());
  // FIXME: Think about a different location for this only be done once
  // static bool Once = false;
  // if (!Once) {
  //  std::vector<SPIRVValue *> LocalSizeElements{3};
  //  auto UIntType = BM->addIntegerType(32);
  //  for (size_t i = 0; i < 3; i++) {
  //    LocalSizeElements[i] = BM->addSpecConstant(UIntType, 1);
  //    LocalSizeElements[i]->addDecorate(DecorationSpecId, 100 + i);
  //  }
  //  auto LocalSize =
  //      BM->addSpecCompositeConstant(WorkgroupSizeType, LocalSizeElements);
  //  LocalSize->addDecorate(new SPIRVDecorate(DecorationBuiltIn, LocalSize,
  //                                           BuiltInWorkgroupSize));
  //  Once = true;
  //}

  // Translate OpenCL/SYCL buffer_location metadata if it's attached to the
  // translated function declaration
  MDNode *BufferLocation = nullptr;
  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_fpga_buffer_location))
    BufferLocation = ((*F).getMetadata("kernel_arg_buffer_location"));

  // Vulkan no linkage decorations
  // else if (F->getLinkage() != GlobalValue::InternalLinkage)
  //  BF->setLinkageType(transLinkageType(F));
  auto Attrs = F->getAttributes();
  for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E;
       ++I) {
    auto ArgNo = I->getArgNo();
    SPIRVFunctionParameter *BA = BF->getArgument(ArgNo);
    if (I->hasName())
      BM->setName(BA, I->getName().str());
    // Vulkan/Shaders support no Function Parameter decorations
    // if (I->hasByValAttr())
    //  BA->addAttr(FunctionParameterAttributeByVal);
    // if (I->hasNoAliasAttr())
    //  BA->addAttr(FunctionParameterAttributeNoAlias);
    // if (I->hasNoCaptureAttr())
    //  BA->addAttr(FunctionParameterAttributeNoCapture);
    // if (I->hasStructRetAttr())
    //  BA->addAttr(FunctionParameterAttributeSret);
    // if (Attrs.hasAttribute(ArgNo + 1, Attribute::ZExt))
    //  BA->addAttr(FunctionParameterAttributeZext);
    // if (Attrs.hasAttribute(ArgNo + 1, Attribute::SExt))
    //  BA->addAttr(FunctionParameterAttributeSext);
    if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1) &&
        Attrs.hasAttribute(ArgNo + 1, Attribute::Dereferenceable))
      BA->addDecorate(DecorationMaxByteOffset,
                      Attrs.getAttribute(ArgNo + 1, Attribute::Dereferenceable)
                          .getDereferenceableBytes());
    if (BufferLocation && I->getType()->isPointerTy()) {
      // Order of integer numbers in MD node follows the order of function
      // parameters on which we shall attach the appropriate decoration. Add
      // decoration only if MD value is not negative.
      int LocID = -1;
      if (!isa<MDString>(BufferLocation->getOperand(ArgNo)) &&
          !isa<MDNode>(BufferLocation->getOperand(ArgNo)))
        LocID = getMDOperandAsInt(BufferLocation, ArgNo);
      if (LocID >= 0) {
        BM->addCapability(CapabilityFPGABufferLocationINTEL);
        BA->addDecorate(DecorationBufferLocationINTEL, LocID);
      }
    }
  }
  // if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::ZExt))
  //  BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
  // if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt))
  //  BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeSext);
  if (Attrs.hasFnAttribute("referenced-indirectly")) {
    assert(!isKernel(F) &&
           "kernel function was marked as referenced-indirectly");
    BF->addDecorate(DecorationReferencedIndirectlyINTEL);
  }

  if (BM->isAllowedToUseExtension(ExtensionID::SPV_INTEL_vector_compute))
    transVectorComputeMetadata(F);

  SPIRVDBG(dbgs() << "[transFunction] " << *F << " => ";
           spvdbgs() << *BF << '\n';)
  return BF;
}

bool LLVMToSPIRVVulkan::transAddressingMode() {
  BM->setAddressingModel(AddressingModelLogical);
  BM->addCapability(CapabilityVariablePointersStorageBuffer);
  BM->addExtension(ExtensionID::SPV_KHR_variable_pointers);
  // BM->setMemoryModel(MemoryModelGLSL450);
  // BM->setSPIRVVersion(static_cast<SPIRVWord>(VersionNumber::SPIRV_1_3));
  return true;
}

bool LLVMToSPIRVVulkan::transExecutionMode() {
  // If Workgroup Size is not needed in shader execution
  // it will not be generated. If not generated
  // generate it here, so passing workgroup size is possible
  if (!WorkgroupSizeAvailable) {
    std::vector<SPIRVValue *> LocalSizeElements{3};
    auto UIntType = BM->addIntegerType(32);
    for (size_t i = 0; i < 3; i++) {
      LocalSizeElements[i] = BM->addSpecConstant(UIntType, 1);
      LocalSizeElements[i]->addDecorate(DecorationSpecId, 100 + i);
    }
    if (!WorkgroupSizeType) {
      // Fallback if there is not already a Vector 3 Type
      // 2 equally types is not allowed...
      WorkgroupSizeType = BM->addVectorType(UIntType, 3u);
    }
    auto LocalSize =
        BM->addSpecCompositeConstant(WorkgroupSizeType, LocalSizeElements);
    LocalSize->addDecorate(
        new SPIRVDecorate(DecorationBuiltIn, LocalSize, BuiltInWorkgroupSize));
  }

  return LLVMToSPIRV::transExecutionMode();
}

SPIRVType *LLVMToSPIRVVulkan::transType(Type *T) {
  LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(dbgs() << "[transTypeVulkan] " << *T << '\n');

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    SPIRVTypeStruct *Ret;
    // if (InParameterStructure) {
    //  Ret = reinterpret_cast<SPIRVTypeStruct *>(LLVMToSPIRV::transType(T));
    //  Ret->addDecorate(DecorationBlock);
    //} else {
    Ret = reinterpret_cast<SPIRVTypeStruct *>(LLVMToSPIRV::transType(T));
    //}

    for (size_t I = 0; I < ST->getNumElements(); I++) {
      auto StructLayout = M->getDataLayout().getStructLayout(ST);
      Ret->addMemberDecorate(I, DecorationOffset,
                             StructLayout->getElementOffset(I));
    }
    return Ret;
    /*for (unsigned I = 0, E = T->getStructNumElements(); I != E; ++I) {
      auto *ElemTy = ST->getElementType(I);
      if ((isa<StructType>(ElemTy) || isa<SequentialType>(ElemTy) ||
           isa<PointerType>(ElemTy)) &&
          recursiveType(ST, ElemTy))
        ForwardRefs.push_back(I);
      else
        Struct->setMemberType(I, transType(ST->getElementType(I)));
    */
  } else if (auto Pt = dyn_cast<PointerType>(T)) {
    if (!InParameterStructure &&
        Pt->getAddressSpace() == SPIRAddressSpace::SPIRAS_StorageBuffer) {
      InParameterStructure = true;
      auto Ret =
          reinterpret_cast<SPIRVTypePointer *>(LLVMToSPIRV::transType(T));
      // if (Pt->getElementType()->isStructTy())
      //  Ret->getElementType()->addDecorate(DecorationBlock);
      InParameterStructure = false;
      return Ret;
    }
    if (InParameterStructure) {
      // if (auto subPtr = dyn_cast<PointerType>(Pt->getElementType())) {
      auto subtype = Pt->getElementType();
      /*auto subsubtype = subtype->getElementType();
      auto addrspace = SPIRSPIRVAddrSpaceMap::map(
          static_cast<SPIRAddressSpace>(Pt->getAddressSpace()));

      return mapType(
          T, BM->addPointerType(
                 addrspace, BM->addRuntimeArrayType(transType(subsubtype))));*/

      auto RtArray = BM->addRuntimeArrayType(transType(subtype));
      RtArray->addDecorate(
          DecorationArrayStride,
          M->getDataLayout().getTypeStoreSize(subtype).getFixedSize());
      /*return BM->addPointerType(SPIRSPIRVAddrSpaceMap::map(
          static_cast<SPIRAddressSpace>(Pt->getAddressSpace())), RtArray);*/
      return RtArray;
    }
  } else if (auto Ar = dyn_cast<ArrayType>(T)) {
    auto NewArr = LLVMToSPIRV::transType(T);
    NewArr->addDecorate(DecorationArrayStride,
                        M->getDataLayout()
                            .getTypeStoreSize(Ar->getArrayElementType())
                            .getFixedSize());
    return NewArr;
  } else if (auto *VecTy = dyn_cast<VectorType>(T)) {
    // Store special Vec3 UInt Type for Workgroup Constant
    if (VecTy->getElementCount().getValue() == 3u &&
        VecTy->getElementType()->isIntegerTy()) {
      return WorkgroupSizeType = LLVMToSPIRV::transType(T);
    }
  }

  return LLVMToSPIRV::transType(T);
}

/// This function traces back all GEP and Load commands
/// and stacks the indices
Value *backtrace(Value *V, std::stack<std::vector<Value *>> &indices) {
  if (auto GEP = dyn_cast<GetElementPtrInst>(V)) {
    indices.emplace();
    for (size_t i = 1; i < GEP->getNumOperands(); i++) {
      indices.top().push_back(GEP->getOperand(i));
    }
    return backtrace(GEP->getPointerOperand(), indices);
  } else if (auto Load = dyn_cast<LoadInst>(V)) {
    return backtrace(Load->getPointerOperand(), indices);
  }
  return V;
}

/// This functions gives the origin of a GEP, where
/// origin will be a pointer to the outermost datastructure.
/// Indices will be relevant indices
Value *backtrace(Value *V, std::vector<Value *> &indices,
                 std::vector<Value *> &origindices) {

  std::stack<std::vector<Value *>> IndicesStack;
  // Use helper to find all the indices used
  Value *base = backtrace(V, IndicesStack);

  // Traverse the indices backwards to find
  // the 2nd layer which will be the access within a runtime array
  // Recall that the first index in a LLVM GEP is not a
  // navigation into the structure, it only resolves the given
  // base pointer
  // Seperate 2nd layer accesses from the rest
  int level = 0;
  while (!IndicesStack.empty()) {
    auto &SomeIndices = IndicesStack.top();
    for (auto Index : SomeIndices) {
      if (level == 2) {
        indices.push_back(Index);
      } else {
        origindices.push_back(Index);
      }
      level += SomeIndices.size() - 1;
    }
    IndicesStack.pop();
  }
  return base;
}

/// An instruction may use an instruction from another BB which has not been
/// translated. SPIRVForward should be created as place holder for these
/// instructions and replaced later by the real instructions.
/// Use CreateForward = true to indicate such situation.
SPIRVValue *
LLVMToSPIRVVulkan::transValueWithoutDecoration(Value *V, SPIRVBasicBlock *BB,
                                               bool CreateForward,
                                               FuncTransMode FuncTrans) {

  auto containsRTArray = [&](Value *V) {
    auto TranslatedStructType =
        transType(V->getType())->getPointerElementType();
    if (TranslatedStructType->isTypeStruct()) {
      for (SPIRVWord i = 0; i < TranslatedStructType->getStructMemberCount();
           i++) {
        if (TranslatedStructType->getStructMemberType(i)->isTypeRuntimeArray())
          return true;
      }
    }
    return false;
  };

  SPIRVValue *Alternative = nullptr;
  if (isSkippable(V, BB, &Alternative)) {
    if (Alternative)
      return mapValue(V, Alternative);
    SPIRVDBG(dbgs() << "[skipped] " << '\n')
    return Alternative;
  }

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {

    // Special case for BuiltIn variable
    spv::BuiltIn Builtin = spv::BuiltInPosition;
    if (GV->hasName() && getSPIRVBuiltin(GV->getName().str(), Builtin)) {
      // Find Special BuiltinVariable
      switch (Builtin) {
      case spv::BuiltInWorkgroupSize: {
        WorkgroupSizeAvailable = true;
        std::vector<SPIRVValue *> LocalSizeElements{3};
        auto UIntType = BM->addIntegerType(32);
        for (size_t i = 0; i < 3; i++) {
          LocalSizeElements[i] = BM->addSpecConstant(UIntType, 1);
          LocalSizeElements[i]->addDecorate(DecorationSpecId, 100 + i);
        }
        auto LocalSize = BM->addSpecCompositeConstant(
            transType(GV->getType()->getElementType()), LocalSizeElements);
        LocalSize->addDecorate(
            new SPIRVDecorate(DecorationBuiltIn, LocalSize, Builtin));
        mapValue(V, LocalSize);
        return LocalSize;
      }
      case spv::BuiltInGlobalOffset: {
        std::vector<SPIRVValue *> GlobalOffsetElements{3};
        auto UIntType = BM->addIntegerType(32);
        for (size_t i = 0; i < 3; i++) {
          GlobalOffsetElements[i] = BM->addSpecConstant(UIntType, 1);
          GlobalOffsetElements[i]->addDecorate(DecorationSpecId, 103 + i);
        }
        auto GlobalOffset = BM->addSpecCompositeConstant(
            transType(GV->getType()->getElementType()), GlobalOffsetElements);
        mapValue(V, GlobalOffset);
        return GlobalOffset;
      }
      default:
        break;
      }
    }

    // Descriptor & Binding Decoration for input structures needed
    auto TransValue = LLVMToSPIRV::transValueWithoutDecoration(
        V, BB, CreateForward, FuncTrans);
    const char *ArgName = "_arg_";
    auto Pos = GV->getName().find(ArgName);
    if (Pos != std::string::npos) {
      auto IDcString = GV->getName().data() + Pos + strlen(ArgName);
      auto ID = std::atoi(IDcString);
      TransValue->addDecorate(DecorationDescriptorSet, 0);
      TransValue->addDecorate(DecorationBinding, ID);
      reinterpret_cast<SPIRVTypePointer *>(TransValue->getType())
          ->getElementType()
          ->addDecorate(DecorationBlock);
    }
    if (containsRTArray(V)) {
      RuntimeArrayArguments.push_back(V);
    }
    return TransValue;
  }

  if (CreateForward)
    return LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward,
                                                    FuncTrans);

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    std::vector<SPIRVValue *> Indices;
    for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
      Indices.push_back(LLVMToSPIRV::transValue(GEP->getOperand(I + 1), BB));
    auto *TransPointerOperand =
        LLVMToSPIRV::transValue(GEP->getPointerOperand(), BB);
    // Certain array-related optimization hints can be expressed via
    // LLVM metadata. For the purpose of linking this metadata with
    // the accessed array variables, our GEP may have been marked into
    // a so-called index group, an MDNode by itself.
    if (MDNode *IndexGroup = GEP->getMetadata("llvm.index.group")) {
      SPIRVId AccessedArrayId = TransPointerOperand->getId();
      unsigned NumOperands = IndexGroup->getNumOperands();
      // When we're working with embedded loops, it's natural that
      // the outer loop's hints apply to all code contained within.
      // The inner loop's specific hints, however, should stay private
      // to the inner loop's scope.
      // Consequently, the following division of the index group metadata
      // nodes emerges:

      // 1) The metadata node has no operands. It will be directly referenced
      //    from within the optimization hint metadata.
      if (NumOperands == 0)
        IndexGroupArrayMap[IndexGroup].insert(AccessedArrayId);
      // 2) The metadata node has several operands. It serves to link an index
      //    group specific to some embedded loop with other index groups that
      //    mark the same array variable for the outer loop(s).
      for (unsigned I = 0; I < NumOperands; ++I) {
        auto *ContainedIndexGroup = getMDOperandAsMDNode(IndexGroup, I);
        IndexGroupArrayMap[ContainedIndexGroup].insert(AccessedArrayId);
      }
    }

    // Backtrace the original data object of the GEP
    std::vector<Value *> OtherIndices;
    std::vector<Value *> OrigIndices;
    Value *originGEP = backtrace(GEP, OtherIndices, OrigIndices);

    // If the origin is a data structure containing a runtime array it
    // needs special handling
    if (std::find(RuntimeArrayArguments.begin(), RuntimeArrayArguments.end(),
                  originGEP) != RuntimeArrayArguments.end()) {

      // Map all other than array accesses into translated data structure
      std::vector<SPIRVValue *> MyIndices;
      for (auto OrgIndex = OrigIndices.begin() + 1;
           OrgIndex != OrigIndices.end(); OrgIndex++) {
        MyIndices.push_back(LLVMToSPIRV::transValue(*OrgIndex, BB));
      }

      // Aggregate the array accesses indices to one index and insert this
      // index to the second position
      if (OtherIndices.size() == 1) {
        // Here is no need for aggregation
        MyIndices.insert(MyIndices.begin() + 1,
                         LLVMToSPIRV::transValue(OtherIndices[0], BB));
        return mapValue(
            V, BM->addAccessChainInst(transType(GEP->getType()),
                                      LLVMToSPIRV::transValue(originGEP, BB),
                                      MyIndices, BB, GEP->isInBounds()));
      } else if (OtherIndices.size() > 1) {
        // Create add instructions for index aggregation
        auto addRes =
            BM->addBinaryInst(OpIAdd, transType(OtherIndices[0]->getType()),
                              LLVMToSPIRV::transValue(OtherIndices[0], BB),
                              LLVMToSPIRV::transValue(OtherIndices[1], BB), BB);
        for (size_t i = 2; i < OtherIndices.size(); i++) {
          addRes = BM->addBinaryInst(
              OpIAdd, addRes->getType(), addRes,
              LLVMToSPIRV::transValue(OtherIndices[i], BB), BB);
        }
        MyIndices.insert(MyIndices.begin() + 1, addRes);
        return mapValue(
            V, BM->addAccessChainInst(transType(GEP->getType()),
                                      LLVMToSPIRV::transValue(originGEP, BB),
                                      MyIndices, BB, GEP->isInBounds()));
      }
    }

    auto TypePointer = cast<PointerType>(GEP->getPointerOperand()->getType());
    if (TypePointer->getElementType()->isStructTy()) {

      auto Type = GEP->getType();
      if (TypePointer->getElementType()->getStructName().find("union") !=
              std::string::npos ||
          TypePointer->getElementType()->getStructName().find("arg") !=
              std::string::npos) {
        auto accessedType =
            TypePointer->getElementType()->getStructElementType(0);
        if (accessedType->isPointerTy()) {
          // Access to RuntimeArray?
          // If yes, add another index to access first element within the
          // runtime array to get a element pointer
          Indices.push_back(
              BM->addConstant(transType(GEP->getOperand(1)->getType()), 0));
          // This additional index also changes the type, which is then
          // one type step down the pointer chain
          Type = Type->getPointerElementType();
        }
        //}
      }
      // Remove First Index due to difference in AccessChain and PtrAccessChain
      // The first index in PtrAccessChain is dereferencing the Pointer, this is
      // already implicit in AccessChain
      Indices.erase(Indices.begin());
      return mapValue(V, BM->addAccessChainInst(transType(Type),
                                                TransPointerOperand, Indices,
                                                BB, GEP->isInBounds()));
    } else {
      return mapValue(
          V, BM->addPtrAccessChainInst(
                 transType(GEP->getType()), TransPointerOperand, Indices, BB,
                 /*isInbounds*/ false)); // Vulkan does not allow
                                         // PtrAccessInbounds, since Address
                                         // Capability is not supported
    }
  }
  if (LoadInst *LD = dyn_cast<LoadInst>(V)) {
    auto TargetTy = LD->getType();
    auto Source = LD->getPointerOperand();
    if (isBuiltIn(Source, spv::BuiltInWorkgroupSize) ||
        isBuiltIn(Source, spv::BuiltInGlobalOffset)) {
      // Is already loaded, do NOOP
      return mapValue(V, LLVMToSPIRV::transValue(Source, BB));
    }
    if (auto GEP = dyn_cast<GEPOperator>(Source)) {
      auto SourceBaseTy =
          GEP->getPointerOperand()->getType()->getPointerElementType();
      if (TargetTy->isPointerTy() &&
          TargetTy->getPointerAddressSpace() == SPIRAS_StorageBuffer &&
          SourceBaseTy->isStructTy() &&
          SourceBaseTy->getStructName().find("_arg_") != std::string::npos) {
        // Well this is a special case, for the data pointer
        return mapValue(V, LLVMToSPIRV::transValue(Source, BB));
      }
    } else if (isa<BitCastInst>(Source)) {
      // Actually should be a noop
      // LLVM again, optimizes first member accesses
      // and if they should be copied, both Source and Target
      // will be accessed by integer pointer and be copied
      // (which does not work in SPIRV for vulkan because it cannot work with
      // casted pointers)
      return mapValue(V, LLVMToSPIRV::transValue(Source, BB));
    }
    std::vector<SPIRVWord> MemoryAccess(1, 0);
    return mapValue(V, BM->addLoadInst(LLVMToSPIRV::transValue(Source, BB),
                                       MemoryAccess, BB));
  }

  if (StoreInst *SD = dyn_cast<StoreInst>(V)) {
    // backtrace(SD->getPointerOperand(), SD->getValueOperand());
    std::vector<llvm::Value *> Ind;
    std::vector<llvm::Value *> OrgInd;

    auto Base = SD->getPointerOperand();
    while (dyn_cast<BitCastInst>(Base)) {
      Base = dyn_cast<BitCastInst>(Base)->getOperand(0);
    }

    auto Load = dyn_cast<LoadInst>(SD->getValueOperand());
    if (Load) {
      auto SourceBase = Load->getOperand(0);
      while (dyn_cast<BitCastInst>(SourceBase)) {
        SourceBase = dyn_cast<BitCastInst>(SourceBase)->getOperand(0);
      }

      if (SourceBase->getType()->getPointerElementType()->isStructTy() &&
          !Base->getType()->getPointerElementType()->isStructTy()) {
        SPIRV::ValueVec Vector;
        std::vector<SPIRV::SPIRVWord> MemoryAccess;
        Vector.push_back(BM->getLiteralAsConstant(0));
        auto Source = BM->addAccessChainInst(
            transType(Base->getType()), LLVMToSPIRV::transValue(SourceBase, BB),
            Vector, BB, true);
        return mapValue(
            V, BM->addCopyMemoryInst(Source, LLVMToSPIRV::transValue(Base, BB),
                                     MemoryAccess, BB));
      }
    }
  }

  if (BranchInst *Branch = dyn_cast<BranchInst>(V)) {
    std::vector<SPIRVWord> Parameters;
    spv::LoopControlMask LoopControl = getLoopControl(Branch, Parameters);

    // Somehow the logic in SPIRVWriter.cpp is not working properly.
    // It seems valid but somehow it is impossible to add
    // loop meta data to the branch instruction and therefore
    // it will not be converted into an structured CFG in SPIRV
    // This "fix" instead uses for for-loop increment block
    // by simple label name match. This then has the back edge
    // and uses the same logic as original code

    // Look if actual Basic block is a Continue Block which got an extra
    // continue exit.
    if (PredecessorBBChangeMap.find(BB) != PredecessorBBChangeMap.end()) {
      // If so, add the branch instruction to the additional block
      // and branch from the actual block simple to the additional block
      auto AdditionalContinueBlock = PredecessorBBChangeMap[BB];
      auto ret = LLVMToSPIRV::transValueWithoutDecoration(
          V, AdditionalContinueBlock, CreateForward, FuncTrans);
      BM->addBranchInst(AdditionalContinueBlock, BB);
      return ret;
    }

    if (Branch->isUnconditional()) {
      auto Loop = LoopInfoObj.getLoopFor(Branch->getParent());
      auto BranchTranslated =
          LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward);
      // Even though it is a unconditional branch, it is possible
      // that the actual block is the header (target of a back edge) of
      // some loop (e.g. do - while loop). If it is a header
      // a merge instruction is needed
      if (Loop && Loop->getHeader() == Branch->getParent() &&
          Loop->getUniqueExitBlock()) {
        auto Continue = static_cast<SPIRVLabel *>(
            LLVMToSPIRV::transValue(Loop->getLoopLatch(), BB));
        auto Merge = static_cast<SPIRVLabel *>(
            LLVMToSPIRV::transValue(Loop->getUniqueExitBlock(), BB));
        BM->addLoopMergeInst(Merge->getId(),    // Merge Block
                             Continue->getId(), // Continue Target
                             LoopControl, Parameters, BB);
      }
      return BranchTranslated;
    }

    if (Branch->isConditional()) {
      auto Loop = LoopInfoObj.getLoopFor(Branch->getParent());
      if (Loop && Loop->getHeader() == Branch->getParent() &&
          Loop->getUniqueExitBlock()) {
        // actual block is actually a loop block
        // however, it may be possible that a loop header
        // ends with an if branching
        auto Continue = static_cast<SPIRVLabel *>(
            LLVMToSPIRV::transValue(Loop->getLoopLatch(), BB));
        auto Merge = static_cast<SPIRVLabel *>(
            LLVMToSPIRV::transValue(Loop->getUniqueExitBlock(), BB));

        // Hard insert an extra block after the original ContinueBlock
        // This prevents that the continue block is also a merge block
        // of an if. This is not allowed in vulkan SPIRV
        // Do only if continue block is not itself <- small loop
        auto BBNewContinue = BB;
        if (Continue != BB) {
          static unsigned int j = 0;
          std::string NewName("ContinueBlock");
          NewName.append(std::to_string(j++));
          BBNewContinue = BM->insertBasicBlockAfter(BB->getParent(), Continue);
          PredecessorBBChangeMap[Continue] = BBNewContinue;
        }

        // After inserting the new Continue Block, adapt possible phi
        // instructions in the header block to the new continue block
        auto Instruction = BB->getInst(0);
        while (Instruction->getOpCode() == Op::OpPhi) {
          auto PhiInstruction = static_cast<SPIRVPhi *>(Instruction);
          replaceBBsInPhi(PhiInstruction, PredecessorBBChangeMap);
          Instruction = BB->getNext(Instruction);
        }

        SPIRVValue *BranchTranslated = nullptr;
        // If neither of the successors of the loop header is the
        // exit block, it must be an if
        if (Branch->getSuccessor(0) != Loop->getUniqueExitBlock() &&
            Branch->getSuccessor(1) != Loop->getUniqueExitBlock()) {
          // in this case, add additional Block for if header due to spirv
          // specification
          static unsigned int i = 0;
          std::string NewName("InsertBlock");
          NewName.append(std::to_string(i++));
          auto BBIfHeader = BM->insertBasicBlockAfter(BB->getParent(), BB);
          BM->setName(BBIfHeader, NewName.c_str());
          BM->addBranchInst(BBIfHeader, BB);
          // Add Block to ChangeMap so future generated Phi instructions
          // will use the inserted header instead of the original
          PredecessorBBChangeMap[BB] = BBIfHeader;

          // Find merge block of if
          if (auto Dominator = PDominatorTree.findNearestCommonDominator(
                  Branch->getSuccessor(0), Branch->getSuccessor(1))) {
            BM->addSelectionMergeInst(
                LLVMToSPIRV::transValue(Dominator, BBIfHeader)->getId(),
                /*SelectionControl None*/ 0, BBIfHeader);
          } else {
            assert(false && "No Common Dominator for Branch found");
          }

          BranchTranslated = LLVMToSPIRV::transValueWithoutDecoration(
              V, BBIfHeader, CreateForward);
        } else {
          // When adding LoopMergeInst the branch instruction must be already
          // terminator instruction of the block, therefore emit this before
          BranchTranslated =
              LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward);
        }
        // Finally add merge instruction for loop
        BM->addLoopMergeInst(Merge->getId(),         // Merge Block
                             BBNewContinue->getId(), // Continue Target
                             LoopControl, Parameters, BB);
        return BranchTranslated;
      } else {
        // Emit selection merge instruction, if either not in loop
        // or be sure that it is not the loop branching, by looking for loop
        // exit block
        if (!Loop || (Branch->getSuccessor(0) != Loop->getUniqueExitBlock() &&
                      Branch->getSuccessor(1) != Loop->getUniqueExitBlock())) {
          if (auto Dominator = PDominatorTree.findNearestCommonDominator(
                  Branch->getSuccessor(0), Branch->getSuccessor(1))) {
            BM->addSelectionMergeInst(
                LLVMToSPIRV::transValue(Dominator, BB)->getId(),
                /*SelectionControl None*/ 0, BB);
          } else {
            assert(false && "No Common Dominator for Branch found");
          }
        }
      }
    }
  }

  if (auto Phi = dyn_cast<PHINode>(V)) {
    // Translate phi instruction
    auto TranslatedPhi =
        static_cast<SPIRVPhi *>(LLVMToSPIRV::transValueWithoutDecoration(
            Phi, BB, CreateForward, FuncTrans));
    // Replace possible income blocks, where a additional block was inserted
    replaceBBsInPhi(TranslatedPhi, PredecessorBBChangeMap);
    return TranslatedPhi;
  }

  // This was a test to fix OpAll but it does not work in OpenCL either
  // if (CallInst *Call = dyn_cast<CallInst>(V)) {
  //  StringRef DemangledName;
  //  if (oclIsBuiltin(Call->getFunction()->getName(), DemangledName) &&
  //      getSPIRVFuncOC(DemangledName) == OpAll) {
  //    if (SelectInst *Select = dyn_cast<SelectInst>(Call->getArgOperand(0))) {
  //      return mapValue(
  //          V, BM->addUnaryInst(
  //                 OpAll, transType(Call->getType()),
  //                 LLVMToSPIRV::transValue(Select->getOperand(0), BB), BB));
  //    }
  //  }
  //  // if (Call->getCalledFunction()->is )
  //}

  return LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward,
                                                  FuncTrans);
}

std::vector<SPIRVWord> LLVMToSPIRVVulkan::transValue(
    const std::vector<Value *> &Args, SPIRVBasicBlock *BB, SPIRVEntry *Entry,
    std::vector<std::pair<SPIRVValue *, SPIRVValue *>> &CopyBack) {
  std::vector<SPIRVWord> Operands;
  for (size_t I = 0, E = Args.size(); I != E; ++I) {
    if (Entry->isOperandLiteral(I)) {
      Operands.push_back(cast<ConstantInt>(Args[I])->getZExtValue());
    } else {
      auto Value = LLVMToSPIRV::transValue(Args[I], BB);
      if (Value->isVariable() || !Value->getType()->isTypePointer() ||
          Value->getType()->getPointerStorageClass() ==
              SPIRVStorageClassKind::StorageClassStorageBuffer) {
        Operands.push_back(Value->getId());
      } else {
        // Function calls only allow Memory Object Declaration, see
        // https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#MemoryObjectDeclaration
        // if operand is non Object, create local variable and use it instead
        // but copy back is needed at the end

        // Theoretically also possible for other storage classes
        // but then type has to be adapted of the created variable
        // and can not be directly used
        assert(Value->getType()->getPointerStorageClass() ==
                   SPIRVStorageClassKind::StorageClassFunction &&
               "For now works only for local variables");

        auto NewValue = BM->addVariable(
            Value->getType(), false, internal::LinkageTypeInternal,
            /*Initializer*/ nullptr, /*Name*/ "",
            SPIRVStorageClassKind::StorageClassFunction, BB);
        BM->addCopyMemoryInst(NewValue, Value,
                              std::vector<SPIRVWord>(1, MemoryAccessMaskNone),
                              BB);
        CopyBack.emplace_back(Value, NewValue);
        Operands.push_back(NewValue->getId());
      }
    }
  }
  return Operands;
}

SPIRVValue *LLVMToSPIRVVulkan::transDirectCallInst(CallInst *CI,
                                                   SPIRVBasicBlock *BB) {
  SPIRVExtInstSetKind ExtSetKind = SPIRVEIS_Count;
  SPIRVWord ExtOp = SPIRVWORD_MAX;
  llvm::Function *F = CI->getCalledFunction();
  auto MangledName = F->getName();
  StringRef DemangledName;

  if (MangledName.startswith(SPCV_CAST) || MangledName == SAMPLER_INIT)
    return oclTransSpvcCastSampler(CI, BB);

  if (oclIsBuiltin(MangledName, DemangledName) ||
      isDecoratedSPIRVFunc(F, DemangledName)) {
    if (auto BV = transBuiltinToConstant(DemangledName, CI))
      return BV;
    if (auto BV = transBuiltinToInst(DemangledName, CI, BB))
      return BV;
  }
  std::vector<std::pair<SPIRVValue *, SPIRVValue *>> CopyBack;
  SmallVector<std::string, 2> Dec;
  if (isBuiltinTransToExtInst(CI->getCalledFunction(), &ExtSetKind, &ExtOp,
                              &Dec))
    // TODO: Think about also copy back for this one...
    return addDecorations(
        BM->addExtInst(
            transType(CI->getType()), BM->getExtInstSetId(ExtSetKind), ExtOp,
            transArguments(CI, BB,
                           SPIRVEntry::createUnique(ExtSetKind, ExtOp).get()),
            BB),
        Dec);

  auto CallInst = BM->addCallInst(
      transFunctionDecl(CI->getCalledFunction()),
      transValue(getArguments(CI), BB,
                 SPIRVEntry::createUnique(OpFunctionCall).get(), CopyBack),
      BB);
  for (auto &Pair : CopyBack) {
    BM->addCopyMemoryInst(Pair.first, Pair.second,
                          std::vector<SPIRVWord>(1, MemoryAccessMaskNone), BB);
  }
  return CallInst;
}

SPIRVInstruction *LLVMToSPIRVVulkan::transUnaryInst(UnaryInstruction *U,
                                                    SPIRVBasicBlock *BB) {

  Op BOC = OpNop;
  if (dyn_cast<AddrSpaceCastInst>(U)) {
    // Do noop and return translated value of the first operand
    return reinterpret_cast<SPIRVInstruction *>(
        getTranslatedValue(U->getOperand(0)));
  } else {
    auto OpCode = U->getOpcode();
    BOC = OpCodeMap::map(OpCode);
  }

  auto Op = LLVMToSPIRV::transValue(U->getOperand(0), BB);
  return BM->addUnaryInst(transBoolOpCode(Op, BOC), transType(U->getType()), Op,
                          BB);
}

SPIRVValue *LLVMToSPIRVVulkan::transOrigin(Value *V, SPIRVBasicBlock *BB) {

  // Follow the Value back to the original value, without any
  // bit- or addressspacecasts
  if (auto Cast = dyn_cast<CastInst>(V)) {
    return transOrigin(Cast->getOperand(0), BB);
  }

  return LLVMToSPIRV::transValue(V, BB);
}

SPIRVValue *LLVMToSPIRVVulkan::transIntrinsicInst(IntrinsicInst *II,
                                                  SPIRVBasicBlock *BB) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::memcpy: {
    // Memcopy in LLVM is quite stupid. It seems copying works only with
    // integer8 pointers (char pointers) as target and source. If a more complex
    // object is copied, it is casted to integer pointers and memory length is
    // hardcoded depending on the object. But SPIRV has the possibility to copy
    // complex objects by type without any length operand.
    // But since LLVM already has casted the object we need to find the original
    // object
    // TODO: Cast expressions already emitted, this unnecessary code should be
    // terminted
    auto Target = transOrigin(II->getOperand(0), BB);
    auto Source = transOrigin(II->getOperand(1), BB);

    if (Target->getType()->getPointerElementType() !=
        Source->getType()->getPointerElementType()) {
      // Yet again, LLVM does unwanted optimizations when working with
      // structures If the first element of a structure is a operand in copying,
      // they skip accessing the first element and directly use the struct
      // pointer as first element pointer. Because of this, we have to cross
      // look if source/target types are different if they match the
      // first element in their structure with the other type
      // if so add an extra accessing operation
      SPIRVType *Target1Type = nullptr;
      SPIRVType *Source1Type = nullptr;
      if (Target->getType()->getPointerElementType()->isTypeStruct())
        Target1Type =
            Target->getType()->getPointerElementType()->getStructMemberType(0);
      if (Source->getType()->getPointerElementType()->isTypeStruct())
        Source1Type =
            Source->getType()->getPointerElementType()->getStructMemberType(0);

      // Well we need to navigate into both copying operands and find
      // at which depth they share the same type
      std::vector<SPIRVValue *> IndicesSource;
      IndicesSource.push_back(BM->getLiteralAsConstant(0));
      std::vector<SPIRVValue *> IndicesTarget;
      IndicesTarget.push_back(BM->getLiteralAsConstant(0));

      while (Target1Type != Source->getType()->getPointerElementType() &&
             Source1Type != Target->getType()->getPointerElementType() &&
             Target1Type != Source1Type) {
        if (Target1Type->isTypeStruct()) {
          Target1Type = Target1Type->getStructMemberType(0);
          IndicesTarget.push_back(BM->getLiteralAsConstant(0));
        } else if (Source1Type->isTypeStruct()) {
          Source1Type = Source1Type->getStructMemberType(0);
          IndicesSource.push_back(BM->getLiteralAsConstant(0));
        } else {
          assert(false && "Can't find valid data types for copying");
        }
      }

      // If we found a depth we need to generate the accesses to
      // the depth respectivly for both operands and replace the
      // operands them for the original copy operands
      if (Target1Type == Source->getType()->getPointerElementType() ||
          Target1Type == Source1Type) {
        Target = BM->addAccessChainInst(
            BM->addPointerType(Target->getType()->getPointerStorageClass(),
                               Target1Type),
            Target, IndicesTarget, BB, true);
      }
      if (Source1Type == Target->getType()->getPointerElementType() ||
          Target1Type == Source1Type) {
        Source = BM->addAccessChainInst(
            BM->addPointerType(Source->getType()->getPointerStorageClass(),
                               Source1Type),
            Source, IndicesSource, BB, true);
      }
    }

    assert(Target->getType()->getPointerElementType() ==
               Source->getType()->getPointerElementType() &&
           "Unequal types for memory copy");

    return BM->addCopyMemoryInst(
        Target, Source, GetIntrinsicMemoryAccess(cast<MemIntrinsic>(II)), BB);
  }
  case Intrinsic::memset: {
    // Generally there is no direct mapping of memset to SPIR-V.  But it turns
    // out that memset is emitted by Clang for initialization in default
    // constructors so we need some basic support.  The code below only handles
    // cases with constant value and constant length.
    MemSetInst *MSI = cast<MemSetInst>(II);
    Value *Val = MSI->getValue();
    if (!isa<Constant>(Val)) {
      assert(!"Can't translate llvm.memset with non-const `value` argument");
      return nullptr;
    }
    Value *Len = MSI->getLength();
    if (!isa<ConstantInt>(Len)) {
      assert(!"Can't translate llvm.memset with non-const `length` argument");
      return nullptr;
    }
    uint64_t NumElements = static_cast<ConstantInt *>(Len)->getZExtValue();
    auto *AT = ArrayType::get(Val->getType(), NumElements);
    SPIRVTypeArray *CompositeTy = static_cast<SPIRVTypeArray *>(transType(AT));
    SPIRVValue *Init;
    if (cast<Constant>(Val)->isZeroValue()) {
      Init = BM->addNullConstant(CompositeTy);
    } else {
      // On 32-bit systems, size_type of std::vector is not a 64-bit type. Let's
      // assume that we won't encounter memset for more than 2^32 elements and
      // insert explicit cast to avoid possible warning/error about narrowing
      // conversion
      auto TNumElts =
          static_cast<std::vector<SPIRVValue *>::size_type>(NumElements);
      std::vector<SPIRVValue *> Elts(TNumElts,
                                     LLVMToSPIRV::transValue(Val, BB));
      Init = BM->addCompositeConstant(CompositeTy, Elts);
    }
    SPIRVType *VarTy = transType(PointerType::get(AT, SPIRV::SPIRAS_Private));
    SPIRVValue *Var = BM->addVariable(VarTy, /*isConstant*/ true,
                                      internal::LinkageTypeInternal, Init,
                                      "Test", StorageClassFunction, BB);
    SPIRVType *SourceTy =
        transType(PointerType::get(Val->getType(), SPIRV::SPIRAS_Private));
    SPIRVValue *Source = BM->addUnaryInst(OpBitcast, SourceTy, Var, BB);
    SPIRVValue *Target = LLVMToSPIRV::transValue(MSI->getRawDest(), BB);
    return BM->addCopyMemoryInst(Target, Source, GetIntrinsicMemoryAccess(MSI),
                                 BB);
  } break;

  default:
    return LLVMToSPIRV::transIntrinsicInst(II, BB);
  }
}

bool LLVMToSPIRVVulkan::transDecoration(Value *V, SPIRVValue *BV) {
  return LLVMToSPIRV::transDecoration(V, BV);
}

// Vulkan allows no allignment decoration (Cap Kernel only)
bool LLVMToSPIRVVulkan::transAlign(Value *V, SPIRVValue *BV) { return true; }

// Vulkan allows not importing OpenCL instruction sets
bool LLVMToSPIRVVulkan::transBuiltinSet() { return true; }

// Vulkan only allows internal linkage, since Capability Linkage is not
// supported
SPIRVLinkageTypeKind
LLVMToSPIRVVulkan::transLinkageType(const GlobalValue *GV) {
  return internal::LinkageTypeInternal;
}

bool LLVMToSPIRVVulkan::isSkippable(Value *V, SPIRVBasicBlock *BB,
                                    SPIRVValue **Alternative) {
  // if (auto allocA = dyn_cast<AllocaInst>(V)) {
  //  size_t find;
  //  if ((find = allocA->getName().find(".addr")) != std::string::npos) {
  //    if (!allocA->getType()->isPointerTy()) {
  //      auto originName = allocA->getName().substr(0, find);
  //      auto result =
  //          std::find_if(ValueMap.begin(), ValueMap.end(), [&](auto &pair) {
  //            return pair.first->getName().compare(originName) == 0;
  //          });
  //      *alternative = result->second;
  //    }
  //    return true;
  //  } else {
  //    return false;
  //  }
  //}
  if (auto BitCast = dyn_cast<BitCastInst>(V)) {
    if (BitCast->getType()->isPointerTy()) {

      auto Elementtype = BitCast->getType()->getPointerElementType();
      if (Elementtype->isIntegerTy() && Elementtype->getIntegerBitWidth() == 8)
        return true;
      return false;
    }
  }

  if (auto IntrinicI = dyn_cast<IntrinsicInst>(V)) {
    // Skip Lifetime intrinsic, because not allowed for shaders
    if (IntrinicI->isLifetimeStartOrEnd())
      return true;

    switch (IntrinicI->getIntrinsicID()) {
    case Intrinsic::dbg_declare:
      return true;
    case Intrinsic::dbg_value:
      return true;
    default:
      return false;
    }
  }

  // if (auto store = dyn_cast<StoreInst>(V)) {
  //  for (auto &op : store->operands()) {
  //    if (op->getName().find(".addr") != std::string::npos)
  //      return true;
  //  }
  //  return false;
  //}
  // if (auto load = dyn_cast<LoadInst>(V)) {
  //  for (auto &op : load->operands()) {
  //    size_t find;
  //    if ((find = op->getName().find(".addr")) != std::string::npos) {
  //      auto originName = load->getName().substr(0, find);
  //      auto result =
  //          std::find_if(ValueMap.begin(), ValueMap.end(), [&](auto &pair) {
  //            return pair.first->getName().compare(originName) == 0;
  //          });
  //      *alternative = result->second;
  //      return true;
  //    }
  //  }
  //  return false;
  //}
  // if (auto copy = dyn_cast<MemCpyInst>(V)) {
  //  // Hm... seems a bit harsh...
  //  return true;
  //}
  // if (auto opt = dyn_cast<GetElementPtrInst>(V)) {
  //  return opt->getPointerOperand()->getName().find(".addr") !=
  //         std::string::npos;
  //}
  return false;
}

bool LLVMToSPIRVVulkan::isBuiltIn(Value *V, spv::BuiltIn builtIn) {
  spv::BuiltIn Builtin = spv::BuiltInPosition;
  if (!V->hasName() || !getSPIRVBuiltin(V->getName().str(), Builtin)) {
    return false;
  }
  return Builtin == builtIn;
}

// Copied from LLVMToSPIRV::translate
// Only without Debug 
bool LLVMToSPIRVVulkan::translate() {
  BM->setGeneratorVer(KTranslatorVer);

  if (isEmptyLLVMModule(M))
    BM->addCapability(CapabilityLinkage);

  if (!transSourceLanguage())
    return false;
  if (!transExtension())
    return false;
  if (!transBuiltinSet())
    return false;
  if (!transAddressingMode())
    return false;
  if (!transGlobalVariables())
    return false;

  for (auto &F : *M) {
    auto FT = F.getFunctionType();
    std::map<unsigned, Type *> ChangedType;
    oclGetMutatedArgumentTypesByBuiltin(FT, ChangedType, &F);
    mutateFuncArgType(ChangedType, &F);
  }

  // SPIR-V logical layout requires all function declarations go before
  // function definitions.
  std::vector<Function *> Decls, Defs;
  for (auto &F : *M) {
    if (isBuiltinTransToInst(&F) || isBuiltinTransToExtInst(&F) ||
        F.getName().startswith(SPCV_CAST) ||
        F.getName().startswith(LLVM_MEMCPY) ||
        F.getName().startswith(SAMPLER_INIT))
      continue;
    if (F.isDeclaration())
      Decls.push_back(&F);
    else
      Defs.push_back(&F);
  }
  for (auto I : Decls)
    transFunctionDecl(I);
  for (auto I : Defs)
    transFunction(I);

  if (!transMetadata())
    return false;
  if (!transExecutionMode())
    return false;

  BM->resolveUnknownStructFields();
  BM->createForwardPointers();
  // No debug info for Vulkan
  // DbgTran->transDebugMetadata();
  return true;
}

void LLVMToSPIRVVulkan::collectInputOutputVariables(SPIRVFunction *SF,
                                                    Function *F) {
  for (auto &GV : M->globals()) {
    const auto AS = GV.getAddressSpace();
    if (AS != SPIRAS_Input && AS != SPIRAS_Output)
      continue;

    // Exception added for WorkgroupSize, since it
    // will be OpSpecConstant instead of OpVariable
    if (isBuiltIn(&GV, spv::BuiltInWorkgroupSize) ||
        isBuiltIn(&GV, spv::BuiltInGlobalOffset))
      continue;

    std::unordered_set<const Function *> Funcs;

    for (const auto &U : GV.uses()) {
      const Instruction *Inst = dyn_cast<Instruction>(U.getUser());
      if (!Inst)
        continue;
      Funcs.insert(Inst->getFunction());
    }

    if (isAnyFunctionReachableFromFunction(F, Funcs)) {
      SF->addVariable(ValueMap[&GV]);
    }
  }
}

} // Namespace SPIRV

ModulePass *llvm::createLLVMToSPIRVVulkan(SPIRVModule *SMod) {
  return new LLVMToSPIRVVulkan(SMod);
}