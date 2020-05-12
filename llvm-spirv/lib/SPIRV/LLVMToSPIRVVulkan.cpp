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

LLVMToSPIRVVulkan::LLVMToSPIRVVulkan(SPIRVModule *SMod) : LLVMToSPIRV(SMod) {}

void LLVMToSPIRVVulkan::transFunction(Function *I) {
  SPIRVFunction *BF = transFunctionDecl(I);
  // Creating all basic blocks before creating any instruction.
  for (auto &FI : *I) {
    transValue(&FI, nullptr);
  }
  for (auto &FI : *I) {
    SPIRVBasicBlock *BB =
        static_cast<SPIRVBasicBlock *>(transValue(&FI, nullptr));
    for (auto &BI : FI) {
      transValue(&BI, BB, false);
    }
  }

  if (BF->getModule()->isEntryPoint(spv::ExecutionModelGLCompute,
                                    BF->getId()) &&
      BF->shouldFPContractBeDisabled()) {
    BF->addExecutionMode(BF->getModule()->add(
        new SPIRVExecutionMode(BF, spv::ExecutionModeContractionOff)));
  }
  if (BF->getModule()->isEntryPoint(spv::ExecutionModelGLCompute,
                                    BF->getId())) {
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

  SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
      transType(getAnalysis<OCLTypeToSPIRV>().getAdaptedType(F)));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName())
    BM->setName(BF, F->getName().str());
  if (oclIsKernel(F))
    BM->addEntryPoint(ExecutionModelGLCompute, BF->getId());
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
    //if (I->hasByValAttr())
    //  BA->addAttr(FunctionParameterAttributeByVal);
    //if (I->hasNoAliasAttr())
    //  BA->addAttr(FunctionParameterAttributeNoAlias);
    //if (I->hasNoCaptureAttr())
    //  BA->addAttr(FunctionParameterAttributeNoCapture);
    //if (I->hasStructRetAttr())
    //  BA->addAttr(FunctionParameterAttributeSret);
    //if (Attrs.hasAttribute(ArgNo + 1, Attribute::ZExt))
    //  BA->addAttr(FunctionParameterAttributeZext);
    //if (Attrs.hasAttribute(ArgNo + 1, Attribute::SExt))
    //  BA->addAttr(FunctionParameterAttributeSext);
    if (BM->isAllowedToUseVersion(VersionNumber::SPIRV_1_1) &&
        Attrs.hasAttribute(ArgNo + 1, Attribute::Dereferenceable))
      BA->addDecorate(DecorationMaxByteOffset,
                      Attrs.getAttribute(ArgNo + 1, Attribute::Dereferenceable)
                          .getDereferenceableBytes());
  }
  //if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::ZExt))
  //  BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
  //if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt))
  //  BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeSext);
  if (Attrs.hasFnAttribute("referenced-indirectly")) {
    assert(!oclIsKernel(F) &&
           "kernel function was marked as referenced-indirectly");
    BF->addDecorate(DecorationReferencedIndirectlyINTEL);
  }
  SPIRVDBG(dbgs() << "[transFunction] " << *F << " => ";
           spvdbgs() << *BF << '\n';)
  return BF;
}

bool LLVMToSPIRVVulkan::transAddressingMode() {
  BM->setAddressingModel(AddressingModelLogical);
  //BM->setMemoryModel(MemoryModelGLSL450);
  //BM->setSPIRVVersion(static_cast<SPIRVWord>(VersionNumber::SPIRV_1_3));
  return true;
}

SPIRVType *LLVMToSPIRVVulkan::transType(Type *T) {

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    if (ST->getStructName().find("_arg_") != std::string::npos) {
      inParameterStructure = true;
      auto ret = LLVMToSPIRV::transType(T);
      inParameterStructure = false;
      return ret;
    }

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
    if (inParameterStructure) {
      auto subtype = Pt->getElementType();
      /*auto subsubtype = subtype->getElementType();
      auto addrspace = SPIRSPIRVAddrSpaceMap::map(
          static_cast<SPIRAddressSpace>(Pt->getAddressSpace()));

      return mapType(
          T, BM->addPointerType(
                 addrspace, BM->addRuntimeArrayType(transType(subsubtype))));*/
      return BM->addRuntimeArrayType(transType(subtype));
    }
  }

  return LLVMToSPIRV::transType(T);
}

/// An instruction may use an instruction from another BB which has not been
/// translated. SPIRVForward should be created as place holder for these
/// instructions and replaced later by the real instructions.
/// Use CreateForward = true to indicate such situation.
SPIRVValue *LLVMToSPIRVVulkan::transValueWithoutDecoration(Value *V,
                                                           SPIRVBasicBlock *BB,
                                                           bool CreateForward) {

  SPIRVValue *alternative = nullptr;
  if (isSkippable(V, BB, &alternative)) {
    if (alternative)
      return mapValue(V, alternative);
    SPIRVDBG(dbgs() << "[skipped] " << '\n')
    return alternative;
  }

  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    std::vector<SPIRVValue *> Indices;
    for (unsigned I = 0, E = GEP->getNumIndices(); I != E; ++I)
      Indices.push_back(transValue(GEP->getOperand(I + 1), BB));
    auto pointerType = cast<PointerType>(GEP->getPointerOperand()->getType());
    auto *TransPointerOperand = transValue(GEP->getPointerOperand(), BB);
    if (pointerType->getElementType()->isStructTy() &&
        pointerType->getElementType()->getStructName().find("_arg_") !=
            std::string::npos) {
      SPIRVValue *x = Indices[Indices.size() - 1];
      if (auto constant = reinterpret_cast<SPIRVConstant *>(x)) {
        if (constant->getZExtIntValue() == 3) {
          Indices.push_back(
              BM->addConstant(transType(GEP->getOperand(1)->getType()), 0));
        }
      }
    }

    // Certain array-related optimization hints can be expressed via
    // LLVM metadata. For the purpose of linking this metadata with
    // the accessed array variables, our GEP may have been marked into
    // a so-called index group, an MDNode by itself.
    if (MDNode *IndexGroup = GEP->getMetadata("llvm.index.group")) {
      // When where we work with embedded loops, it's natural that
      // the outer loop's hints apply to all code contained within.
      // The inner loop's specific hints, however, should stay private
      // to the inner loop's scope.
      // Consequently, the following division of the index group metadata
      // nodes emerges:
      // 1) The metadata node has no operands. It will be directly referenced
      //    from within the optimization hint metadata.
      // 2) The metadata node has several operands. It serves to link an index
      //    group specific to some embedded loop with other index groups that
      //    mark the same array variable for the outer loop(s).
      unsigned NumOperands = IndexGroup->getNumOperands();
      if (NumOperands > 0)
        // The index group for this particular "embedded loop depth" is always
        // signalled by the last variable. We'll want to associate this loop's
        // control parameters with this inner-loop-specific index group
        IndexGroup = getMDOperandAsMDNode(IndexGroup, NumOperands - 1);
      IndexGroupArrayMap[IndexGroup] = TransPointerOperand->getId();
    }

    return mapValue(V, BM->addAccessChainInst(transType(GEP->getType()),
                                              TransPointerOperand, Indices, BB,
                                              GEP->isInBounds()));
  }
  
  return LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward);
}

SPIRVInstruction *LLVMToSPIRVVulkan::transUnaryInst(UnaryInstruction *U,
                                                    SPIRVBasicBlock *BB) {

  Op BOC = OpNop;
  if (auto Cast = dyn_cast<AddrSpaceCastInst>(U)) {
    // Do noop and return translated value of the first operand
    return reinterpret_cast<SPIRVInstruction *>(
        getTranslatedValue(U->getOperand(0)));
  } else {
    auto OpCode = U->getOpcode();
    BOC = OpCodeMap::map(OpCode);
  }

  auto Op = transValue(U->getOperand(0), BB);
  return BM->addUnaryInst(transBoolOpCode(Op, BOC), transType(U->getType()), Op,
                          BB);
}

bool LLVMToSPIRVVulkan::transDecoration(Value *V, SPIRVValue *BV) {
  return LLVMToSPIRV::transDecoration(V, BV);
}

// Vulkan allows no allignment decoration (Cap Kernel)
bool LLVMToSPIRVVulkan::transAlign(Value *V, SPIRVValue *BV) {
  return true;
}

// Vulkan only allows internal linkage, since Capability Linkage is not supported
SPIRVLinkageTypeKind LLVMToSPIRVVulkan::transLinkageType(const GlobalValue *GV) {
  return SPIRVLinkageTypeKind::LinkageTypeInternal;
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
  // if (auto bitCast = dyn_cast<BitCastInst>(V)) {
  //  for (auto &op : bitCast->operands()) {
  //    if (op->getName().find(".addr") != std::string::npos)
  //      return true;
  //  }
  //  return false;
  //}
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
  // Skip Lifetime intrinsic, because not allowed for shaders
  if (auto intrinsic = dyn_cast<IntrinsicInst>(V)) {
    if (intrinsic->isLifetimeStartOrEnd())
      return true;
    return false;
  }
  return false;
}

} // Namespace SPIRV

ModulePass *llvm::createLLVMToSPIRVVulkan(SPIRVModule *SMod) {
  return new LLVMToSPIRVVulkan(SMod);
}