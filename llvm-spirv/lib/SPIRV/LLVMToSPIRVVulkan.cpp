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
      BF->getModule()->isEntryPoint(spv::ExecutionModelKernel, BF->getId());

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

  SPIRVTypeFunction *BFT = static_cast<SPIRVTypeFunction *>(
      transType(getAnalysis<OCLTypeToSPIRV>().getAdaptedType(F)));
  SPIRVFunction *BF =
      static_cast<SPIRVFunction *>(mapValue(F, BM->addFunction(BFT)));
  BF->setFunctionControlMask(transFunctionControlMask(F));
  if (F->hasName())
    BM->setName(BF, F->getName().str());
  if (oclIsKernel(F)) {
    BM->addEntryPoint(ExecutionModelGLCompute, BF->getId());
    // FIXME: Make execution mode variable
    BF->addExecutionMode(BM->add(new SPIRVExecutionMode(
        BF, ExecutionMode::ExecutionModeLocalSize, 1, 1, 1)));
  }
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
  }
  // if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::ZExt))
  //  BF->addDecorate(DecorationFuncParamAttr, FunctionParameterAttributeZext);
  // if (Attrs.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt))
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
  BM->addCapability(CapabilityVariablePointersStorageBuffer);
  BM->addExtension(ExtensionID::SPV_KHR_variable_pointers);
  // BM->setMemoryModel(MemoryModelGLSL450);
  // BM->setSPIRVVersion(static_cast<SPIRVWord>(VersionNumber::SPIRV_1_3));
  return true;
}

SPIRVType *LLVMToSPIRVVulkan::transType(Type *T) {
  LLVMToSPIRVTypeMap::iterator Loc = TypeMap.find(T);
  if (Loc != TypeMap.end())
    return Loc->second;

  SPIRVDBG(dbgs() << "[transTypeVulkan] " << *T << '\n');

  if (auto ST = dyn_cast<StructType>(T)) {
    assert(ST->isSized());

    SPIRVTypeStruct *Ret;
    if (ST->getStructName().find("_arg_") != std::string::npos) {
      InParameterStructure = true;
       Ret =
          reinterpret_cast<SPIRVTypeStruct*>(LLVMToSPIRV::transType(T));
      InParameterStructure = false;
      Ret->addDecorate(DecorationBlock);

      //Ret->size
    } else {
      Ret = reinterpret_cast<SPIRVTypeStruct *>(LLVMToSPIRV::transType(T));
    }

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
    if (InParameterStructure) {
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
      return RtArray;
    }
  } else if (auto Ar = dyn_cast<ArrayType>(T)) {
    auto NewArr = LLVMToSPIRV::transType(T);
    NewArr->addDecorate(DecorationArrayStride, 
          M->getDataLayout().getTypeStoreSize(Ar->getArrayElementType()).getFixedSize());
    return NewArr;
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

  SPIRVValue *Alternative = nullptr;
  if (isSkippable(V, BB, &Alternative)) {
    if (Alternative)
      return mapValue(V, Alternative);
    SPIRVDBG(dbgs() << "[skipped] " << '\n')
    return Alternative;
  }

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    auto TransValue = LLVMToSPIRV::transValueWithoutDecoration(V, BB);
    auto Pos = GV->getName().find("_arg_");
    if (Pos != std::string::npos) {
      auto IDString = GV->getName()[Pos + 5];
      auto ID = std::stoi(&IDString);
      TransValue->addDecorate(DecorationDescriptorSet, 0);
      TransValue->addDecorate(DecorationBinding, ID);
    } 
    return TransValue;
  }
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

    auto TypePointer = cast<PointerType>(GEP->getPointerOperand()->getType());
    if (TypePointer->getElementType()->isStructTy()) {

      auto Type = GEP->getType();
      if (TypePointer->getElementType()->getStructName().find("_arg_") !=
          std::string::npos) {
        // Check if the the last index (the runtime array) is accessed
        SPIRVValue *LastIndex = Indices[Indices.size() - 1];
        if (auto constant = reinterpret_cast<SPIRVConstant *>(LastIndex)) {
          if (constant->getZExtIntValue() == 3) {
            // If yes, add another index to access first element within the 
            // runtime array to get a element pointer
            Indices.push_back(
                BM->addConstant(transType(GEP->getOperand(1)->getType()), 0));
            // This additional index also changes the type, which is then
            // one type step down the pointer chain
            Type = Type->getPointerElementType();
          }
        }
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
    }
  }

  if (BranchInst *Branch = dyn_cast<BranchInst>(V)) {
    SPIRVLabel *SuccessorTrue = static_cast<SPIRVLabel *>(
        LLVMToSPIRV::transValue(Branch->getSuccessor(0), BB));

    std::vector<SPIRVWord> Parameters;
    spv::LoopControlMask LoopControl = getLoopControl(Branch, Parameters);

    // Somehow the logic in SPIRVWriter.cpp is not working properly.
    // It seems valid but somehow it is impossible to add
    // loop meta data to the branch instruction and therefore
    // it will not be converted into an structured CFG in SPIRV
    // This "fix" instead uses for for-loop increment block
    // by simple label name match. This then has the back edge
    // and uses the same logic as original code 
    if (Branch->isUnconditional()) {
      // For "for" and "while" loops llvm.loop metadata is attached to
      // an unconditional branch instruction.
      // if (LoopControl != spv::LoopControlMaskNone) {
      if (BB->getName().find(".inc") != std::string::npos) {
        // SuccessorTrue is the loop header BB.
        const SPIRVInstruction *Term = SuccessorTrue->getTerminateInstr();
        if (Term && Term->getOpCode() == OpBranchConditional) {
          const auto *Br = static_cast<const SPIRVBranchConditional *>(Term);
          BM->addLoopMergeInst(Br->getFalseLabel()->getId(), // Merge Block
                               BB->getId(),                  // Continue Target
                               LoopControl, Parameters, SuccessorTrue);
        } else {
          if (BM->isAllowedToUseExtension(
                  ExtensionID::SPV_INTEL_unstructured_loop_controls)) {
            // For unstructured loop we add a special loop control instruction.
            // Simple example of unstructured loop is an infinite loop, that has
            // no terminate instruction.
            BM->addLoopControlINTELInst(LoopControl, Parameters, SuccessorTrue);
          }
        }
      }
      return mapValue(V, BM->addBranchInst(SuccessorTrue, BB));
    }
    // TODO: maybe same has to be done with the do; while
  }

  return LLVMToSPIRV::transValueWithoutDecoration(V, BB, CreateForward);
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
            Value->getType(), false, SPIRVLinkageTypeKind::LinkageTypeInternal,
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
  if (auto Cast = dyn_cast<AddrSpaceCastInst>(U)) {
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

      std::vector<SPIRVValue *> Indices;
      Indices.push_back(BM->getLiteralAsConstant(0));

      if (Target1Type == Source->getType()->getPointerElementType()) {
        Target = BM->addAccessChainInst(
            BM->addPointerType(Target->getType()->getPointerStorageClass(),
                               Target1Type),
            Target, Indices, BB, true);
      } else if (Source1Type == Target->getType()->getPointerElementType()) {
        Source = BM->addAccessChainInst(
            BM->addPointerType(Source->getType()->getPointerStorageClass(),
                               Source1Type),
            Source, Indices, BB, true);
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
    SPIRVValue *Var =
        BM->addVariable(VarTy, /*isConstant*/ true, spv::LinkageTypeInternal,
                        Init, "", StorageClassFunction, BB);
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

// Vulkan only allows internal linkage, since Capability Linkage is not
// supported
SPIRVLinkageTypeKind
LLVMToSPIRVVulkan::transLinkageType(const GlobalValue *GV) {
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