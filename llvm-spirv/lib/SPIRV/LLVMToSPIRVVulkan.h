//===- SPIRVWriter.h - Converts LLVM to SPIR-V ------------------*- C++ -*-===//
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

#ifndef LLVMTOSPIRVVULKAN_H
#define LLVMTOSPIRVVULKAN_H

#include "OCLTypeToSPIRV.h"
#include "OCLUtil.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVEnum.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVModule.h"
#include "SPIRVType.h"
#include "SPIRVValue.h"
#include "SPIRVWriter.h"

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/IntrinsicInst.h"

#include <memory>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

class LLVMToSPIRVVulkan : public LLVMToSPIRV {
public:
  LLVMToSPIRVVulkan(SPIRVModule *SMod = nullptr);

  virtual StringRef getPassName() const override { return "LLVMToSPIRVVulkan"; }

  bool translate() override;

  bool transExecutionMode() override;
  SPIRVType *transType(Type *T) override;

  bool transAddressingMode() override;
  bool transAlign(Value *V, SPIRVValue *BV) override;
  bool transBuiltinSet() override;

  SPIRVValue *transIntrinsicInst(IntrinsicInst *Intrinsic,
                                 SPIRVBasicBlock *BB) override;
  SPIRVValue *transDirectCallInst(CallInst *Call, SPIRVBasicBlock *BB) override;
  bool transDecoration(Value *V, SPIRVValue *BV) override;
  SPIRVFunction *transFunctionDecl(Function *F) override;
  SPIRVValue *transValueWithoutDecoration(
      Value *V, SPIRVBasicBlock *BB, bool CreateForward,
      FuncTransMode FuncTrans = FuncTransMode::Decl) override;

protected:
  std::vector<SPIRVWord>
  transValue(const std::vector<Value *> &Values, SPIRVBasicBlock *BB,
             SPIRVEntry *Entry,
             std::vector<std::pair<SPIRVValue *, SPIRVValue *>> &CopyBack);
  void transFunction(Function *I) override;

  SPIRV::SPIRVInstruction *transUnaryInst(UnaryInstruction *U,
                                          SPIRVBasicBlock *BB) override;
  void collectInputOutputVariables(SPIRVFunction *SF, Function *F) override;

  bool isSkippable(Value *V, SPIRVBasicBlock *BB, SPIRVValue **Alternative);
  SPIRV::SPIRVLinkageTypeKind transLinkageType(const GlobalValue *GV);
  bool isBuiltIn(Value *V, spv::BuiltIn builtIn);

  bool InParameterStructure = false;

  SPIRVValue *transOrigin(Value *V, SPIRVBasicBlock *BB);

  std::vector<Value *> RuntimeArrayArguments;
  bool WorkgroupSizeAvailable = false;
  SPIRVType *WorkgroupSizeType = nullptr;
  llvm::LoopInfoBase<llvm::BasicBlock, llvm::Loop> LoopInfoObj;
  llvm::PostDominatorTree PDominatorTree;
  std::map<SPIRVBasicBlock *, SPIRVBasicBlock *> PredecessorBBChangeMap;
};

} // Namespace SPIRV

#endif // LLVMTOSPIRVVULKAN_H