//===- PDSE.h - PartialDead Store Elimination -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates fully and partially dead stores in a global and sparse
// manner. Consult the header comment in PDSE.cpp for further details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_PDSE_H
#define LLVM_TRANSFORMS_SCALAR_PDSE_H

#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
struct PDSEPass : public PassInfoMixin<PDSEPass> {
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};
}

#endif // LLVM_TRANSFORMS_SCALAR_PDSE_H
