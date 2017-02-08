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
