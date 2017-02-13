//===- PDSE.cpp - Partial Dead Store Elimination --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs a variation of partial dead store elimination as described
// in:
//
//   Register Promotion Sparse Partial Redundancy Elimination of Loads and Store
//   https://doi.org/10.1145/277650.277659
//
// "Partial" refers to the ability to transform partial store redundancies into
// full redundancies by inserting stores at appropriate split points. For
// instance:
//
// bb0:
//   store i8 undef, i8* %x
//   br i1 undef label %true, label %false
// true:
//   store i8 undef, i8* %x
//   br label %exit
// false:
//   br label %exit
// exit:
//   ret void
//
// The store in bb0 counts as partially redundant (with respect to the store in
// the true block) and can be made fully redundant by inserting a copy into the
// false block.
//
// For a gentler introduction to PRE, see:
//
//   Partial Redundancy Elimination in SSA Form
//   https://doi.org/10.1145/319301.319348
//
// Differences between the papers and this implementation:
// - May-throw instructions count as killing occurrences in the factored
//   redundancy graph of escaping stores;
// - TODO: Figure out partial overwrite tracking.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/PDSE.h"

#define DEBUG_TYPE "pdse"

using namespace llvm;

static cl::opt<bool>
    PrintFRG("print-frg", cl::init(false), cl::Hidden,
             cl::desc("Print the factored redundancy graph of stores."));

namespace {
// Representations of factored redundancy graph elements.
enum struct OccTy {
  Real,
  Lambda,
};

struct Occurrence {
  BasicBlock *Block;
  OccTy Type;
};

struct RealOcc final : public Occurrence {
  Instruction *Inst;
  Occurrence *ReprOcc;
  // ^ Points to this real occurrence's representative occurrence, which is the
  // closest post-dominating non-redundant RealOcc without an intervening kill.

  RealOcc(Instruction *I, Occurrence *ReprOcc)
      : Occurrence{I->getParent(), OccTy::Real}, Inst(I), ReprOcc(ReprOcc) {}

  RealOcc()
      : Occurrence{nullptr, OccTy::Real}, Inst(nullptr), ReprOcc(nullptr) {}

  friend class FRGAnnot;
};

struct LambdaOcc final : public Occurrence {
  // Lambda operand representation.
  struct Operand {
    Occurrence *ReprOcc;
    // ^ Representative occurrence dominating this operand. nullptr = _|_.
    bool HasRealUse;
    // ^ Is there a real occurrence on some path from ReprOcc to this operand?
    // Always false for _|_ operands.
  };
  SmallVector<Operand, 8> Operands;

  // Consult the Kennedy et al. paper for these.
  bool UpSafe;
  bool CanBeAnt;
  bool Later;

  LambdaOcc(BasicBlock *Block)
      : Occurrence{Block, OccTy::Lambda}, Operands{}, UpSafe(true),
        CanBeAnt(true), Later(true) {}

  friend class FRGAnnot;
};

// Faux occurrence used to detect stores to non-escaping memory that are
// post-dommed by function exit.
RealOcc DeadOnExit;

// Inherited from old DSE.
MemoryLocation getLocForWrite(Instruction *Inst) {
  if (StoreInst *SI = dyn_cast<StoreInst>(Inst))
    return MemoryLocation::get(SI);

  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(Inst)) {
    MemoryLocation Loc = MemoryLocation::getForDest(MI);
    return Loc;
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
  if (!II)
    return MemoryLocation();

  switch (II->getIntrinsicID()) {
  default:
    return MemoryLocation(); // Unhandled intrinsic.
  case Intrinsic::init_trampoline:
    return MemoryLocation(II->getArgOperand(0));
  case Intrinsic::lifetime_end:
    return MemoryLocation(
        II->getArgOperand(1),
        cast<ConstantInt>(II->getArgOperand(0))->getZExtValue());
  }
}

struct OccTracker {
  // Must-aliasing group of store occurrences..
  struct OccClass {
    MemoryLocation Loc;
    DenseSet<Instruction *> Members;
    DenseSet<BasicBlock *> Blocks;
  };

  SmallVector<OccClass, 32> Inner;

  OccTracker &push_back(MemoryLocation Loc, AliasAnalysis &AA) {
    // TODO: Run faster than quadratic.
    auto OC = find_if(Inner, [&](const OccClass &OC) {
      return AA.alias(Loc, OC.Loc) == MustAlias;
    });
    if (OC == Inner.end())
      Inner.push_back(OccClass{std::move(Loc), {}, {}});
    return *this;
  }
};

// Wraps an instruction that modrefs and/or mayThrow. Throwables are considered
// killing occurrences for escaping stores.
struct MemOrThrow {
  Instruction *I;
  bool MemInst;
};

bool runPDSE(Function &F, AliasAnalysis &AA, const PostDominatorTree &PDT,
             const TargetLibraryInfo &TLI) {
  OccTracker Worklist;
  DenseMap<BasicBlock *, SmallVector<MemOrThrow, 8>> PerBlock;

  // Build occurrence classes.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      ModRefInfo MRI = AA.getModRefInfo(&I);
      if (MRI & MRI_ModRef || I.mayThrow()) {
        PerBlock.insert({&BB, {MemOrThrow{&I, bool(MRI & MRI_ModRef)}}});
        if (MRI & MRI_Mod)
          if (MemoryLocation WriteLoc = getLocForWrite(&I))
            Worklist.push_back(WriteLoc, AA);
      }
    }
  }

  if (PrintFRG) {
    DEBUG(dbgs() << "TODO: Print factored redundancy graph.\n");
    return false;
  } else {
    DEBUG(dbgs() << "Dummy PDSE pass.\n");
  }
  return false;
}

class PDSELegacyPass : public FunctionPass {
public:
  PDSELegacyPass() : FunctionPass(ID) {
    initializePDSELegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    return runPDSE(F, getAnalysis<AAResultsWrapperPass>().getAAResults(),
                   getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
                   getAnalysis<TargetLibraryInfoWrapperPass>().getTLI());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    AU.setPreservesCFG();
    AU.addPreserved<PostDominatorTreeWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  static char ID; // Pass identification, replacement for typeid
};
} // end anonymous namespace

char PDSELegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(PDSELegacyPass, "pdse", "Partial Dead Store Elimination",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(PDSELegacyPass, "pdse", "Partial Dead Store Elimination",
                    false, false)

namespace llvm {
PreservedAnalyses PDSEPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!runPDSE(F, AM.getResult<AAManager>(F),
               AM.getResult<PostDominatorTreeAnalysis>(F),
               AM.getResult<TargetLibraryAnalysis>(F)))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}
} // end namespace llvm
