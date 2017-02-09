#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/PDSE.h"

// TODO: Require BreakCriticalEdges

using namespace llvm;

static cl::opt<bool>
    PrintFRG("print-pdse", cl::init(false), cl::Hidden,
             cl::desc("Print the factored redundancy graph of stores."));

namespace {
bool runPDSE(Function &F, AliasAnalysis &AA, const PostDominatorTree &PDT,
             const TargetLibraryInfo &TLI) {
  if (PrintFRG) {
    dbgs() << "TODO: Print factored redundancy graph.\n";
    return false;
  } else
    dbgs() << "Dummy PDSE pass.\n";
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
  bool Changed = runPDSE(F, AM.getResult<AAManager>(F),
                         AM.getResult<PostDominatorTreeAnalysis>(F),
                         AM.getResult<TargetLibraryAnalysis>(F));

  if (Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}
} // end namespace llvm
