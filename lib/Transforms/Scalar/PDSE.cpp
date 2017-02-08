namespace {
bool runPDSE(Function &F, AliasAnalysis &AA, const PostDominatorTree &PDT,
             const TargetLibraryInfo &TLI) {
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
    AU.addRequired<BreakCriticalEdges>();
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
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(PDSELegacyPass, "pdse", "Partial Dead Store Elimination",
                    false, false)

namespace llvm {
PreservedAnalyses PDSEPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &Unused = AM.getResult<BreakCriticalEdgesPass>(F);
  bool Changed = runPDSE(F, AM.getResult<AAManager>(F),
                         AM.getResult<PostDominatorTreeAnalysis>(F),
                         AM.getResult<TargetLibraryAnalysis>(F));

  if (Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserveSet<PostDominatorTreeAnalysis>();
  PA.preserve<GlobalsAA>();
  return PA;
}
} // end namespace llvm
