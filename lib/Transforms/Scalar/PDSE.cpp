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
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/PDSE.h"

#include <list>

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

struct RealOcc;
struct LambdaOcc;

struct Occurrence {
  BasicBlock *Block;
  OccTy Type;

  const RealOcc *asReal() const {
    return reinterpret_cast<const RealOcc *>(this);
  }

  const LambdaOcc *asLambda() const {
    return reinterpret_cast<const LambdaOcc *>(this);
  }

  RealOcc *asReal() { return reinterpret_cast<RealOcc *>(this); }

  LambdaOcc *asLambda() { return reinterpret_cast<LambdaOcc *>(this); }
};

struct RealOcc final : public Occurrence {
  Instruction *Inst;
  Occurrence *ReprOcc;
  // ^ Points to this real occurrence's representative occurrence, which is the
  // closest post-dominating non-redundant RealOcc without an intervening kill.
  // For representative occurrences themselves, this is nullptr.
  enum OneSidedKill { NoKill, UpKill, DownKill } AlsoKills;
  // ^ Records whether Inst also acts as a kill occurrence. UpKill =
  // load-then-store (e.g., memmove with aliasing operands); DownKill =
  // store-then-load.

  RealOcc(Instruction &I, Occurrence *ReprOcc)
      : Occurrence{I.getParent(), OccTy::Real}, Inst(&I), ReprOcc(ReprOcc) {}

  RealOcc(Instruction &I, OneSidedKill AlsoKills)
      : Occurrence{I.getParent(), OccTy::Real}, Inst(&I), ReprOcc(nullptr),
        AlsoKills(AlsoKills) {}

  static RealOcc upKill(Instruction &I) { return RealOcc(I, UpKill); }

  static RealOcc downKill(Instruction &I) { return RealOcc(I, DownKill); }

  static RealOcc noKill(Instruction &I) { return RealOcc(I, NoKill); }

  // "Null" real occurrence -- only used to create DeadOnExit.
  RealOcc()
      : Occurrence{nullptr, OccTy::Real}, Inst(nullptr), ReprOcc(nullptr),
        AlsoKills(NoKill) {}
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
};

// A faux occurrence used to detect stores to non-escaping memory that are
// redundant with respect to function exit. TODO: Include in version map when
// needed.
RealOcc DeadOnExit;

// Maximal group of must-aliasing stores.
struct OccClass {
  MemoryLocation Loc;
  // ^ The memory location that each member writes to.
  DenseSet<const Instruction *> Members;
  // ^ Used to avoid calling getModRefInfo on known real occs during renaming.
  DenseSet<BasicBlock *> Blocks;
  // ^ Set of BasicBlocks inhabited by members. Used to compute lambda
  // placement.
  bool CanEscape;
};

// Wraps an instruction that modrefs and/or may-throw. May-throws are
// significant because they count as killing occurrences for escaping stores.
struct MemOrThrow {
  Instruction *I;
  bool MemInst;
};

// Analogous to MemorySSA's AccessList, but for both memory and may-throw
// instructions, in reverse.
using BlockInsts = DenseMap<const BasicBlock *, SmallVector<MemOrThrow, 8>>;

// Handles lambda insertion and occurrence renaming by walking a generalized
// renamer state over the post-dom tree.
template <typename State> class PostDomRenamer {
  const OccClass &CurOcc;
  const BlockInsts &PerBlock;
  AliasAnalysis &AA;
  PostDominatorTree &PDT;

  void computeLambdaBlocks(SmallVectorImpl<BasicBlock *> &LambdaBlocks) {
    // Enumerate def blocks, which are all blocks containing kill and/or real
    // occurrences. TODO: Possibly use CurOcc.Blocks directly.
    SmallPtrSet<BasicBlock *, 8> KillBlocks(CurOcc.Blocks.begin(),
                                            CurOcc.Blocks.end());

    // Account for kill-only blocks; if it contains a real occ, we already know
    // about it.
    for (const BlockInsts::value_type &BB : PerBlock)
      if (CurOcc.Blocks.count(const_cast<BasicBlock *>(BB.first)) == 0)
        for (const MemOrThrow &MOT : BB.second)
          if (MOT.MemInst && AA.getModRefInfo(MOT.I, CurOcc.Loc) & MRI_Ref) {
            KillBlocks.insert(const_cast<BasicBlock *>(BB.first));
            break;
          }

    // Compute lambdas.
    ReverseIDFCalculator RIDF(PDT);
    RIDF.setDefiningBlocks(KillBlocks);
    RIDF.calculate(LambdaBlocks);
  }

  void renameBlock(BasicBlock &BB, State &S) {
    if (PerBlock.count(&BB)) {
      for (const MemOrThrow &MOT : PerBlock.find(&BB)->second) {
        DEBUG(dbgs() << "Visiting " << *MOT.I << "\n");
        if (MOT.I->mayThrow() && CurOcc.CanEscape)
          S.handleMayThrowKill(MOT.I);
        else if (MOT.MemInst) {
          if (CurOcc.Members.count(MOT.I))
            S.handleRealOcc(MOT.I);
          else {
            ModRefInfo MRI = AA.getModRefInfo(MOT.I, CurOcc.Loc);
            if (MRI & MRI_Ref)
              S.handleAliasingKill(MOT.I);
            else if (MRI & MRI_Mod)
              S.handleAliasingStore(MOT.I);
          }
        }
      }
    }

    if (&BB == &BB.getParent()->getEntryBlock())
      S.handlePostDomExit();

    for (BasicBlock *Pred : predecessors(&BB))
      S.handlePredecessor(*Pred);
  }

public:
  DenseMap<const BasicBlock *, LambdaOcc> insertLambdas() {
    SmallVector<BasicBlock *, 8> LambdaBlocks;
    computeLambdaBlocks(LambdaBlocks);

    DenseMap<const BasicBlock *, LambdaOcc> RetVal;
    for (BasicBlock *BB : LambdaBlocks)
      RetVal.insert({BB, LambdaOcc(BB)});
    return RetVal;
  }

  void renamePass(const State &RootState) {
    struct StackEntry {
      DomTreeNode *Node;
      DomTreeNode::iterator ChildIt;
      State S;
    };

    SmallVector<StackEntry, 16> Stack;
    DomTreeNode *Root = PDT.getRootNode();
    if (Root->getBlock()) {
      // Real and unique exit block.
      Stack.push_back({Root, Root->begin(), RootState});
      DEBUG(dbgs() << "Entering root " << Root->getBlock()->getName() << "\n");
      renameBlock(*Root->getBlock(), Stack.back().S);
    } else {
      // Multiple exits and/or infinite loops.
      for (DomTreeNode *N : *Root) {
        Stack.push_back({N, N->begin(), RootState.enterBlock(*N->getBlock())});
        renameBlock(*Stack.back().Node->getBlock(), Stack.back().S);
      }
    }

    // Visit blocks in post-dom pre-order
    while (!Stack.empty()) {
      if (Stack.back().ChildIt == Stack.back().Node->end())
        Stack.pop_back();
      else {
        DomTreeNode *Cur = *Stack.back().ChildIt++;
        State NewS = Stack.back().S.enterBlock(*Cur->getBlock());
        renameBlock(*Cur->getBlock(), NewS);
        if (Cur->begin() != Cur->end())
          Stack.push_back({Cur, Cur->begin(), NewS});
      }
    }
  }

  PostDomRenamer(const OccClass &CurOcc, const BlockInsts &PerBlock,
                 AliasAnalysis &AA, PostDominatorTree &PDT)
      : CurOcc(CurOcc), PerBlock(PerBlock), AA(AA), PDT(PDT) {}
};

struct RedGraph {
  DenseMap<const BasicBlock *, std::list<RealOcc>> BlockOccs;
  // ^ TODO: Figure out iplist for this?
  DenseMap<const BasicBlock *, LambdaOcc> Lambdas;

  const LambdaOcc *getLambda(const BasicBlock &BB) const {
    return Lambdas.count(&BB) ? &Lambdas.find(&BB)->second : nullptr;
  }

  LambdaOcc *getLambda(const BasicBlock &BB) {
    return Lambdas.count(&BB) ? &Lambdas.find(&BB)->second : nullptr;
  }

  RealOcc &addRealOcc(RealOcc R, const Instruction &I) {
    std::list<RealOcc> &OccList = BlockOccs[I.getParent()];
    OccList.push_back(std::move(R));
    return OccList.back();
  }
};

// CRTP.
template <typename T> struct RenameState {
  RedGraph *const FRG;
  Occurrence *ReprOcc;
  // ^ Current representative occurrence, or nullptr for _|_.
  bool CrossedRealOcc;
  // ^ Have we crossed a real occurrence since the last non-kill occurrence?

  void updateUpSafety() {
    // We can immediately conclude a lambda to be up-unsafe if it
    // reverse-reaches any of the following without first crossing a real
    // occurrence:
    //   - aliasing kill
    //   - may-alias store
    //   - function exit
    // In a post-dom pre-order walk, this is equivalent to encountering any of
    // these while the current repr occ is a lambda.
    if (ReprOcc && ReprOcc->Type == OccTy::Lambda)
      ReprOcc->asLambda()->UpSafe &= CrossedRealOcc;
  }

  void kill(Instruction *I) {
    ReprOcc = nullptr;
    CrossedRealOcc = false;
  }

  T enterBlock(const BasicBlock &BB) const {
    // Set the current repr occ to the new block's lambda, if it contains one.
    return FRG->getLambda(BB) ? T{FRG, FRG->getLambda(BB), false} : *this;
  }

  RealOcc &handleRealOcc(Instruction *I) {
    CrossedRealOcc = true;
    RealOcc &R = FRG->addRealOcc(RealOcc(*I, ReprOcc), *I);
    // Current occ is a repr occ if we've just emerged from a kill.
    ReprOcc = ReprOcc ? ReprOcc : &R;
    return R;
  }

  void handleMayThrowKill(Instruction *I) {
    reinterpret_cast<T *>(this)->kill(I);
    reinterpret_cast<T *>(this)->updateUpSafety();
  }

  void handleAliasingKill(Instruction *I) {
    reinterpret_cast<T *>(this)->kill(I);
    reinterpret_cast<T *>(this)->updateUpSafety();
  }

  void handleAliasingStore(Instruction *I) {
    reinterpret_cast<T *>(this)->updateUpSafety();
  }

  void handlePostDomExit() { reinterpret_cast<T *>(this)->updateUpSafety(); }

  void handlePredecessor(const BasicBlock &Pred) {
    if (LambdaOcc *L = FRG->getLambda(Pred))
      L->Operands.push_back({ReprOcc, CrossedRealOcc});
  }
};

// Renaming state for pure PDSE that frugally omits version numbers.
struct NonVersioning : public RenameState<NonVersioning> {
  using Base = RenameState<NonVersioning>;
  NonVersioning(RedGraph *const FRG, Occurrence *ReprOcc, bool CrossedRealOcc)
      : Base{FRG, ReprOcc, CrossedRealOcc} {}
};

// Maps basic blocks/instructions to lambdas/real occs and their versions,
// respectively.
using VersionMap =
    DenseMap<const Value *, std::pair<const Occurrence *, unsigned>>;

// Track occurrence version numbers for pretty printing.
struct Versioning : RenameState<Versioning> {
  using Base = RenameState<Versioning>;

  VersionMap *const OccVersion;
  // ^ nullptr Elem.RealOcc = kill occurrencs.
  unsigned CurrentVer;

  void kill(Instruction *I) {
    Base::kill(I);
    // Track kill occurrences for the pretty printer.
    OccVersion->insert({I, {nullptr, -1}});
    CurrentVer += 1;
  }

  Versioning(RedGraph *const FRG, Occurrence *ReprOcc, bool CrossedRealOcc,
             VersionMap *const OccVersion, unsigned CurrentVer)
      : Base{FRG, ReprOcc, CrossedRealOcc}, OccVersion(OccVersion),
        CurrentVer(CurrentVer) {}

  Versioning enterBlock(const BasicBlock &BB) const {
    DEBUG(dbgs() << "Entering block " << BB.getName()
                 << (FRG->getLambda(BB) ? " with lambda\n" : "\n"));
    if (LambdaOcc *L = FRG->getLambda(BB)) {
      OccVersion->insert({&BB, {L, CurrentVer + 1}});
      return Versioning(FRG, L, false, OccVersion, CurrentVer + 1);
    }
    return *this;
  }

  RealOcc &handleRealOcc(Instruction *I) {
    RealOcc &R = Base::handleRealOcc(I);
    // Assign a version number to the real occ and tag its instruction.
    OccVersion->insert({I, {&R, CurrentVer}});
    return R;
  }
};

struct FRGAnnot final : public AssemblyAnnotationWriter {
  const RedGraph &FRG;
  const VersionMap &OccVersion;

  FRGAnnot(const RedGraph &FRG, const VersionMap &OccVersion)
      : FRG(FRG), OccVersion(OccVersion) {}

  virtual void emitBasicBlockEndAnnot(const BasicBlock *BB,
                                      formatted_raw_ostream &OS) override {
    if (const LambdaOcc *L = FRG.getLambda(*BB)) {
      assert(L->Operands.size() > 1 &&
             "IDFCalculator computed an unnecessary lambda.");
      auto PrintOperand = [&](const LambdaOcc::Operand &Op) {
        if (!Op.ReprOcc)
          OS << "_|_";
        else if (Op.ReprOcc->Type == OccTy::Real)
          OS << OccVersion.find(Op.ReprOcc->asReal()->Inst)->second.second;
        else
          OS << OccVersion.find(Op.ReprOcc->asLambda()->Block)->second.second;
        if (Op.HasRealUse)
          OS << "*";
      };
      OS << "; Lambda(";
      PrintOperand(L->Operands[0]);
      for (const LambdaOcc::Operand &Op :
           make_range(std::next(L->Operands.begin()), L->Operands.end())) {
        OS << ", ";
        PrintOperand(Op);
      }
      OS << ") = " << OccVersion.find(BB)->second.second << "\n";
    }
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) override {
    if (OccVersion.count(I)) {
      const auto &RV = OccVersion.find(I)->second;
      if (const RealOcc *R = RV.first->asReal())
        OS << "; " << (R->ReprOcc ? "Real" : "Repr") << "(" << RV.second
           << ")\n";
      else
        OS << "; Kill\n";
    }
  }
};

// Inherited from old DSE.
Optional<std::pair<MemoryLocation, RealOcc>> makeRealOcc(Instruction &Inst) {
  using std::make_pair;
  if (StoreInst *SI = dyn_cast<StoreInst>(&Inst))
    return make_pair(MemoryLocation::get(SI), RealOcc(Inst, nullptr));
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst))
    return make_pair(MemoryLocation::getForDest(MI), RealOcc(Inst, nullptr));
  return None;
}

struct OccTracker {
  SmallVector<OccClass, 32> Classes;

  OccTracker &push_back(MemoryLocation Loc, Instruction &I, AliasAnalysis &AA) {
    // TODO: Run faster than quadratic.
    auto OC = find_if(Classes, [&](const OccClass &OC) {
      return AA.alias(Loc, OC.Loc) == MustAlias;
    });
    if (OC == Classes.end()) {
      DEBUG(dbgs() << "New real occ class: " << I << "\n");
      // TODO: Analyze escapability.
      Classes.push_back({std::move(Loc), {&I}, {I.getParent()}, true});
    } else {
      DEBUG(dbgs() << "Collected real occ: " << I << "\n");
      OC->Members.insert(&I);
      OC->Blocks.insert(I.getParent());
    }
    return *this;
  }
};

bool nonEscapingOnUnwind(Instruction &I, const TargetLibraryInfo &TLI) {
  return isa<AllocaInst>(&I) ||
         (isAllocLikeFn(&I, &TLI) && !PointerMayBeCaptured(&I, false, true));
}

bool runPDSE(Function &F, AliasAnalysis &AA, PostDominatorTree &PDT,
             const TargetLibraryInfo &TLI) {
  OccTracker Worklist;
  BlockInsts PerBlock;
  DenseSet<const Value *> NonEscapes;

  // Record non-escaping args.
  for (Argument &Arg : F.args())
    if (Arg.hasByValOrInAllocaAttr())
      NonEscapes.insert(&Arg);

  // Roll through every instruction to collect occurrence classes, build
  // reversed lists of interesting instructions per block, and enumerate all
  // non-escaping memory locations.
  for (BasicBlock &BB : F) {
    for (Instruction &I : reverse(BB)) {
      if (nonEscapingOnUnwind(I, TLI))
        NonEscapes.insert(&I);

      ModRefInfo MRI = AA.getModRefInfo(&I);
      if (MRI & MRI_ModRef || I.mayThrow()) {
        DEBUG(dbgs() << "Interesting: " << I << "\n");
        PerBlock[&BB].push_back({&I, bool(MRI & MRI_ModRef)});
        if (MRI & MRI_Mod)
          if (auto LocOcc = makeRealOcc(I))
            Worklist.push_back(LocOcc->first, I, AA);
      }
    }
  }

  if (PrintFRG) {
    for (const OccClass &OC : Worklist.Classes) {
      PostDomRenamer<Versioning> R(OC, PerBlock, AA, PDT);
      VersionMap OccVersion;
      RedGraph FRG;
      FRG.Lambdas = R.insertLambdas();

      // TODO: Use a different root renamer state for non-escapes.
      R.renamePass(Versioning(&FRG, nullptr, false, &OccVersion, 0));
      dbgs() << "Factored redundancy graph for stores to " << *OC.Loc.Ptr
             << ":\n";
      FRGAnnot Annot(FRG, OccVersion);
      F.print(dbgs(), &Annot);
      dbgs() << "\n";
    }
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
