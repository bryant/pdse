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
#include "llvm/Analysis/IteratedDominanceFrontier.h"
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

struct Occurrence {
  BasicBlock *Block;
  OccTy Type;
};

struct RealOcc final : public Occurrence {
  Instruction *Inst;
  Occurrence *ReprOcc;
  // ^ Points to this real occurrence's representative occurrence, which is the
  // closest post-dominating non-redundant RealOcc without an intervening kill.
  // For representative occurrences themselves, this is nullptr.
  enum { NoKill, UpKill, DownKill } AlsoKills;
  // ^ Records whether Inst also acts as a kill occurrence. UpKill =
  // load-then-store (e.g., memmove with aliasing operands); DownKill =
  // store-then-load.

  RealOcc(Instruction *I, Occurrence *ReprOcc)
      : Occurrence{I->getParent(), OccTy::Real}, Inst(I), ReprOcc(ReprOcc) {}

  RealOcc(Instruction *I)
      : Occurrence{I->getParent(), OccTy::Real}, Inst(I), ReprOcc(nullptr) {}

  RealOcc()
      : Occurrence{nullptr, OccTy::Real}, Inst(nullptr), ReprOcc(nullptr) {}
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

// Faux occurrence used to detect stores to non-escaping memory that are
// post-dommed by function exit.
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
      renameBlock(*Stack.back().Node->getBlock(), Stack.back().S);
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

struct FRG {
  DenseMap<const BasicBlock *, std::list<RealOcc>> BlockOccs;
  // ^ TODO: Figure out iplist for this?
  DenseMap<const BasicBlock *, LambdaOcc> Lambdas;

  LambdaOcc *getLambda(const BasicBlock &BB) {
    return Lambdas.count(&BB) ? &Lambdas.find(&BB)->second : nullptr;
  }

  RealOcc &addRealOcc(RealOcc R, const BasicBlock &BB) {
    std::list<RealOcc> &OccList = (*BlockOccs)[I->getParent()];
    OccList.push_back(std::move(R));
    return OccList.back();
  }
};

// CRTP.
template <typename T> struct RenameState {
  DenseMap<const BasicBlock *, std::list<RealOcc>> *const BlockOccs;
  // ^ TODO: Figure out iplist for this?
  DenseMap<const BasicBlock *, LambdaOcc> *const Lambdas;
  Occurrence *ReprOcc;
  // ^ Current representative occurrence, or nullptr for _|_.
  bool CrossedRealOcc;
  // ^ Have we crossed a real occurrence since the last non-kill occurrence?

protected:
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
      reinterpret_cast<LambdaOcc *>(ReprOcc)->UpSafe &= CrossedRealOcc;
  }

  void kill(Instruction *I) {
    ReprOcc = nullptr;
    CrossedRealOcc = false;
  }

public:
  T enterBlock(BasicBlock &BB) const {
    // Set the current repr occ to the new block's lambda, if it contains one.
    return Lambdas->count(&BB)
               ? T{BlockOccs, Lambdas, &Lambdas->find(&BB)->second, false}
               : *this;
  }

  RealOcc &handleRealOcc(Instruction *I) {
    CrossedRealOcc = true;
    std::list<RealOcc> &OccList = (*BlockOccs)[I->getParent()];
    OccList.push_back(RealOcc(I, ReprOcc));
    // Current occ is a repr occ if we've just emerged from a kill.
    ReprOcc = ReprOcc ? ReprOcc : &OccList.back();
    return OccList.back();
  }

  void handleMayThrowKill(Instruction *I) {
    kill(I);
    updateUpSafety();
  }

  void handleAliasingKill(Instruction *I) {
    kill(I);
    updateUpSafety();
  }

  void handleAliasingStore(Instruction *I) { updateUpSafety(); }

  void handlePostDomExit() { updateUpSafety(); }

  void handlePredecessor(BasicBlock &Pred) {
    if (Lambdas->count(&Pred)) {
      LambdaOcc &L = Lambdas->find(&Pred)->second;
      L.Operands.push_back({ReprOcc, CrossedRealOcc});
    }
  }
};

// Frugal renaming state for pure PDSE.
struct NonVersioning : public RenameState<NonVersioning> {
  using Base = RenameState<NonVersioning>;
  NonVersioning(
      DenseMap<const BasicBlock *, std::list<RealOcc>> *const BlockOccs,
      DenseMap<const BasicBlock *, LambdaOcc> *const Lambdas,
      Occurrence *ReprOcc, bool CrossedRealOcc)
      : Base{BlockOccs, Lambdas, ReprOcc, CrossedRealOcc} {}
};

// Track occurrence version numbers for pretty printing.
struct Versioning : RenameState<Versioning> {
  DenseMap<const Occurrence *, unsigned> *const OccVersion;
  DenseMap<const Instruction *, RealOcc *> *const Occs;
  // ^ nullptr RealOcc = kill occurrencs.
  unsigned CurrentVer;

  using Base = RenameState<Versioning>;

protected:
  void kill(Instruction *I) {
    Base::kill(I);
    // Track kill occurrences for the pretty printer.
    Occs->insert({I, nullptr});
    CurrentVer += 1;
  }

public:
  Versioning(DenseMap<const BasicBlock *, std::list<RealOcc>> *const BlockOccs,
             DenseMap<const BasicBlock *, LambdaOcc> *const Lambdas,
             Occurrence *ReprOcc, bool CrossedRealOcc,
             DenseMap<const Occurrence *, unsigned> *const OccVersion,
             DenseMap<const Instruction *, RealOcc *> *const Occs,
             unsigned CurrentVer)
      : Base{BlockOccs, Lambdas, ReprOcc, CrossedRealOcc},
        OccVersion(OccVersion), Occs(Occs), CurrentVer(CurrentVer) {}

  Versioning enterBlock(BasicBlock &BB) const {
    DEBUG(dbgs() << "Entering block " << BB.getName()
                 << (Lambdas->count(&BB) ? " with lambda\n" : "\n"));
    if (Lambdas->count(&BB)) {
      LambdaOcc &L = Lambdas->find(&BB)->second;
      OccVersion->insert({&L, CurrentVer + 1});
      return Versioning(BlockOccs, Lambdas, &L, false, OccVersion, Occs,
                        CurrentVer + 1);
    }
    return *this;
  }

  RealOcc &handleRealOcc(Instruction *I) {
    RealOcc &R = Base::handleRealOcc(I);
    // Assign a version number to the real occ and tag its instruction.
    OccVersion->insert({&R, CurrentVer});
    Occs->insert({I, &R});
    return R;
  }
};

struct FRGAnnot final : public AssemblyAnnotationWriter {
  const DenseMap<const Occurrence *, unsigned> &OccVersion;
  const DenseMap<const Instruction *, RealOcc *> &Occs;
  const DenseMap<const BasicBlock *, LambdaOcc> &Lambdas;

  FRGAnnot(const DenseMap<const Occurrence *, unsigned> &OccVersion,
           const DenseMap<const Instruction *, RealOcc *> &Occs,
           const DenseMap<const BasicBlock *, LambdaOcc> &Lambdas)
      : OccVersion(OccVersion), Occs(Occs), Lambdas(Lambdas) {}

  virtual void emitBasicBlockEndAnnot(const BasicBlock *BB,
                                      formatted_raw_ostream &OS) override {
    if (Lambdas.count(BB)) {
      const LambdaOcc &L = Lambdas.find(BB)->second;
      assert(L.Operands.size() > 1 &&
             "IDFCalculator computed an unnecessary lambda.");
      auto PrintOperand = [&](const LambdaOcc::Operand &Op) {
        if (Op.ReprOcc) {
          OS << OccVersion.find(Op.ReprOcc)->second;
          if (Op.HasRealUse)
            OS << "*";
        } else
          OS << "_|_";
      };
      OS << "; Lambda(";
      PrintOperand(L.Operands[0]);
      for (const LambdaOcc::Operand &Op :
           make_range(std::next(L.Operands.begin()), L.Operands.end())) {
        OS << ", ";
        PrintOperand(Op);
      }
      OS << ") = " << OccVersion.find(&L)->second << "\n";
    }
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) override {
    if (Occs.count(I)) {
      const RealOcc *R = Occs.find(I)->second;
      if (R)
        OS << "; " << (R->ReprOcc ? "Real" : "Repr") << "("
           << OccVersion.find(R)->second << ")\n";
      else
        OS << "; Kill\n";
    }
  }
};

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

bool runPDSE(Function &F, AliasAnalysis &AA, PostDominatorTree &PDT,
             const TargetLibraryInfo &TLI) {
  OccTracker Worklist;
  BlockInsts PerBlock;

  // Simultaneously collect occurrence classes and build reversed lists of
  // interesting instructions per block.
  for (BasicBlock &BB : F) {
    for (Instruction &I : reverse(BB)) {
      ModRefInfo MRI = AA.getModRefInfo(&I);
      if (MRI & MRI_ModRef || I.mayThrow()) {
        DEBUG(dbgs() << "Interesting: " << I << "\n");
        PerBlock[&BB].push_back({&I, bool(MRI & MRI_ModRef)});
        if (MRI & MRI_Mod)
          if (MemoryLocation WriteLoc = getLocForWrite(&I))
            Worklist.push_back(WriteLoc, I, AA);
      }
    }
  }

  if (PrintFRG) {
    for (const OccClass &OC : Worklist.Classes) {
      PostDomRenamer<Versioning> R(OC, PerBlock, AA, PDT);
      DenseMap<const BasicBlock *, LambdaOcc> Lambdas = R.insertLambdas();
      DenseMap<const BasicBlock *, std::list<RealOcc>> BlockOccs;
      DenseMap<const Occurrence *, unsigned> OccVersion;
      DenseMap<const Instruction *, RealOcc *> Occs;

      // TODO: Use a different root renamer state for non-escapes.
      R.renamePass(Versioning(&BlockOccs, &Lambdas, nullptr, false, &OccVersion,
                              &Occs, 0));
      dbgs() << "Factored redundancy graph for stores to " << *OC.Loc.Ptr
             << ":\n";
      FRGAnnot Annot(OccVersion, Occs, Lambdas);
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
