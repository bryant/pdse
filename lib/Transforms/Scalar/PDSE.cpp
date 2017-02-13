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
#include "llvm/Analysis/ValueTracking.h"
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
  unsigned ID;
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
  // closest post-dominating non-redundant occurrence without an intervening
  // kill. For representative occurrences themselves, this is nullptr.
  enum OneSidedKill { NoKill, UpKill, DownKill } AlsoKills;
  // ^ Records whether Inst also acts as a kill occurrence. UpKill =
  // load-then-store (e.g., memmove with aliasing operands); DownKill =
  // store-then-load.

  RealOcc(Instruction &I, Occurrence *ReprOcc)
      : Occurrence{-1u, I.getParent(), OccTy::Real}, Inst(&I),
        ReprOcc(ReprOcc) {}

  RealOcc(Instruction &I, OneSidedKill AlsoKills)
      : Occurrence{-1u, I.getParent(), OccTy::Real}, Inst(&I), ReprOcc(nullptr),
        AlsoKills(AlsoKills) {}

  // "Null" real occurrence -- only used to create DeadOnExit.
  RealOcc()
      : Occurrence{0, nullptr, OccTy::Real}, Inst(nullptr), ReprOcc(nullptr),
        AlsoKills(NoKill) {}

  static RealOcc upKill(Instruction &I) { return RealOcc(I, UpKill); }

  static RealOcc downKill(Instruction &I) { return RealOcc(I, DownKill); }

  static RealOcc noKill(Instruction &I) { return RealOcc(I, NoKill); }
};

struct LambdaOcc final : public Occurrence {
  struct Operand {
    Occurrence *ReprOcc;
    // ^ Representative occurrence dominating this operand. nullptr = _|_.
    bool HasRealUse;
    // ^ Is there a real occurrence on some path from ReprOcc to this operand's
    // lambda? Always false for _|_ operands.
  };

  SmallVector<Operand, 8> Operands;

  // Consult the Kennedy et al. paper for these.
  bool UpSafe;
  bool CanBeAnt;
  bool Later;

  LambdaOcc(BasicBlock *Block)
      : Occurrence{-1u, Block, OccTy::Lambda}, Operands{}, UpSafe(true),
        CanBeAnt(true), Later(true) {}
};

// A faux occurrence used to detect stores to non-escaping memory that are
// redundant with respect to function exit.
RealOcc DeadOnExit;

// Factored redundancy graph representation for each maximal group of
// must-aliasing stores.
struct RedGraph {
  DenseMap<const BasicBlock *, std::list<RealOcc>> BlockOccs;
  // ^ TODO: Figure out iplist for this?
  DenseMap<const BasicBlock *, LambdaOcc> Lambdas;
  DenseMap<const Instruction *, RealOcc *> InstMap;
  // ^ Used to avoid calling getModRefInfo on known real occs during renaming,
  // and for pretty-printing. Kills are tracked with nullptr.
  MemoryLocation Loc;
  // ^ The memory location that each RealOcc mods and must-alias.
  bool Escapes;
  // ^ Upon function unwind, can Loc escape?
  bool Returned;
  // ^ Is Loc returned by the function?

  RedGraph(MemoryLocation &&Loc, RealOcc &&Starter)
      : BlockOccs(), Lambdas(), InstMap(), Loc(std::move(Loc)), Escapes(true),
        Returned(true) {
    addRealOcc(std::move(Starter));
  }

  void setEscapesReturned(const DenseSet<const Value *> &NonEscapes,
                          const DenseSet<const Value *> &Returns) {
    assert(FRG.Loc.Ptr &&
           "Real occurrences must store to an analyzable memory address.");
    const DataLayout &DL = FRG.Loc.Ptr.getModule()->getDataLayout();
    const Value *Und = GetUnderlyingObject(FRG.Loc.Ptr, DL);
    Returned = Returns.count(Und);
    Escapes = !NonEscapes.count(Und) && !([&]() {
      SmallVector<Value *, 4> Unds;
      GetUnderlyingObjects(const_cast<Value *>(Und), Unds, DL);
      return all_of(Unds, [&](Value *V) { return NonEscapes.count(V); });
    })();
  }

  const LambdaOcc *getLambda(const BasicBlock &BB) const {
    return Lambdas.count(&BB) ? &Lambdas.find(&BB)->second : nullptr;
  }

  LambdaOcc *getLambda(const BasicBlock &BB) {
    return Lambdas.count(&BB) ? &Lambdas.find(&BB)->second : nullptr;
  }

  LambdaOcc &addLambda(LambdaOcc &&L, const BasicBlock &BB) {
    return Lambdas.insert({&BB, std::move(L)}).first->second;
  }

  const RealOcc *getRealOcc(const Instruction &I) const {
    return InstMap.count(&I) ? InstMap.find(&I)->second : nullptr;
  }

  RealOcc *getRealOcc(const Instruction &I) {
    return InstMap.count(&I) ? InstMap.find(&I)->second : nullptr;
  }

  RealOcc &addRealOcc(RealOcc &&R) {
    std::list<RealOcc> &OccList = BlockOccs[R.Inst->getParent()];
    OccList.push_back(std::move(R));
    InstMap.insert({R.Inst, &OccList.back()});
    return OccList.back();
  }

  void addKillOcc(const Instruction &I) { InstMap.insert({&I, nullptr}); }

  bool isKillOcc(const Instruction &I) const {
    return InstMap.count(&I) && !InstMap.find(&I)->second;
  }
};

struct RenameState {
  RedGraph *FRG;
  unsigned *NextID;
  Occurrence *ReprOcc;
  // ^ Current representative occurrence, or nullptr for _|_.
  bool CrossedRealOcc;
  // ^ Have we crossed a real occurrence since the last non-kill occurrence?

private:
  void updateUpSafety() const {
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
    if (I)
      FRG->addKillOcc(*I);
  }

  // This works because post-dom pre-order visits each block exactly once.
  Occurrence &assignID(Occurrence &Occ) const {
    Occ.ID = *NextID;
    *NextID += 1;
    return Occ;
  }

public:
  RenameState enterBlock(const BasicBlock &BB) const {
    DEBUG(dbgs() << "Entering block " << BB.getName() << "\n");
    // Set the current repr occ to the new block's lambda, if it contains one.
    LambdaOcc *L = FRG->getLambda(BB);
    return L ? RenameState{FRG, NextID, &assignID(*L), false} : *this;
  }

  RealOcc &handleRealOcc(RealOcc &R) {
    assignID(R);
    switch (R.AlsoKills) {
    case RealOcc::DownKill:
      // DownKill means a kill on the higher-RPO-index side of the real occ.
      kill(nullptr);
    case RealOcc::NoKill: {
      CrossedRealOcc = true;
      R.ReprOcc = ReprOcc;
      // Current occ is a repr occ if we've just emerged from a kill.
      ReprOcc = ReprOcc ? ReprOcc : &R;
      return R;
    }
    case RealOcc::UpKill:
      R.ReprOcc = ReprOcc;
      kill(nullptr);
      return R;
    }
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

  void handlePredecessor(const BasicBlock &Pred) {
    if (LambdaOcc *L = FRG->getLambda(Pred))
      L->Operands.push_back({ReprOcc, CrossedRealOcc});
  }
};

// Tags an instruction that modrefs and/or may-throw. May-throws are
// significant because they count as killing occurrences for escaping stores.
struct MemOrThrow {
  Instruction *I;
  bool MemInst;
};

// Analogous to MemorySSA's AccessList, but for both memory and may-throw
// instructions, in reverse.
using BlockInsts = DenseMap<const BasicBlock *, SmallVector<MemOrThrow, 8>>;

class PostDomRenamer {
  const BlockInsts &PerBlock;
  AliasAnalysis &AA;
  PostDominatorTree &PDT;

  void computeLambdaBlocks(SmallVectorImpl<BasicBlock *> &LambdaBlocks,
                           RedGraph &FRG) {
    // Enumerate def blocks, which are all blocks containing kill and/or real
    // occurrences.
    SmallPtrSet<BasicBlock *, 8> DefBlocks;
    for (const auto &BB : FRG.BlockOccs)
      DefBlocks.insert(const_cast<BasicBlock *>(BB.first));

    // Account for kill-only blocks; if it contains a real occ, we already know
    // about it.
    for (const BlockInsts::value_type &BB : PerBlock)
      if (!FRG.BlockOccs.count(const_cast<BasicBlock *>(BB.first)))
        for (const MemOrThrow &MOT : BB.second)
          if (MOT.MemInst && AA.getModRefInfo(MOT.I, FRG.Loc) & MRI_Ref) {
            DefBlocks.insert(const_cast<BasicBlock *>(BB.first));
            break;
          }

    // Compute lambdas.
    ReverseIDFCalculator RIDF(PDT);
    RIDF.setDefiningBlocks(DefBlocks);
    RIDF.calculate(LambdaBlocks);
  }

  RenameState renameBlock(BasicBlock &BB, const RenameState &IPostDom,
                          RedGraph &FRG) {
    RenameState S = IPostDom.enterBlock(BB);
    if (PerBlock.count(&BB)) {
      for (const MemOrThrow &MOT : PerBlock.find(&BB)->second) {
        DEBUG(dbgs() << "Visiting " << *MOT.I << "\n");
        if (MOT.I->mayThrow() && FRG.Escapes)
          S.handleMayThrowKill(MOT.I);
        else if (RealOcc *R = FRG.getRealOcc(*MOT.I))
          S.handleRealOcc(*R);
        else if (MOT.MemInst) {
          ModRefInfo MRI = AA.getModRefInfo(MOT.I, FRG.Loc);
          if (MRI & MRI_Ref)
            S.handleAliasingKill(MOT.I);
          else if (MRI & MRI_Mod)
            S.handleAliasingStore(MOT.I);
        }
      }
    }

    if (&BB == &BB.getParent()->getEntryBlock())
      S.handlePostDomExit();

    for (BasicBlock *Pred : predecessors(&BB))
      S.handlePredecessor(*Pred);
    return S;
  }

public:
  PostDomRenamer &insertLambdas(RedGraph &FRG) {
    SmallVector<BasicBlock *, 8> LambdaBlocks;
    computeLambdaBlocks(LambdaBlocks, FRG);
    for (BasicBlock *BB : LambdaBlocks)
      FRG.addLambda(LambdaOcc(BB), *BB);
    return *this;
  }

  PostDomRenamer &renamePass(RedGraph &FRG) {
    unsigned NextID = 0;
    RenameState RootState{&FRG, &NextID, nullptr, false};
    if (!FRG.Escapes && !FRG.Returned) {
      // Use an exit occurrence (cf. Brethour, Stanley, Wendling 2002) to
      // detect trivially dead non-escaping non-returning stores.
      RootState.ReprOcc = &DeadOnExit;
      RootState.CrossedRealOcc = true;
      NextID = 1;
    }

    struct StackEntry {
      DomTreeNode *Node;
      DomTreeNode::iterator ChildIt;
      RenameState S;
    };

    SmallVector<StackEntry, 16> Stack;
    DomTreeNode *Root = PDT.getRootNode();
    if (Root->getBlock()) {
      // Real and unique exit block.
      DEBUG(dbgs() << "Entering root " << Root->getBlock()->getName() << "\n");
      Stack.push_back({Root, Root->begin(),
                       renameBlock(*Root->getBlock(), RootState, FRG)});
    } else
      // Multiple exits and/or infinite loops.
      for (DomTreeNode *N : *Root)
        Stack.push_back(
            {N, N->begin(), renameBlock(*N->getBlock(), RootState, FRG)});

    // Visit blocks in post-dom pre-order
    while (!Stack.empty()) {
      if (Stack.back().ChildIt == Stack.back().Node->end())
        Stack.pop_back();
      else {
        DomTreeNode *Cur = *Stack.back().ChildIt++;
        RenameState NewS = renameBlock(*Cur->getBlock(), Stack.back().S, FRG);
        if (Cur->begin() != Cur->end())
          Stack.push_back({Cur, Cur->begin(), NewS});
      }
    }
    return *this;
  }

  PostDomRenamer(const BlockInsts &PerBlock, AliasAnalysis &AA,
                 PostDominatorTree &PDT)
      : PerBlock(PerBlock), AA(AA), PDT(PDT) {}
};

struct FRGAnnot final : public AssemblyAnnotationWriter {
  const RedGraph &FRG;

  FRGAnnot(const RedGraph &FRG) : FRG(FRG) {}

  virtual void emitBasicBlockEndAnnot(const BasicBlock *BB,
                                      formatted_raw_ostream &OS) override {
    if (const LambdaOcc *L = FRG.getLambda(*BB)) {
      assert(L->Operands.size() > 1 &&
             "IDFCalculator computed an unnecessary lambda.");
      auto PrintOperand = [&](const LambdaOcc::Operand &Op) {
        if (!Op.ReprOcc)
          OS << "_|_";
        else if (Op.ReprOcc == &DeadOnExit)
          OS << "DeadOnExit";
        else
          OS << Op.ReprOcc->ID;
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
      OS << ") = " << L->ID << "\n";
    }
  }

  virtual void emitInstructionAnnot(const Instruction *I,
                                    formatted_raw_ostream &OS) override {
    auto printID = [&](const Occurrence &Occ) {
      if (&Occ == &DeadOnExit)
        OS << "DeadOnExit";
      else
        OS << Occ.ID;
    };
    if (FRG.isKillOcc(*I))
      OS << "; Kill\n";
    else if (const RealOcc *R = FRG.getRealOcc(*I)) {
      OS << "; ";
      if (R->AlsoKills == RealOcc::UpKill)
        OS << "Kill + ";
      if (R->ReprOcc) {
        OS << "Real(";
        printID(*R->ReprOcc);
      } else {
        OS << "Repr(";
        printID(*R);
      }
      OS << ")";
      if (R->AlsoKills == RealOcc::DownKill)
        OS << " + Kill";
      OS << "\n";
    }
  }
};

// If Inst has the potential to be a DSE candidate, return its write location
// and a real occurrence wrapper. Derived from old DSE.
Optional<std::pair<MemoryLocation, RealOcc>> makeRealOcc(Instruction &Inst,
                                                         AliasAnalysis &AA) {
  using std::make_pair;
  if (StoreInst *SI = dyn_cast<StoreInst>(&Inst))
    return make_pair(MemoryLocation::get(SI), RealOcc::noKill(Inst));
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst)) {
    auto Loc = MemoryLocation::getForDest(MI);
    return make_pair(Loc, (AA.getModRefInfo(MI, Loc) & MRI_Ref
                               ? RealOcc::upKill
                               : RealOcc::noKill)(Inst));
  }
  return None;
}

struct OccTracker {
  SmallVector<RedGraph, 32> Inner;

  OccTracker &push_back(MemoryLocation &&Loc, RealOcc &&R, AliasAnalysis &AA) {
    // TODO: Match faster than quadratic.
    auto OC = find_if(Inner, [&](const RedGraph &FRG) {
      return AA.alias(Loc, FRG.Loc) == MustAlias;
    });
    DEBUG(dbgs() << "Collected real occ: " << *R.Inst << "\n");
    if (OC == Inner.end())
      Inner.push_back(RedGraph(std::move(Loc), std::move(R)));
    else
      OC->addRealOcc(std::move(R));
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
  DenseSet<const Value *> Returns;

  // Record non-escaping args.
  for (Argument &Arg : F.args())
    if (Arg.hasByValOrInAllocaAttr())
      NonEscapes.insert(&Arg);

  // Roll through every instruction to collect groups of must-alias stores,
  // build reversed lists of interesting instructions per block, and enumerate
  // all non-escaping memory locations.
  for (BasicBlock &BB : F) {
    for (Instruction &I : reverse(BB)) {
      if (nonEscapingOnUnwind(I, TLI)) {
        NonEscapes.insert(&I);
        continue;
      }

      if (auto *RI = dyn_cast<ReturnInst>(&I)) {
        if (Value *RetVal = RI->getReturnValue())
          Returns.insert(
              GetUnderlyingObject(RetVal, F.getParent()->getDataLayout()));
        continue;
      }

      ModRefInfo MRI = AA.getModRefInfo(&I);
      if (MRI & MRI_ModRef || I.mayThrow()) {
        DEBUG(dbgs() << "Interesting: " << I << "\n");
        PerBlock[&BB].push_back({&I, bool(MRI & MRI_ModRef)});
        if (MRI & MRI_Mod)
          if (auto LocOcc = makeRealOcc(I, AA))
            Worklist.push_back(std::move(LocOcc->first),
                               std::move(LocOcc->second), AA);
      }
    }
  }

  if (PrintFRG) {
    for (RedGraph &FRG : Worklist.Inner) {
      assert(FRG.InstMap.size() > 0 &&
             "Occurrence groups should be non-empty.");

      // Now that NonEscapes and Returned are complete, compute escapability and
      // return-ness.
      FRG.setEscapesReturned(NonEscapes, Returned);

      PostDomRenamer(PerBlock, AA, PDT).insertLambdas(FRG).renamePass(FRG);
      FRGAnnot Annot(FRG);
      dbgs() << "Factored redundancy graph for stores to " << *FRG.Loc.Ptr
             << ":\n";
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
