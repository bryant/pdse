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
// - TODO: Handle partial overwrite tracking during the full redundancy
//   elimination phase.
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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

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

// Indexes PDSE.Worklist.
using RedIdx = unsigned;

struct Occurrence {
  unsigned ID;
  RedIdx Class;
  // ^ Index of the redundancy class that this belongs to.
  OccTy Type;

  const RealOcc *isReal() const {
    return Type == Real ? reinterpret_cast<const RealOcc *>(this) : nullptr;
  }

  const LambdaOcc *isLambda() const {
    return Type == Lambda ? reinterpret_cast<const LambdaOcc *>(this) : nullptr;
  }

  RealOcc *isReal() {
    return Type == Real ? reinterpret_cast<RealOcc *>(this) : nullptr;
  }

  LambdaOcc *isLambda() {
    return Type == Lambda ? reinterpret_cast<LambdaOcc *>(this) : nullptr;
  }
};

struct RealOcc final : public Occurrence {
  Instruction *Inst;
  Occurrence *Def;
  Optional<MemoryLocation> KillLoc;

  RealOcc(unsigned ID, Instruction &I)
      : Occurrence{ID, I.getParent(), OccTy::Real}, Inst(&I), KillLoc(None) {}

  RealOcc(unsigned ID, Instruction &I, MemoryLocation &&KillLoc)
      : RealOcc(ID, I), KillLoc(KillLoc) {}

  // FIXME: sequester volatiles into their own occ class.
  bool canDSE() const { return Inst->isUnordered(); }
};

struct LambdaOcc final : public Occurrence {
  struct Operand {
    Occurrence *Inner;

    bool hasRealUse() const { return Occ->isReal(); }

    LambdaOcc *getLambda() {
      return Occ->isReal() ? Occ->isReal()->isLambda() : Occ->isLambda();
    }

    Operand(PointerUnion<BasicBlock *, Occurrence *> Inner) : Inner(Inner) {}
  };

  struct RealUse {
    RealOcc *Occ;
    BasicBlock *Pred;

    Instruction &getInst() { return *Occ->Inst; }
  };

  BasicBlock *Block;
  SmallVector<Operand, 4> Defs;
  SmallVector<BasicBlock *, 4> NullDefs;
  SmallVector<RealUse, 4> Uses;
  // ^ All uses that alias or kill this lambda's occurrence class. A necessary
  // condition for this lambda to be up-safe is that all its uses are the same
  // class.
  SmallVector<std::pair<LambdaOcc *, Operand *>, 4> LambdaUses;
  // ^ Needed by the lambda refinement phases `CanBeAnt` and `Earlier`.

  // Consult the Kennedy et al. paper for these.
  bool UpSafe;
  bool CanBeAnt;
  bool Earlier;

  LambdaOcc(BasicBlock &Block)
      : Occurrence{-1u, OccTy::Lambda}, Block(&Block), Defs(), NullDefs(),
        Uses(), LambdaUses(), UpSafe(true), CanBeAnt(true), Earlier(true) {}

  void addUse(RealOcc &Occ, BasicBlock &Pred) { Uses.push_back({&Occ, &Pred}); }

  void addUse(LambdaOcc &L, Operand &Op) { LambdaUses.push_back({&L, &Op}); }

  LambdaOcc &addOperand(BasicBlock &Succ, Occurrence *ReprOcc) {
    if (ReprOcc) {
      Defs.push_back(ReprOcc);
      if (LambdaOcc *L = Defs.back().getLambda())
        L->addUse(*this, Defs.back());
    } else
      NullDefs.push_back(&Succ);
    return *this;
  }

  void resetUpSafe() { UpSafe = false; }

  void resetCanBeAnt() {
    CanBeAnt = false;
    Earlier = false;
  }

  void resetEarlier() { Earlier = false; }

  bool willBeAnt() const { return CanBeAnt && !Earlier; }

  static Value *getStoreOp(Instruction &I) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      return SI->getValueOperand();
    } else if (auto *MI = dyn_cast<MemSetInst>(&I)) {
      return MI->getValue();
    } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
      return MI->getRawSource();
    } else {
      llvm_unreachable("Unknown real occurrence type.");
    }
  }

  static Instruction &setStoreOp(Instruction &I, Value &V) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      SI->setOperand(0, &V);
    } else if (auto *MI = dyn_cast<MemSetInst>(&I)) {
      MI->setValue(&V);
    } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
      MI->setSource(&V);
    } else {
      llvm_unreachable("Unknown real occurrence type.");
    }
    return I;
  }

  // See if this lambda's _|_ operands can be filled in. This requires that all
  // uses of this lambda are the same instruction type and DSE-able (e.g., not
  // volatile).
  Instruction *createInsertionOcc() {
    if (willBeAnt() && !NullDefs.empty() &&
        all_of(Uses, [](const RealUse &Use) { return Use.Occ->canDSE(); })) {
      if (Uses.size() == 1) {
        return Uses[0]->Inst->clone();
      } else if (Uses.size() > 1) {
        // The closest real occ users must have the same instruction type
        auto Same = [&](const RealUse &Use) {
          return Use.getInst().getOpcode() == Uses[0].getInst().getOpcode();
        };
        if (std::all_of(std::next(Uses.begin()), Uses.end(), Same)) {
          assert(getStoreOp(Uses[0].getInst()) && "Expected store operand.");
          PHINode *P = IRBuilder<>(Block, Block->begin())
                           .CreatePHI(getStoreOp(Uses[0].getInst())->getType(),
                                      Uses.size());
          for (RealUse &Use : Uses)
            P->addIncoming(getStoreOp(Use.getInst()), *Use.Pred);
          return &setStoreOp(*Uses[0].getInst().clone(), *P);
        }
      }
    }
    return nullptr;
  }
};

// A faux occurrence used to detect stores to non-escaping memory that are
// redundant with respect to function exit.
RealOcc DeadOnExit;

// Factored redundancy graph representation for each maximal group of
// must-aliasing stores.
struct RedClass {
  MemoryLocation Loc;
  // ^ The memory location that each RealOcc mods and must-alias.
  SmallVector<RedIdx, 8> Overwrites;
  // ^ Indices of redundancy classes that can DSE this class.
  bool Escapes;
  // ^ Upon function unwind, can Loc escape?
  bool Returned;
  // ^ Is Loc returned by the function?
  SmallVector<LambdaOcc *, 8> Lambdas;

  RedClass(MemoryLocation &&Loc, bool Escapes, bool Returned)
      : Loc(std::move(Loc)), Escapes(Escapes), Returned(Returned), Lambdas() {}

private:
  using LambdaStack = SmallVector<LambdaOcc *, 16>;

  // All of the lambda occ refinement phases follow this depth-first structure
  // to propagate some lambda flag from an initial set to the rest of the graph.
  // Consult figures 8 and 10 of Kennedy et al.
  void depthFirst(void (*push)(LambdaOcc &, LambdaStack &),
                  bool (*initial)(LambdaOcc &),
                  bool (*alreadyTraversed)(LambdaOcc &L)) {
    LambdaStack Stack;

    for (LambdaOcc *L : Lambdas)
      if (initial(*L)
        push(*L, Stack);

    while (!Stack.empty()) {
        LambdaOcc &L = *Stack.pop_back_val();
        if (!alreadyTraversed(L))
          push(L, Stack);
    }
  }

  RedClass &propagateUpUnsafe() {
    auto push = [](LambdaOcc &L, LambdaStack &Stack) {
      for (LambdaOcc::Operand &Op : L.Defs)
        if (LambdaOcc *L = Op.Inner.isLambda())
          Stack.push_back(*L);
    };
    auto initialCond = [](LambdaOcc &L) { return !L.UpSafe; };
    // If the top entry of the lambda stack is up-unsafe, then it and its
    // operands already been traversed.
    auto &alreadyTraversed = initialCond;

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

  RedClass &computeCanBeAnt() {
    auto push = [](LambdaOcc &L, LambdaStack &Stack) {
      L.resetCanBeAnt();
      for (auto &LO : L.LambdaUses)
        if (!LO.second->hasRealUse() && !LO.first->UpSafe && LO.first->CanBeAnt)
          Stack.push_back(LO.first);
    };
    auto initialCond = [](LambdaOcc &L) {
      return !L.UpSafe && L.CanBeAnt && !L.NullDefs.empty();
    };
    auto alreadyTraversed = [](LambdaOcc &L) { return !L.CanBeAnt; };

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

  RedClass &computeEarlier() {
    auto push = [](LambdaOcc &L, LambdaStack &Stack) {
      L.resetEarlier();
      for (auto &LO : L.LambdaUses)
        if (LO.first->Earlier)
          Stack.push_back(LO.first);
    };
    auto initialCond = [](LambdaOcc &L) {
      return L.Earlier && any_of(L.Defs, [](const LambdaOcc::Operand &Op) {
               return Op.hasRealUse();
             });
    };
    auto alreadyTraversed = [](LambdaOcc &L) { return !L.Earlier; };

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

public:
  // If:
  //   - lambda P is repr occ to an operand of lambda Q,
  //   - Q is up-unsafe (i.e., there is a reverse path from Q to function entry
  //     that doesn't cross any real occs of Q's class), and
  //   - there are no real occs from P to Q,
  // then we can conclude that P is up-unsafe too. We use this to propagate
  // up-unsafety to the rest of the FRG.
  RedClass &willBeAnt() {
    return propagateUpUnsafe().computeCanBeAnt().computeEarlier();
  }
};

struct RenameState {
  RedGraph *FRG;
  unsigned *NextID;
  Occurrence *ReprOcc;
  // ^ Current representative occurrence, or nullptr for _|_.
  bool CrossedRealOcc;
  // ^ Have we crossed a real occurrence since the last non-kill occurrence?
  BasicBlock *LambdaPred;

private:
  void updateUpSafety() const {
    // We can immediately conclude a lambda to be up-unsafe if it
    // reverse-reaches any of the following without first crossing a real
    // occurrence:
    //   - aliasing kill
    //   - may-alias store
    //   - function entry
    // In a post-dom pre-order walk, this is equivalent to encountering any of
    // these while the current repr occ is a lambda.
    if (ReprOcc && ReprOcc->Type == OccTy::Lambda && !CrossedRealOcc)
      ReprOcc->asLambda()->resetUpSafe();
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
  RenameState enterBlock(BasicBlock &BB) const {
    DEBUG(dbgs() << "Entering block " << BB.getName() << "\n");
    if (LambdaOcc *L = FRG->getLambda(BB)) {
      // This block's lambda is the new repr occ.
      return RenameState{FRG, NextID, &assignID(*L), false, &BB};
    } else if (LambdaPred && FRG->getLambda(*LambdaPred)) {
      // Record that we've entered a lambda block's predecessor.
      RenameState S = *this;
      S.LambdaPred = &BB;
      return S;
    }
    return *this;
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
      // If lambda, track its PRE insertion candidates. These are real occs to
      // which the lambda is directly exposed.
      if (LambdaPred) {
        assert(ReprOcc->Type == OccTy::Lambda);
        ReprOcc->asLambda()->PartialOccs.push_back({&R, LambdaPred});
        LambdaPred = nullptr;
      }
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

  void handlePredecessor(const BasicBlock &Pred, BasicBlock &CurBlock) {
    if (LambdaOcc *L = FRG->getLambda(Pred)) {
      DEBUG(dbgs() << "Connecting " << CurBlock.getName() << " to lambda in "
                   << Pred.getName() << "\n");
      L->addOperand(CurBlock, ReprOcc, CrossedRealOcc);
    }
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
using BlockInsts = DenseMap<const BasicBlock *, std::list<MemOrThrow>>;

class PostDomRenamer {
  const BlockInsts &PerBlock;
  AliasAnalysis &AA;
  PostDominatorTree &PDT;

  void computeLambdaBlocks(SmallVectorImpl<BasicBlock *> &LambdaBlocks,
                           RedGraph &FRG) {
    // Enumerate def blocks, which are all blocks containing kill and/or real
    // occurrences. TODO: This could be done for all RedGraphs in a single pass
    // through the function.
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
        else if (S.ReprOcc && MOT.MemInst) {
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
      S.handlePredecessor(*Pred, BB);
    return S;
  }

public:
  PostDomRenamer &insertLambdas(RedGraph &FRG) {
    SmallVector<BasicBlock *, 8> LambdaBlocks;
    computeLambdaBlocks(LambdaBlocks, FRG);
    for (BasicBlock *BB : LambdaBlocks) {
      DEBUG(dbgs() << "Inserting lambda at " << BB->getName() << "\n");
      FRG.addLambda(LambdaOcc(BB), *BB);
    }
    return *this;
  }

  PostDomRenamer &renamePass(RedGraph &FRG) {
    unsigned NextID = 0;
    RenameState RootState{&FRG, &NextID, nullptr, false, nullptr};
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
      assert(L->Operands.size() + L->NullOperands.size() > 1 &&
             "IDFCalculator computed an unnecessary lambda.");
      auto PrintOperand = [&](const LambdaOcc::Operand &Op) {
        assert(Op.ReprOcc && "_|_ operands belong to NullOperands.");
        OS << "{" << Op.Block->getName() << ", ";
        if (Op.ReprOcc == &DeadOnExit)
          OS << "DeadOnExit";
        else
          OS << Op.ReprOcc->ID;
        if (Op.HasRealUse)
          OS << "*";
        OS << "}";
      };
      OS << "; Lambda(";
      for (const LambdaOcc::Operand &Op : L->Operands) {
        PrintOperand(Op);
        OS << ", ";
      }
      for (const BasicBlock *Pred : L->NullOperands)
        OS << "{" << Pred->getName() << ", _|_}, ";
      bool WillBeAnt = L->CanBeAnt && !L->Earlier;
      OS << ") = " << L->ID << "\t" << (L->UpSafe ? "U " : "~U ")
         << (L->CanBeAnt ? "C " : "~C ") << (L->Earlier ? "E " : "~E ")
         << (WillBeAnt ? "W" : "~W") << "\n";
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

class EscapeTracker {
  const DataLayout *DL;
  const TargetLibraryInfo *TLI;
  DenseSet<const Value *> NonEscapes;
  DenseSet<const Value *> Returns;

public:
  bool escapesOnUnwind(const Value *V) {
    if (NonEscapes.count(V))
      return false;
    if (isa<AllocaInst>(V) ||
        (isAllocLikeFn(V, &TLI) && !PointerMayBeCaptured(V, false, true))) {
      NonEscapes.insert(V);
      return false;
    }
    return true;
  }

  bool escapesOnUnwind(const MemoryLocation &Loc) {
    return escapesOnUnwind(GetUnderlyingObject(Loc.Ptr, DL));
  }

  bool returned(const Value *V) const { return Returns.count(V); }

  bool returned(const MemoryLocation &Loc) const {
    return returned(GetUnderlyingObject(Loc.Ptr, DL));
  }

  EscapeTracker(Function &F, const TargetLibraryInfo &TLI)
      : DL(F.getParent()->getDataLayout()), TLI(&TLI) {
    // Record non-escaping args.
    for (Argument &Arg : F.args())
      if (Arg.hasByValOrInAllocaAttr())
        NonEscapes.insert(&Arg);

    // Record return values.
    for (BasicBlock &BB : F)
      if (auto *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
        if (Value *RetVal = RI->getReturnValue())
          Returns.insert(GetUnderlyingObject(RetVal, DL));
  }
};

class AliasCache {
  const SmallVectorImpl<RedClass> &Worklist;
  DenseMap<std::pair<RedIdx, MemoryLocation>, AliasResult> Aliases;
  // ^ used to check if a) one class aliases with another, b) a real occ's kill
  // loc aliases a certain class.
  DenseMap<std::pair<RedIdx, const Instruction *>, ModRefInfo> MRI;

public:
  AliasAnalysis &AA;

  AliasCache(AliasAnalysis &AA) : AA(AA) {}

  AliasResult alias(RedIdx A, RedIdx B) const {
    auto Key = std::make_pair(std::min(A, B), Worklist[std::max(A, B)].Loc);
    assert(Aliases.count(Key) &&
           "Aliasing between all classes should have been pre-computed.");
    return *Aliases.find(Key);
  }

  AliasResult alias(RedIdx A, const MemoryLocation &Loc) const {
    auto Key = std::make_pair(A, Loc);
    return Aliases.count(Key) ? Aliases[Key]
                              : (Aliases[Key] = AA.alias(A.Loc, Loc));
  }

  AliasResult setAlias(RedIdx A, RedIdx B, AliasResult R) {
    auto Key = std::make_pair(std::min(A, B), Worklist[std::max(A, B)].Loc);
    return Aliases[Key] = R;
  }

  ModRefInfo getModRefInfo(RedIdx A, const Instruction &I) {
    auto Key = std::make_pair(A, &I);
    return MRI.count(Key) ? MRI[Key]
                          : (MRI[Key] = AA.getModRefInfo(&I, Worklist[A].Loc));
  }
};

using InstOrReal = PointerUnion<Instruction *, RealOcc *>;

struct BlockInfo {
  std::list<InstOrReal> Insts;
  std::list<RealOcc> Occs;
  std::list<LambdOcc> Lambdas;
};

struct PDSE {
  Function &F;
  PostDominatorTree &PDT;
  const TargetLibraryInfo &TLI;

  unsigned NextID;
  AliasCache AC;
  EscapeTracker Tracker;
  DenseMap<const BasicBlock *, BlockInfo> Blocks;
  SmallVector<Instruction *, 16> DeadStores;
  SmallVector<RedGraph, 16> Worklist;

  PDSE(Function &F, AliasAnalysis &AA, PostDominatorTree &PDT,
       const TargetLibraryInfo &TLI)
      : F(F), PDT(PDT), TLI(TLI), NextID(1), AC(AA), Tracker(F, TLI) {}

  // If Inst has the potential to be a DSE candidate, return its write location
  // and a real occurrence wrapper.
  Optional<std::pair<MemoryLocation, RealOcc>> makeRealOcc(Iruction &I) {
    using std::make_pair;
    if (auto *SI = dyn_cast<StoreI>(&I)) {
      return make_pair(MemoryLocation::get(SI), RealOcc(NextID++, I));
    } else if (auto *MI = dyn_cast<MemSetI>(&I)) {
      return make_pair(MemoryLocation::getForDest(MI), RealOcc(NextID++, I));
    } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
      // memmove, memcpy.
      return make_pair(MemoryLocation::getForDest(MI),
                       RealOcc(NextID++, I, MemoryLocation::getForSource(MI)))
    }
    return None;
  }

  RedIdx assignClass(const MemoryLocation &Loc, RealOcc &Occ,
                     DenseMap<MemoryLocation, RedIdx> &BelongsToClass) {
    SmallVector<AliasResult, 16> CachedAliases;

    if (BelongsToClass.count(Loc))
      return BelongsToClass[Loc];

    for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1) {
      RedClass &Class = Worklist[Idx];
      CachedAliases.emplace_back(AC.AA.alias(Class.Loc, Loc));
      if (CachedAliases.back() == MustAlias && Class.Loc.Size == Loc.Size) {
        Class.addRealOcc(std::move(Occ));
        BelongsToClass[Idx] = &Class;
        return Idx;
      }
    }

    // Occ doesn't belong to any existing class, so start its own.
    Worklist.emplace_back(
        RedClass(Loc, Tracker.escapesOnUnwind(Loc), Tracker.returned(Loc)));
    BelongsToClass[&Worklist.back().Loc] = &Worklist.back();
    // CachedAliases.size() == Worklist.size() + 1
    for (auto &Alias : enumerate(CachedAliases)) {
      if (Alias.Value == MustAlias) {
        // Found a class that could either overwrite or be overwritten by the
        // new class.
        if (Worklist.back().Loc.Size >= Worklist[Alias.Index].Loc.Size)
          Worklist[Alias.Index].Overwrites.push_back(Worklist.size() - 1);
        else if (Worklist.back().Loc.Size <= Worklist[Alias.Index].Loc.Size)
          Worklist.back().Overwrites.push_back(Alias.Index);
      }
      AC.setAlias(Alias.Index, Worklist.size() - 1, Alias.Value);
    }
    return Worklist.size() - 1;
  }

  void addLambda(BasicBlock &BB, RedIdx Idx) {
    Blocks[&BB].Lambdas.push_back(LambdaOcc(BB));
    Worklist[Idx].Lambdas.push_back(&Blocks[&BB].Lambdas.back());
  }

  struct RenameState {
    struct Incoming {
      Occurrence *ReprOcc;
      BasicBlock *LambdaPred;
      // ^ If ReprOcc is a lambda, then this is the predecessor (to the
      // lambda-containing block) that post-doms us.
    };

    DomTreeNode *Node;
    DomTreeNode::iterator ChildIt;
    SmallVector<Incoming, 16> States;

    static decltype(States)
    makeStates(const SmallVectorImpl<RedClass> &Worklist) {
      decltype(States) Ret(Worklist.size());
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (!Worklist[Idx].Escapes && !Worklist[Idx].Returned)
          Ret[Idx] = {&DeadOnExit, &DeadOnExit};
      return Ret;
    }

    void kill(RedIdx Idx) { States[Idx] = {nullptr, nullptr}; }

    bool live(RedIdx Idx) const { return States[Idx].ReprOcc; }

    LambdaOcc *exposedLambda(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc.isLambda() : nullptr;
    }

    RealOcc *exposedRepr() const {
      return live(Idx) ? States[Idx].ReprOcc.isReal() : nullptr;
    }

    void updateUpSafety(RedIdx Idx) {
      if (LambdaOcc *L = exposedLambda(Idx))
        L->resetUpSafe();
    }

    bool live() const { return ReprOcc; }

    bool canDSE(RealOcc &Occ, const SmallVectorImpl<RedClass> &Worklist) {
      // Can DSE if post-dommed by an overwrite.
      return Occ.canDSE() && (exposedRepr(Occ.Class) ||
                              any_of(Worklist[Occ.Class].Overwrites,
                                     [&](RedIdx R) { return exposedRepr(R); }));
    }

    void handleRealOcc(RealOcc &Occ,
                       const SmallVectorImpl<RedClass> &Worklist) {
      // Set Occ as the repr occ of its class.
      if (!live(Occ.Class))
        States[Occ.Class] = {&Occ, &Occ};
      else if (LambdaOcc *L = exposedLambda(Occ.Class)) {
        L->addUse(Occ, *LambdaPred);
        States[Occ.Class] = {&Occ, nullptr};
      }

      // Occ could stomp on an aliasing class's lambda, or outright kill another
      // class if it has a KillLoc (e.g., if it's a memcpy).
      for (RedIdx Idx = 0; Idx < States.size(); Idx += 1)
        if (Idx != Occ.Class && live(Idx)) {
          if (Occ.KillLoc && AC.alias(Idx, *Occ.KillLoc) != NoAlias)
            // TODO: link up use-def edge
            kill(Idx);
          else if (AC.alias(Idx, Occ.Class) != NoAlias)
            updateUpSafety(Idx);
        }
      return false;
    }

    void handleMayKill(Instruction &I,
                       const SmallVectorImpl<RedClass> &Worklist) {
      for (RedIdx Idx = 0; Idx < States.size(); Idx += 1)
        if (live(Idx) && Worklist[Idx].Escapes && I.mayThrow()) {
          kill(Idx);
        } else if (live(Idx)) {
          ModRefInfo MRI = AC.getModRefInfo(Idx, I);
          if (MRI & MRI_Ref)
            // Aliasing load
            kill(Idx);
          else if (MRI & MRI_Mod)
            // Aliasing store
            updateUpSafety(Idx);
        }
    }
  };

  RenameState renameBlock(BasicBlock &BB, RenameState S) {
    // Record this block if it precedes a lambda block.
    for (RenameState::Incoming &Inc : S.States)
      if (Inc.ReprOcc.isLambda() && !Inc.LambdaPred)
        Inc.LambdaPred = &BB;

    // Set repr occs to lambdas, if present.
    for (LambdaOcc &L : Blocks[&BB].Lambdas)
      S.States[L.Class] = {&L, nullptr};

    // Simultaneously rename and DSE in post-order.
    for (InstOrReal &I : reverse(Blocks[&BB].Insts))
      if (auto *Occ = I.dyn_cast<RealOcc *>()) {
        if (canDSE(*Occ, Worklist))
          DeadStores.push_back(&I);
        else
          S.handleRealOcc(*Occ, Worklist);
      } else
        // Not a real occ, but still a meminst that could kill or alias.
        S.handleMayKill(*I.get<Instruction *>(), Worklist);

    // Lambdas directly exposed to reverse-exit are up-unsafe.
    if (&BB == &BB.getParent()->getEntryBlock())
      for (LambdaOcc &L : Blocks[&BB].Lambdas)
        S.updateUpSafety(L.Class);

    // Connect to predecessor lambdas.
    for (BasicBlock *Pred : predecessors(&BB))
      for (LambdaOcc &L : Blocks[Pred].Lambdas)
        L.addOperand(BB, S.States[L.Class].ReprOcc,
                     S.States[L.Class].ReprLambda);

    return S;
  }

  void renamePass() {
    decltype(RenameState.States) Incs = RenameState::makeStates(Worklist);
    SmallVector<RenameState, 8> Stack;
    if (BasicBlock *Root = PDT.getRootNode()->getBlock())
      // Real and unique exit block.
      Stack.emplace_back(renameBlock(
          *Root, {PDT.getRootNode(), PDT.getRootNode()->begin(), Incs}));
    else
      // Multiple exits and/or infinite loops.
      for (DomTreeNode *N : *PDT.getRootNode())
        Stack.emplace_back(renameBlock(*N->getBlock(), {N, N->begin(), Incs}));

    // Visit blocks in post-dom pre-order
    while (!Stack.empty()) {
      if (Stack.back().ChildIt == Stack.back().Node->end())
        Stack.pop_back();
      else {
        DomTreeNode *Cur = *Stack.back().ChildIt++;
        RenameState NewS = renameBlock(*Cur->getBlock(), Stack.back());
        if (Cur->begin() != Cur->end())
          Stack.emplace_back({Cur, Cur->begin(), std::move(NewS)});
      }
    }
  }

  bool run() {
    DenseMap<MemoryLocation, RedIdx> BelongsToClass;
    SmallVector<SmallPtrSet<BasicBlock *, 8>, 8> DefBlocks;

    // Collect real occs and track their basic blocks.
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto LocOcc = makeRealOcc(I, AA)) {
          // Found a real occ for this instruction.
          RedIdx Idx = LocOcc->second.Class =
              assignClass(LocOcc->first, LocOcc->second, BelongsToClass);
          if (Idx > DefBlocks.size() - 1)
            DefBlocks.emplace_back({&BB});
          else
            DefBlocks[Idx].insert(&BB);
          Blocks[&BB].Occs.emplace_back(std::move(LocOcc->second));
          Blocks[&BB].Insts.emplace_back(&Blocks[&BB].Occs.back());
        } else if (AC.AA.getModRefInfo(&I))
          Blocks[&BB].Insts.emplace_back(&I);

    // Insert lambdas at reverse IDF of real occs and aliasing loads.
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      // Find kill-only blocks.
      for (BasicBlock &BB : F)
        for (InstOrReal &I : Blocks[&BB].Insts) {
          auto *Occ = I.dyn_cast<RealOcc *>();
          if ((Occ && Occ->KillLoc &&
               AC.alias(Idx, *Occ->KillLoc) != NoAlias) ||
              AC.getModRefInfo(Idx, I.dyn_cast<Instruction *>()) & MRI_Ref) {
            DefBlocks[Idx].insert(&BB);
            break;
          }
        }

      // Compute lambdas.
      ReverseIDFCalculator RIDF(PDT);
      RIDF.setDefiningBlocks(DefBlocks[Idx]);
      SmallVector<BasicBlock *, 8> LambdaBlocks;
      RIDF.calculate(LambdaBlocks);

      for (BasicBlock *BB : LambdaBlocks) {
        DEBUG(dbgs() << "Inserting lambda at " << BB->getName() << "\n");
        addLambda(*BB, Idx);
      }
    }

    renamePass();

    // Convert partial redundancies.
    DenseMap<BasicBlock *, BasicBlock *> SplitBlocks;
    for (RedClass &Class : Worklist) {
      Class.willBeAnt();
      for (LambdaOcc *L : Class.Lambdas)
        if (L->NullDefs.empty()) {
          // L is fully redundant and trivially DSEs its uses.
          for (LambdaOcc::RealUse &Use : L->Uses)
            if (Use.Occ->canDSE())
              DeadStores.push_back(Use.getInst());
        } else if (Instruction *I = L->createInsertionOcc()) {
          // L is parially redundant and can be converted.
          for (BasicBlock *Succ : L.NullDefs) {
            if (SplitBlocks.count(Succ))
              Succ = SplitBlocks[Succ];
            else if (BasicBlock *Split = SplitCriticalEdge(L->Block, Succ))
              Succ = SplitBlocks[Succ] = Split;
            else
              Succ = SplitBlocks[Succ] = Succ;
            I->insertBefore(&*Succ->begin());
          }
          for (LambdaOcc::RealUse &Use : L->Uses)
            DeadStores.push_back(Use.getInst());
        }
    }

    // DSE.
    while (!DeadStores.empty()) {
      Instruction *Dead = DeadStores.pop_back_val();
      for (Use &U : Dead->operands()) {
        Instruction *Op = dyn_cast<Instruction>(*U);
        U.set(nullptr);
        if (Op && isInstructionTriviallyDead(Op, &TLI))
          DeadStores.push_back(*Op);
      }
      Dead->eraseFromParent();
    }

    return true;
  }
};

struct OccTracker {
  SmallVector<RedGraph, 32> Inner;

  OccTracker &push_back(MemoryLocation &&Loc, RealOcc &&R, AliasAnalysis &AA) {
    // TODO: Match faster than quadratic.
    auto OC = find_if(Inner, [&](const RedGraph &FRG) {
      return AA.alias(Loc, FRG.Loc) == MustAlias && Loc.Size == FRG.Loc.Size;
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
  DenseMap<const BasicBlock *, std::list<Instruction *>> PerBlock;
  EscapeTracker Tracker(F, TLI);

  for (BasicBlock &BB : F)
    for (Instruction &I : reverse(BB))
      if (auto LocOcc = makeRealOcc(I, AA))
        if (MRI & MRI_ModRef)
          if (MRI & MRI_Mod)
            if (auto LocOcc = makeRealOcc(I, AA)) {
              DEBUG(dbgs() << "Got a loc: " << *LocOcc->first.Ptr << " + "
                           << LocOcc->first.Size << "\n");
              Worklist.push_back(std::move(LocOcc->first),
                                 std::move(LocOcc->second), AA);
            }

  for (RedGraph &FRG : Worklist.Inner) {
    // Now that NonEscapes and Returned are complete, compute escapability
    // and
    // return-ness.
    FRG.setEscapesReturned(NonEscapes, Returns, F.getParent()->getDataLayout());

    PostDomRenamer(PerBlock, AA, PDT).insertLambdas(FRG).renamePass(FRG);
    FRG.propagateUpUnsafe().willBeAnt();

    if (PrintFRG) {
      FRGAnnot Annot(FRG);
      dbgs() << "Factored redundancy graph for stores to " << *FRG.Loc.Ptr
             << ":\n";
      F.print(dbgs(), &Annot);
      dbgs() << "\n";
    }

    // Convert partial redundancies to full ones, if we can.
    for (auto &BBL : FRG.Lambdas) {
      LambdaOcc &L = BBL.second;
      if (Instruction *I = L.createInsertionOcc()) {
        for (BasicBlock *Succ : L.NullOperands) {
          BasicBlock *Split = SplitCriticalEdge(L.Block, Succ);
          BasicBlock *BB = Split ? Split : Succ;
          I->insertBefore(&*BB->begin());
          // In the future, we should not need to update PerBlock.
          PerBlock[BB].emplace_back(MemOrThrow{I, true});
        }
        L.NullOperands.clear();
      }
    }

    for (auto &BlockOccs : FRG.BlockOccs)
      for (RealOcc &R : BlockOccs.second)
        if (R.ReprOcc && (R.ReprOcc->Type == OccTy::Real ||
                          (R.ReprOcc->Type == OccTy::Lambda &&
                           R.ReprOcc->asLambda()->NullOperands.empty()))) {
          DEBUG(dbgs() << "DSEing " << *R.Inst << "\n");
          // In the future, we should not need to update PerBlock.
          if (PerBlock.count(R.Block))
            PerBlock[R.Block].erase(InstToMOT.find(R.Inst)->second);
          R.Inst->eraseFromParent();
        }
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

    return PDSE(F, getAnalysis<AAResultsWrapperPass>().getAAResults(),
                getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree(),
                getAnalysis<TargetLibraryInfoWrapperPass>().getTLI())
        .run();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PostDominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    AU.setPreservesCFG();
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
  if (!PDSE(F, AM.getResult<AAManager>(F),
            AM.getResult<PostDominatorTreeAnalysis>(F),
            AM.getResult<TargetLibraryAnalysis>(F))
           .run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<GlobalsAA>();
  return PA;
}

FunctionPass *createPDSEPass() { return new PDSELegacyPass(); }
} // end namespace llvm
