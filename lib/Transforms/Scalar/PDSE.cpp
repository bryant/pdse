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
#include "llvm/Transforms/Utils/Local.h"

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
    return Type == OccTy::Real ? reinterpret_cast<const RealOcc *>(this)
                               : nullptr;
  }

  const LambdaOcc *isLambda() const {
    return Type == OccTy::Lambda ? reinterpret_cast<const LambdaOcc *>(this)
                                 : nullptr;
  }

  RealOcc *isReal() {
    return Type == OccTy::Real ? reinterpret_cast<RealOcc *>(this) : nullptr;
  }

  LambdaOcc *isLambda() {
    return Type == OccTy::Lambda ? reinterpret_cast<LambdaOcc *>(this)
                                 : nullptr;
  }

  RedIdx setClass(RedIdx Class_) { return Class = Class_; }
};

struct RealOcc final : public Occurrence {
  Instruction *Inst;
  Occurrence *Def;
  Optional<MemoryLocation> KillLoc;

  RealOcc(unsigned ID, Instruction &I)
      : Occurrence{ID, -1u, OccTy::Real}, Inst(&I), KillLoc(None) {}

  RealOcc(unsigned ID, Instruction &I, MemoryLocation &&KillLoc)
      : Occurrence{ID, -1u, OccTy::Real}, Inst(&I), KillLoc(KillLoc) {}

  bool canDSE() const {
    if (auto *SI = dyn_cast<StoreInst>(Inst)) {
      return SI->isUnordered();
    } else if (auto *MI = dyn_cast<MemIntrinsic>(Inst)) {
      return !MI->isVolatile();
    } else {
      llvm_unreachable("Unknown real occurrence type.");
    }
  }
};

struct RedClass;

struct LambdaOcc final : public Occurrence {
  struct Operand {
    Occurrence *Inner;

    bool hasRealUse() const { return Inner->isReal(); }

    LambdaOcc *getLambda() {
      return Inner->isReal() ? Inner->isReal()->isLambda() : Inner->isLambda();
    }
  };

  struct RealUse {
    RealOcc *Occ;
    BasicBlock *Pred;

    Instruction &getInst() { return *Occ->Inst; }

    const Instruction &getInst() const { return *Occ->Inst; }
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

  LambdaOcc(BasicBlock &Block, RedIdx Class)
      : Occurrence{-1u, Class, OccTy::Lambda}, Block(&Block), Defs(),
        NullDefs(), Uses(), LambdaUses(), UpSafe(true), CanBeAnt(true),
        Earlier(true) {}

  void addUse(RealOcc &Occ, BasicBlock &Pred) { Uses.push_back({&Occ, &Pred}); }

  void addUse(LambdaOcc &L, Operand &Op) { LambdaUses.push_back({&L, &Op}); }

  LambdaOcc &addOperand(BasicBlock &Succ, Occurrence *ReprOcc) {
    if (ReprOcc) {
      Defs.push_back(Operand{ReprOcc});
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

  // See if this lambda's _|_ operands can be filled in. This requires that
  // all
  // uses of this lambda are the same instruction type and DSE-able (e.g., not
  // volatile).
  Instruction *createInsertionOcc() {
    if (willBeAnt() && !NullDefs.empty() &&
        all_of(Uses, [](const RealUse &Use) { return Use.Occ->canDSE(); })) {
      if (Uses.size() == 1) {
        // If there's only one use, PRE can happen even if volatile.
        return Uses[0].getInst().clone();
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
            P->addIncoming(getStoreOp(Use.getInst()), Use.Pred);
          return &setStoreOp(*Uses[0].getInst().clone(), *P);
        }
      }
    }
    return nullptr;
  }

  raw_ostream &print(raw_ostream &, const SmallVectorImpl<RedClass> &) const;
};

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

  RedClass(MemoryLocation Loc, bool Escapes, bool Returned)
      : Loc(std::move(Loc)), Escapes(Escapes), Returned(Returned), Lambdas() {}

private:
  using LambdaStack = SmallVector<LambdaOcc *, 16>;

  // All of the lambda occ refinement phases follow this depth-first structure
  // to propagate some lambda flag from an initial set to the rest of the
  // graph.
  // Consult figures 8 and 10 of Kennedy et al.
  void depthFirst(void (*push)(LambdaOcc &, LambdaStack &),
                  bool (*initial)(LambdaOcc &),
                  bool (*alreadyTraversed)(LambdaOcc &L)) {
    LambdaStack Stack;

    for (LambdaOcc *L : Lambdas)
      if (initial(*L))
        push(*L, Stack);

    while (!Stack.empty()) {
      LambdaOcc &L = *Stack.pop_back_val();
      if (!alreadyTraversed(L))
        push(L, Stack);
    }
  }

  // If lambda P is repr occ to an operand of lambda Q and:
  //   - Q is up-unsafe (i.e., there is a reverse path from Q to function
  //   entry
  //     that doesn't cross any real occs of Q's class), and
  //   - there are no real occs from P to Q,
  // then we can conclude that P is up-unsafe too. We use this to propagate
  // up-unsafety to the rest of the FRG.
  RedClass &propagateUpUnsafe() {
    auto push = [](LambdaOcc &L, LambdaStack &Stack) {
      for (LambdaOcc::Operand &Op : L.Defs)
        if (LambdaOcc *L = Op.Inner->isLambda())
          Stack.push_back(L);
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
  RedClass &willBeAnt() {
    return propagateUpUnsafe().computeCanBeAnt().computeEarlier();
  }
};

raw_ostream &LambdaOcc::print(raw_ostream &O,
                              const SmallVectorImpl<RedClass> &Worklist) const {
  const MemoryLocation &Loc = Worklist[Class].Loc;
  O << "Lambda @ " << Block->getName() << " (" << *Loc.Ptr << " x " << Loc.Size
    << ") [" << (UpSafe ? "U " : "!U ") << (CanBeAnt ? "C " : "!C ")
    << (Earlier ? "E " : "!E ") << (willBeAnt() ? "W" : "!W") << "]";
  return O;
}

class EscapeTracker {
  const DataLayout &DL;
  const TargetLibraryInfo &TLI;
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
      : DL(F.getParent()->getDataLayout()), TLI(TLI) {
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
  // ^ used to check if a) one class aliases with another, b) a real occ's
  // kill
  // loc aliases a certain class.
  DenseMap<std::pair<RedIdx, const Instruction *>, ModRefInfo> MRI;
  AliasAnalysis &AA;

public:
  AliasCache(const SmallVectorImpl<RedClass> &Worklist, AliasAnalysis &AA)
      : Worklist(Worklist), AA(AA) {}

  AliasResult alias(RedIdx A, RedIdx B) {
    auto Key = std::make_pair(std::min(A, B), Worklist[std::max(A, B)].Loc);
    assert(Aliases.count(Key) &&
           "Aliasing between all classes should have been pre-computed.");
    return Aliases.find(Key)->second;
  }

  AliasResult alias(RedIdx A, const MemoryLocation &Loc) {
    auto Key = std::make_pair(A, Loc);
    return Aliases.count(Key) ? Aliases[Key]
                              : (Aliases[Key] = AA.alias(Worklist[A].Loc, Loc));
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
  std::list<LambdaOcc> Lambdas;
};

struct PDSE {
  Function &F;
  AliasAnalysis &AA;
  PostDominatorTree &PDT;
  const TargetLibraryInfo &TLI;

  unsigned NextID;
  AliasCache AC;
  EscapeTracker Tracker;
  DenseMap<const BasicBlock *, BlockInfo> Blocks;
  SmallVector<Instruction *, 16> DeadStores;
  SmallVector<RedClass, 16> Worklist;
  RealOcc DeadOnExit;
  // ^ A faux occurrence used to detect stores to non-escaping memory that are
  // redundant with respect to function exit.

  PDSE(Function &F, AliasAnalysis &AA, PostDominatorTree &PDT,
       const TargetLibraryInfo &TLI)
      : F(F), AA(AA), PDT(PDT), TLI(TLI), NextID(1), AC(Worklist, AA),
        Tracker(F, TLI), DeadOnExit(0, *F.getEntryBlock().getTerminator()) {}

  // If Inst has the potential to be a DSE candidate, return its write
  // location
  // and a real occurrence wrapper.
  Optional<std::pair<MemoryLocation, RealOcc>> makeRealOcc(Instruction &I) {
    using std::make_pair;
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      return make_pair(MemoryLocation::get(SI), RealOcc(NextID++, I));
    } else if (auto *MI = dyn_cast<MemSetInst>(&I)) {
      return make_pair(MemoryLocation::getForDest(MI), RealOcc(NextID++, I));
    } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
      // memmove, memcpy.
      return make_pair(MemoryLocation::getForDest(MI),
                       RealOcc(NextID++, I, MemoryLocation::getForSource(MI)));
    }
    return None;
  }

  RedIdx assignClass(const MemoryLocation &Loc, RealOcc &Occ,
                     DenseMap<MemoryLocation, RedIdx> &BelongsToClass) {
    SmallVector<AliasResult, 16> CachedAliases;

    if (BelongsToClass.count(Loc))
      return Occ.setClass(BelongsToClass[Loc]);

    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      RedClass &Class = Worklist[Idx];
      CachedAliases.emplace_back(AA.alias(Class.Loc, Loc));
      if (CachedAliases.back() == MustAlias && Class.Loc.Size == Loc.Size)
        return Occ.setClass(BelongsToClass[Loc] = Idx);
    }

    // Occ doesn't belong to any existing class, so start a new class.
    Worklist.emplace_back(Loc, Tracker.escapesOnUnwind(Loc),
                          Tracker.returned(Loc));
    BelongsToClass[Worklist.back().Loc] = Worklist.size() - 1;

    // Cache aliasing info between new and existing classes.
    for (RedIdx Idx = 0; Idx < CachedAliases.size(); Idx += 1) {
      if (CachedAliases[Idx] == MustAlias) {
        // Found a class that could either overwrite or be overwritten by the
        // new class.
        if (Worklist.back().Loc.Size >= Worklist[Idx].Loc.Size)
          Worklist[Idx].Overwrites.push_back(Worklist.size() - 1);
        else if (Worklist.back().Loc.Size <= Worklist[Idx].Loc.Size)
          Worklist.back().Overwrites.push_back(Idx);
      }
      AC.setAlias(Idx, Worklist.size() - 1, CachedAliases[Idx]);
    }
    return Occ.setClass(Worklist.size() - 1);
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
    makeStates(const SmallVectorImpl<RedClass> &Worklist, RealOcc &DeadOnExit) {
      decltype(States) Ret(Worklist.size());
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (!Worklist[Idx].Escapes && !Worklist[Idx].Returned)
          Ret[Idx] = RenameState::Incoming{&DeadOnExit, nullptr};
      return Ret;
    }

    void kill(RedIdx Idx) { States[Idx] = {nullptr, nullptr}; }

    bool live(RedIdx Idx) const { return States[Idx].ReprOcc; }

    LambdaOcc *exposedLambda(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isLambda() : nullptr;
    }

    RealOcc *exposedRepr(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isReal() : nullptr;
    }

    void updateUpSafety(RedIdx Idx) {
      if (LambdaOcc *L = exposedLambda(Idx))
        L->resetUpSafe();
    }
  };

  bool canDSE(RealOcc &Occ, RenameState &S) {
    // Can DSE if post-dommed by an overwrite.
    return Occ.canDSE() && (S.exposedRepr(Occ.Class) ||
                            any_of(Worklist[Occ.Class].Overwrites,
                                   [&](RedIdx R) { return S.exposedRepr(R); }));
  }

  void handleRealOcc(RealOcc &Occ, RenameState &S) {
    DEBUG(dbgs() << "Hit a new occ: " << *Occ.Inst << " ("
                 << Occ.Inst->getParent()->getName() << ")\n");
    // Set Occ as the repr occ of its class.
    if (!S.live(Occ.Class))
      S.States[Occ.Class] = RenameState::Incoming{&Occ, nullptr};
    else if (LambdaOcc *L = S.exposedLambda(Occ.Class)) {
      L->addUse(Occ, *S.States[Occ.Class].LambdaPred);
      S.States[Occ.Class] = {&Occ, nullptr};
    }

    // Occ could stomp on an aliasing class's lambda, or outright kill another
    // class if it has a KillLoc (e.g., if it's a memcpy).
    for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1)
      if (Idx != Occ.Class && S.live(Idx)) {
        if (Occ.KillLoc && AC.alias(Idx, *Occ.KillLoc) != NoAlias)
          // TODO: link up use-def edge
          S.kill(Idx);
        else if (AC.alias(Idx, Occ.Class) != NoAlias)
          S.updateUpSafety(Idx);
      }
  }

  void handleMayKill(Instruction &I, RenameState &S) {
    for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1)
      if (S.live(Idx) && Worklist[Idx].Escapes && I.mayThrow()) {
        S.kill(Idx);
      } else if (S.live(Idx)) {
        ModRefInfo MRI = AC.getModRefInfo(Idx, I);
        if (MRI & MRI_Ref)
          // Aliasing load
          S.kill(Idx);
        else if (MRI & MRI_Mod)
          // Aliasing store
          S.updateUpSafety(Idx);
      }
  }

  void dse(Instruction &I) {
    DEBUG(dbgs() << "DSE-ing " << I << " (" << I.getParent()->getName()
                 << ")\n");
    DeadStores.push_back(&I);
  }

  RenameState renameBlock(BasicBlock &BB, RenameState S) {
    DEBUG(dbgs() << "Entering block " << BB.getName() << "\n");
    // Record this block if it precedes a lambda block.
    for (RenameState::Incoming &Inc : S.States)
      if (Inc.ReprOcc && Inc.ReprOcc->isLambda() && !Inc.LambdaPred)
        Inc.LambdaPred = &BB;

    // Set repr occs to lambdas, if present.
    for (LambdaOcc &L : Blocks[&BB].Lambdas)
      S.States[L.Class] = {&L, nullptr};

    // Simultaneously rename and DSE in post-order.
    for (InstOrReal &I : reverse(Blocks[&BB].Insts))
      if (auto *Occ = I.dyn_cast<RealOcc *>()) {
        if (canDSE(*Occ, S))
          dse(*Occ->Inst);
        else
          handleRealOcc(*Occ, S);
      } else
        // Not a real occ, but still a meminst that could kill or alias.
        handleMayKill(*I.get<Instruction *>(), S);

    // Lambdas directly exposed to reverse-exit are up-unsafe.
    if (&BB == &BB.getParent()->getEntryBlock())
      for (LambdaOcc &L : Blocks[&BB].Lambdas)
        S.updateUpSafety(L.Class);

    // Connect to predecessor lambdas.
    for (BasicBlock *Pred : predecessors(&BB))
      for (LambdaOcc &L : Blocks[Pred].Lambdas)
        L.addOperand(BB, S.States[L.Class].ReprOcc);

    return S;
  }

  void renamePass() {
    decltype(RenameState::States) Incs =
        RenameState::makeStates(Worklist, DeadOnExit);
    SmallVector<RenameState, 8> Stack;
    if (BasicBlock *Root = PDT.getRootNode()->getBlock())
      // Real and unique exit block.
      Stack.push_back(renameBlock(
          *Root, {PDT.getRootNode(), PDT.getRootNode()->begin(), Incs}));
    else
      // Multiple exits and/or infinite loops.
      for (DomTreeNode *N : *PDT.getRootNode())
        Stack.push_back(renameBlock(*N->getBlock(), {N, N->begin(), Incs}));

    // Visit blocks in post-dom pre-order
    while (!Stack.empty()) {
      if (Stack.back().ChildIt == Stack.back().Node->end())
        Stack.pop_back();
      else {
        DomTreeNode *Cur = *Stack.back().ChildIt++;
        RenameState NewS = renameBlock(*Cur->getBlock(), Stack.back());
        if (Cur->begin() != Cur->end()) {
          NewS.Node = Cur;
          NewS.ChildIt = Cur->begin();
          Stack.push_back(std::move(NewS));
        }
      }
    }
  }

  void convertPartialReds() {
    // Maps a lambda block successor to either itself or its split edge block.
    DenseMap<BasicBlock *, BasicBlock *> SplitBlocks;
    for (RedClass &Class : Worklist) {
      // Determine PRE-ability of this class' lambdas.
      Class.willBeAnt();
      for (LambdaOcc *L : Class.Lambdas) {
        DEBUG(L->print(dbgs() << "Trying to PRE ", Worklist) << "\n\tUses:\n";
              for (LambdaOcc::RealUse &Use
                   : L->Uses) {
                dbgs() << "\t\t" << Use.getInst() << " ("
                       << Use.getInst().getParent()->getName() << ")\n";
              } dbgs()
              << "\tDefs:\n";
              for (LambdaOcc::Operand &Def
                   : L->Defs) {
                if (RealOcc *Occ = Def.Inner->isReal())
                  dbgs() << "\t\t" << *Occ->Inst << " ("
                         << Occ->Inst->getParent()->getName() << ")\n";
                else
                  Def.getLambda()->print(dbgs() << "\t", Worklist) << "\n";
              });
        if (L->NullDefs.empty()) {
          // Already fully redundant, no PRE needed, trivially DSEs its uses.
          DEBUG(L->print(dbgs(), Worklist) << " is already fully redun\n");
          for (LambdaOcc::RealUse &Use : L->Uses)
            if (Use.Occ->canDSE())
              dse(Use.getInst());
        } else if (Instruction *I = L->createInsertionOcc()) {
          // L is partially redundant and can be PRE-ed.
          DEBUG(L->print(dbgs(), Worklist) << " can be PRE-ed with:\n\t" << *I
                                           << "\n");
          for (BasicBlock *Succ : L->NullDefs) {
            if (SplitBlocks.count(Succ))
              Succ = SplitBlocks[Succ];
            else if (BasicBlock *Split = SplitCriticalEdge(L->Block, Succ))
              Succ = SplitBlocks[Succ] = Split;
            else
              Succ = SplitBlocks[Succ] = Succ;
            I->insertBefore(&*Succ->begin());
            DEBUG(dbgs() << "Inserting into " << Succ->getName() << "\n");
          }
          for (LambdaOcc::RealUse &Use : L->Uses)
            dse(Use.getInst());
        }
      }
    }
  }

  bool run() {
    DenseMap<MemoryLocation, RedIdx> BelongsToClass;
    SmallVector<SmallPtrSet<BasicBlock *, 8>, 8> DefBlocks;

    // Collect real occs and track their basic blocks.
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto LocOcc = makeRealOcc(I)) {
          // Found a real occ for this instruction.
          RedIdx Idx =
              assignClass(LocOcc->first, LocOcc->second, BelongsToClass);
          if (Idx + 1 > DefBlocks.size())
            DefBlocks.emplace_back();
          DefBlocks[Idx].insert(&BB);
          Blocks[&BB].Occs.emplace_back(std::move(LocOcc->second));
          Blocks[&BB].Insts.emplace_back(&Blocks[&BB].Occs.back());
        } else if (AA.getModRefInfo(&I))
          Blocks[&BB].Insts.emplace_back(&I);

    // Insert lambdas at reverse IDF of real occs and aliasing loads.
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      // Find kill-only blocks.
      for (BasicBlock &BB : F)
        for (InstOrReal &I : Blocks[&BB].Insts) {
          auto *Occ = I.dyn_cast<RealOcc *>();
          auto *II = I.dyn_cast<Instruction *>();
          if ((Occ && Occ->KillLoc &&
               AC.alias(Idx, *Occ->KillLoc) != NoAlias) ||
              (II && AC.getModRefInfo(Idx, *II) & MRI_Ref)) {
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
        Blocks[BB].Lambdas.emplace_back(*BB, Idx);
        Worklist[Idx].Lambdas.push_back(&Blocks[BB].Lambdas.back());
        DEBUG(Blocks[BB].Lambdas.back().print(dbgs() << "Inserted ", Worklist)
              << "\n");
      }
    }

    renamePass();
    convertPartialReds();

    // DSE.
    while (!DeadStores.empty()) {
      Instruction *Dead = DeadStores.pop_back_val();
      for (Use &U : Dead->operands()) {
        Instruction *Op = dyn_cast<Instruction>(U);
        U.set(nullptr);
        if (Op && isInstructionTriviallyDead(Op, &TLI))
          DeadStores.push_back(Op);
      }
      Dead->eraseFromParent();
    }

    return true;
  }
};

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
