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
#include "llvm/ADT/Statistic.h"
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

STATISTIC(NumStores, "Number of stores deleted");
STATISTIC(NumPartialReds, "Number of partial redundancies converted.");

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

struct RedClass;

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

  raw_ostream &print(raw_ostream &, ArrayRef<RedClass>) const;
};

struct LambdaOcc final : public Occurrence {
  struct Operand {
    Occurrence *Inner;

    RealOcc *hasRealUse() const { return Inner->isReal(); }

    LambdaOcc *getLambda() const {
      return Inner->isReal() ? Inner->isReal()->isLambda() : Inner->isLambda();
    }

    Operand(Occurrence &Inner) : Inner(&Inner) {}
  };

  struct RealUse {
    RealOcc *Occ;
    BasicBlock *Pred;
    BasicBlock *Pred;

    Instruction &getInst() const { return *Occ->Inst; }
  };

  struct LambdaUse {
    LambdaOcc *L;
    size_t OpIdx;

    Operand &getOp() const { return L->Defs[OpIdx]; }

    LambdaUse(LambdaOcc &L, size_t OpIdx) : L(&L), OpIdx(OpIdx) {}
  };

  BasicBlock *Block;
  SmallVector<Operand, 4> Defs;
  SmallVector<BasicBlock *, 4> NullDefs;
  SmallVector<RealUse, 4> Uses;
  // ^ All uses that alias or kill this lambda's occurrence class. A necessary
  // condition for this lambda to be up-safe is that all its uses are the same
  // class.
  SmallVector<LambdaUse, 4> LambdaUses;
  // ^ Needed by the lambda refinement phases `CanBeAnt` and `Earlier`.

  // Consult the Kennedy et al. paper for these.
  bool UpSafe;
  bool CanBeAnt;
  bool Earlier;

  LambdaOcc(unsigned ID, BasicBlock &Block, RedIdx Class)
      : Occurrence{ID, Class, OccTy::Lambda}, Block(&Block), Defs(), NullDefs(),
        Uses(), LambdaUses(), UpSafe(true), CanBeAnt(true), Earlier(true) {}

  void addUse(RealOcc &Occ, BasicBlock &Pred) { Uses.push_back({&Occ, &Pred}); }

  void addUse(LambdaOcc &L, size_t OpIdx) { LambdaUses.emplace_back(L, OpIdx); }

  LambdaOcc &addOperand(BasicBlock &Succ, Occurrence *ReprOcc) {
    if (ReprOcc) {
      Defs.emplace_back(*ReprOcc);
      if (LambdaOcc *L = Defs.back().getLambda()) {
        L->addUse(*this, Defs.size() - 1);
      }
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

  raw_ostream &print(raw_ostream &, ArrayRef<RedClass>) const;
};

// Factored redundancy graph representation for each maximal group of
// must-aliasing stores.
struct RedClass {
  MemoryLocation Loc;
  // ^ The memory location that each RealOcc mods and must-alias.
  SmallVector<RedIdx, 8> Overwrites;
  // ^ Indices of redundancy classes that this class can DSE.
  SmallVector<RedIdx, 8> Interferes;
  // ^ Indices of redundancy classes that alias this class.
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
  // to propagate some lambda flag from an initial set to the rest of the graph.
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
      L.resetUpSafe();
      for (LambdaOcc::Operand &Op : L.Defs)
        if (!Op.hasRealUse())
          if (LambdaOcc *L = Op.getLambda())
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
      for (LambdaOcc::LambdaUse &Use : L.LambdaUses) {
        if (!Use.getOp().hasRealUse() && !Use.L->UpSafe && Use.L->CanBeAnt)
          Stack.push_back(Use.L);
      }
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
      for (LambdaOcc::LambdaUse &Use : L.LambdaUses)
        if (Use.L->Earlier)
          Stack.push_back(Use.L);
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

  friend raw_ostream &operator<<(raw_ostream &O, const RedClass &Class) {
    return O << *Class.Loc.Ptr << " x " << Class.Loc.Size;
  }
};

raw_ostream &RealOcc::print(raw_ostream &O, ArrayRef<RedClass> Worklist) const {
  return ID ? (O << "Real @ " << Inst->getParent()->getName() << " ("
                 << Worklist[Class] << ") " << *Inst)
            : (O << "DeadOnExit");
}

raw_ostream &LambdaOcc::print(raw_ostream &O,
                              ArrayRef<RedClass> Worklist) const {
  return O << "Lambda @ " << Block->getName() << " (" << Worklist[Class]
           << ") [" << (UpSafe ? "U " : "!U ") << (CanBeAnt ? "C " : "!C ")
           << (Earlier ? "E " : "!E ") << (willBeAnt() ? "W" : "!W") << "]";
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
  ArrayRef<RedClass> Worklist;
  DenseMap<std::pair<RedIdx, MemoryLocation>, AliasResult> Aliases;
  // ^ Caches aliases between memcpy-like kill locs with each class.
  SmallVector<SmallVector<AliasResult, 8>, 8> ClassAliases;
  // ^ Caches aliasing info between occurrence classes.
  DenseMap<std::pair<RedIdx, const Instruction *>, ModRefInfo> MRI;
  AliasAnalysis &AA;

public:
  AliasCache(ArrayRef<RedClass> Worklist, AliasAnalysis &AA)
      : Worklist(Worklist), AA(AA) {}

  AliasResult alias(RedIdx A, const MemoryLocation &Loc) {
    auto Key = std::make_pair(A, Loc);
    return Aliases.count(Key) ? Aliases[Key]
                              : (Aliases[Key] = AA.alias(Worklist[A].Loc, Loc));
  }

  AliasResult alias(RedIdx A, RedIdx B) {
    return ClassAliases[std::max(A, B)][std::min(A, B)];
  }

  decltype(ClassAliases)::value_type &push() {
    ClassAliases.emplace_back();
    return ClassAliases.back();
  }

  void pop() { ClassAliases.pop_back(); }

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
    if (BelongsToClass.count(Loc))
      return Occ.setClass(BelongsToClass[Loc]);

    auto &CachedAliases = AC.push();
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      RedClass &Class = Worklist[Idx];
      CachedAliases.emplace_back(AA.alias(Class.Loc, Loc));
      if (CachedAliases.back() == MustAlias && Class.Loc.Size == Loc.Size) {
        AC.pop();
        return Occ.setClass(BelongsToClass[Loc] = Idx);
      }
    }

    // Occ doesn't belong to any existing class, so start a new class.
    Worklist.emplace_back(Loc, Tracker.escapesOnUnwind(Loc),
                          Tracker.returned(Loc));
    RedIdx NewIdx = BelongsToClass[Worklist.back().Loc] = Worklist.size() - 1;

    // Copy must-aliases and may-alias into Overwrites and Interferes.
    for (RedIdx Idx = 0; Idx < CachedAliases.size(); Idx += 1) {
      if (CachedAliases[Idx] == MustAlias) {
        // Found a class that could either overwrite or be overwritten by the
        // new class.
        if (Worklist[NewIdx].Loc.Size >= Worklist[Idx].Loc.Size)
          Worklist[NewIdx].Overwrites.push_back(Idx);
        else if (Worklist[NewIdx].Loc.Size <= Worklist[Idx].Loc.Size)
          Worklist[Idx].Overwrites.push_back(NewIdx);
      } else if (CachedAliases[Idx] != NoAlias) {
        Worklist[Idx].Interferes.push_back(NewIdx);
        Worklist[NewIdx].Interferes.push_back(Idx);
      }
    }
    return Occ.setClass(NewIdx);
  }

  struct RenameState {
    struct Incoming {
      Occurrence *ReprOcc;
      BasicBlock *LambdaPred;
      // ^ If ReprOcc is a lambda, then this is the predecessor (to the
      // lambda-containing block) that post-doms us.
    };

    SmallVector<Incoming, 16> States;

    RenameState(SmallVectorImpl<RedClass> &Worklist, RealOcc &DeadOnExit)
        : States(Worklist.size()) {
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (!Worklist[Idx].Escapes && !Worklist[Idx].Returned)
          States[Idx] = {&DeadOnExit, nullptr};
    }

    bool live(RedIdx Idx) const { return States[Idx].ReprOcc; }

    LambdaOcc *exposedLambda(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isLambda() : nullptr;
    }

    RealOcc *exposedRepr(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isReal() : nullptr;
    }
  };

  void kill(RedIdx Idx, RenameState &S) {
    DEBUG(dbgs() << "Killing class " << Worklist[Idx] << "\n");
    S.States[Idx] = {nullptr, nullptr};
  }

  void updateUpSafety(RedIdx Idx, RenameState &S) {
    if (LambdaOcc *L = S.exposedLambda(Idx)) {
      DEBUG(L->print(dbgs() << "Setting up-unsafe: ", Worklist) << "\n");
      L->resetUpSafe();
    }
  }

  void handleRealOcc(RealOcc &Occ, RenameState &S) {
    DEBUG(Occ.print(dbgs() << "Hit a new occ: ", Worklist) << "\n");
    // Occ can't be DSE-ed, so set it as representative of its occ class.
    if (!S.live(Occ.Class))
      S.States[Occ.Class] = RenameState::Incoming{&Occ, nullptr};
    else if (LambdaOcc *L = S.exposedLambda(Occ.Class)) {
      L->addUse(Occ, *S.States[Occ.Class].LambdaPred);
      S.States[Occ.Class] = {&Occ, nullptr};
    }

    // Find out how Occ's KillLoc, if any,  interacts with incoming occ classes.
    if (Occ.KillLoc)
      // Has a load that could kill some incoming class, in addition to the same
      // store loc interaction above.
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (S.live(Idx) && AC.alias(Idx, *Occ.KillLoc) != NoAlias) {
          DEBUG(dbgs() << "KillLoc aliases: " << AC.alias(Idx, *Occ.KillLoc)
                       << "\n");
          kill(Idx, S);
        }

    // Examine interactions with its store loc.
    for (RedIdx Idx : Worklist[Occ.Class].Interferes)
      updateUpSafety(Idx, S);
    for (RedIdx Idx : Worklist[Occ.Class].Overwrites)
      if (!S.live(Idx))
        // Any of Idx's occs post-dommed by Occ can be DSE-ed (barring some
        // intervening load that aliases Idx). Since Idx is _|_, this occ is
        // Idx's new repr.
        S.States[Idx] = {&Occ, nullptr};
      else
        // Otherwise, if Idx is a lambda, this occ stomps its up-safety.
        updateUpSafety(Idx, S);
  }

  void handleMayKill(Instruction &I, RenameState &S) {
    for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1)
      if (S.live(Idx) && Worklist[Idx].Escapes && I.mayThrow()) {
        kill(Idx, S);
      } else if (S.live(Idx)) {
        ModRefInfo MRI = AC.getModRefInfo(Idx, I);
        if (MRI & MRI_Ref)
          // Aliasing load
          kill(Idx, S);
        else if (MRI & MRI_Mod)
          // Aliasing store
          updateUpSafety(Idx, S);
      }
  }

  void dse(Instruction &I) {
    DEBUG(dbgs() << "DSE-ing " << I << " (" << I.getParent()->getName()
                 << ")\n");
    ++NumStores;
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
        if (Occ.canDSE() && S.exposedRepr(Occ.Class))
          dse(*Occ->Inst);
        else
          handleRealOcc(*Occ, S);
      } else
        // Not a real occ, but still a meminst that could kill or alias.
        handleMayKill(*I.get<Instruction *>(), S);

    // Lambdas directly exposed to reverse CFG exit are up-unsafe.
    if (&BB == &BB.getParent()->getEntryBlock())
      for (RenameState::Incoming &Inc : S.States)
        if (Inc.ReprOcc && Inc.ReprOcc->isLambda())
          updateUpSafety(Inc.ReprOcc->Class, S);

    // Connect to predecessor lambdas.
    for (BasicBlock *Pred : predecessors(&BB))
      for (LambdaOcc &L : Blocks[Pred].Lambdas)
        L.addOperand(BB, S.States[L.Class].ReprOcc);

    return S;
  }

  void renamePass() {
    struct Entry {
      DomTreeNode *Node;
      DomTreeNode::iterator ChildIt;
      RenameState Inner;
    };

    SmallVector<Entry, 8> Stack;
    RenameState RootState(Worklist, DeadOnExit);
    if (BasicBlock *Root = PDT.getRootNode()->getBlock())
      // Real and unique exit block.
      Stack.push_back({PDT.getRootNode(), PDT.getRootNode()->begin(),
                       renameBlock(*Root, RootState)});
    else
      // Multiple exits and/or infinite loops.
      for (DomTreeNode *N : *PDT.getRootNode())
        Stack.push_back(
            {N, N->begin(), renameBlock(*N->getBlock(), RootState)});

    // Visit blocks in post-dom pre-order
    while (!Stack.empty()) {
      if (Stack.back().ChildIt == Stack.back().Node->end())
        Stack.pop_back();
      else {
        DomTreeNode *Cur = *Stack.back().ChildIt++;
        if (Cur->begin() != Cur->end())
          Stack.push_back({Cur, Cur->begin(),
                           renameBlock(*Cur->getBlock(), Stack.back().Inner)});
        else
          renameBlock(*Cur->getBlock(), Stack.back().Inner);
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

        DEBUG(L->print(dbgs() << "Trying to PRE ", Worklist) << "\n\tUses:\n");
        for (LambdaOcc::RealUse &Use : L->Uses)
          DEBUG(Use.Occ->print(dbgs() << "\t\t", Worklist) << "\n");

        DEBUG(dbgs() << "\tDefs:\n");
        for (LambdaOcc::Operand &Def : L->Defs) {
          if (RealOcc *Occ = Def.hasRealUse())
            DEBUG(Occ->print(dbgs() << "\t\t", Worklist) << "\n");
          else
            DEBUG(Def.getLambda()->print(dbgs() << "\t\t", Worklist) << "\n");
        }

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
          auto pre = [&](BasicBlock *Succ, Instruction *I) {
            assert(!I->getParent() && "Can only insert new instructions.");
            if (SplitBlocks.count(Succ))
              Succ = SplitBlocks[Succ];
            else if (BasicBlock *Split = SplitCriticalEdge(L->Block, Succ))
              Succ = SplitBlocks[Succ] = Split;
            else
              SplitBlocks[Succ] = Succ;
            I->insertBefore(&*Succ->begin());
            DEBUG(dbgs() << "Inserted into " << Succ->getName() << "\n");
          };

          pre(L->NullDefs[0], I);
          for (BasicBlock *Succ : make_range(std::next(std::begin(L->NullDefs)),
                                             std::end(L->NullDefs)))
            pre(Succ, I->clone());

          for (LambdaOcc::RealUse &Use : L->Uses) {
            ++NumPartialReds;
            dse(Use.getInst());
          }
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
          if ((II && II->mayThrow() && Worklist[Idx].Escapes) ||
              (Occ && Occ->KillLoc &&
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
        Blocks[BB].Lambdas.emplace_back(NextID++, *BB, Idx);
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
