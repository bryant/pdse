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
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SCCIterator.h"
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
#include "llvm/Transforms/Utils/SSAUpdater.h"

#include <forward_list>
#include <list>

#define DEBUG_TYPE "pdse"

using namespace llvm;

STATISTIC(NumStores, "Number of stores deleted");
STATISTIC(NumPartialReds, "Number of partial redundancies converted.");

static cl::opt<bool>
    PrintFRG("print-frg", cl::init(false), cl::Hidden,
             cl::desc("Print the factored redundancy graph of stores."));

namespace llvm {
namespace detail {
// Iterates over the defs of an instruction that are themselves instructions.
struct DefIter {
  filter_iterator<Instruction::const_op_iterator, bool (*)(const Value *)>
      Inner;

  DefIter(const Use *B, const Use *E)
      : Inner(B, E, [](const Value *V) { return isa<Instruction>(V); }) {}

  DefIter(const Instruction *I) : Inner(I->op_end()) {}

  DefIter &operator++() {
    ++Inner;
    return *this;
  }

  DefIter operator++(int) {
    DefIter RV = *this;
    ++Inner;
    return RV;
  }

  const Instruction *operator*() const { return cast<Instruction>(&*Inner); }

  bool operator!=(const DefIter &Other) const { return Other.Inner != Inner; }

  bool operator==(const DefIter &Other) const { return Other.Inner == Inner; }
};
}

// SSA graph of instructions where edge (I_m, I_n) iff I_n is an operand of I_m.
template <> struct GraphTraits<const Instruction *> {
  using NodeRef = const Instruction *;
  using ChildIteratorType = detail::DefIter;
  using It = ChildIteratorType;

  static NodeRef getEntryNode(NodeRef I) { return I; }

  static ChildIteratorType child_begin(NodeRef I) {
    return It(I->op_begin(), I->op_end());
  }

  static ChildIteratorType child_end(NodeRef I) { return It(I); }
};
}

namespace {
raw_ostream &operator<<(raw_ostream &OS, const MemoryLocation &Loc) {
  return OS << "MemLoc(" << *Loc.Ptr << " x " << Loc.Size << ")";
}

// Representations of factored redundancy graph elements.
enum struct OccTy {
  Real,
  Lambda,
};

struct RealOcc;
struct LambdaOcc;

// Indexes PDSE.Worklist.
using RedIdx = unsigned;

// Indexes LambdaOcc::Flags
using SubIdx = unsigned;

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
};

struct RedClass;

struct RealOcc final : public Occurrence {
  SubIdx Subclass;
  Instruction *Inst;
  Occurrence *Def;
  MemoryLocation KillLoc;

  RealOcc(unsigned ID, Instruction &I)
      : Occurrence{ID, -1u, OccTy::Real}, Subclass(-1u), Inst(&I), KillLoc() {}

  RealOcc(unsigned ID, Instruction &I, MemoryLocation &&KillLoc)
      : Occurrence{ID, -1u, OccTy::Real}, Subclass(-1u), Inst(&I),
        KillLoc(KillLoc) {}

  bool canDSE() const {
    if (auto *SI = dyn_cast<StoreInst>(Inst)) {
      return SI->isUnordered();
    } else if (auto *MI = dyn_cast<MemIntrinsic>(Inst)) {
      return !MI->isVolatile();
    } else {
      llvm_unreachable("Unknown real occurrence type.");
    }
  }

  RedIdx setClass(RedIdx Class_, SubIdx Subclass_) {
    Subclass = Subclass_;
    return Class = Class_;
  }

  friend raw_ostream &operator<<(raw_ostream &O, const RealOcc &R) {
    return R.ID ? (O << "Real @ " << R.Inst->getParent()->getName() << " ("
                     << R.Class << ") " << *R.Inst)
                : (O << "DeadOnExit");
  }
};

Value *getStoreOp(Instruction &);
Instruction &setStoreOp(Instruction &, Value &);

bool isCriticalEdge(const BasicBlock &From, const BasicBlock &To) {
  return From.getTerminator()->getNumSuccessors() > 1 && [&]() {
    for (auto B = pred_begin(&To); B != pred_end(&From); ++B)
      if (*B != &From)
        return true;
    return false;
  }();
}

bool criticalSucc(const BasicBlock &LBlock, const BasicBlock &Succ) {
  assert(LBlock.getTerminator()->getNumSuccessors() > 1 &&
         "Expected LBlock to be a lambda block.");
  return isCriticalEdge(LBlock, Succ);
}

bool cantSplit(const BasicBlock &From, const BasicBlock &To) {
  return isa<IndirectBrInst>(From.getTerminator()) || To.isEHPad();
}

bool cantPREInsert(const BasicBlock &BB) {
  return BB.isEHPad() && !isa<LandingPadInst>(BB.getFirstNonPHI());
}

struct LambdaOcc final : public Occurrence {
  struct Operand {
    BasicBlock *Succ;
    Occurrence *Def;

    RealOcc *hasRealUse() const { return Def->isReal(); }

    LambdaOcc *getLambda() const {
      return Def->isReal() ? Def->isReal()->isLambda() : Def->isLambda();
    }

    Operand(BasicBlock &Succ, Occurrence &Def) : Succ(&Succ), Def(&Def) {}
  };

  struct RealUse {
    RealOcc *Occ;

    Instruction &getInst() const { return *Occ->Inst; }
  };

  struct LambdaUse {
    LambdaOcc *L;
    size_t OpIdx;

    Operand &getOp() const { return L->Operands[OpIdx]; }

    LambdaUse(LambdaOcc &L, size_t OpIdx) : L(&L), OpIdx(OpIdx) {}
  };

  BasicBlock *Block;
  std::vector<Operand> Operands;
  std::vector<BasicBlock *> NullOperands;
  std::vector<RealUse> Uses;
  // ^ Real occurrences for which this lambda is representative (a def). Each
  // use will be in the same redundancy class as this lambda (meaning that the
  // stores they represent are same-sized and must-alias), but can have
  // different sub-class indexes (memset, memcpy, plain store, etc.).
  std::vector<LambdaUse> LambdaUses;
  // ^ These lambdas either directly use this lambda, or use a real use of this
  // lambda. Needed to propagate `CanBeAnt` and `Earlier`.
  DenseSet<const BasicBlock *> AddedSuccs;

  struct SubFlags {
    // Consult the Kennedy et al. paper for these.
    bool UpSafe;
    bool CanBeAnt;
    bool Earlier;

    SubFlags() : UpSafe(true), CanBeAnt(true), Earlier(true) {}
  };

  std::vector<SubFlags> Flags;
  // ^ Anticipation computation, indexed by subclass.

  bool upSafe(SubIdx Sub) const { return Flags[Sub].UpSafe; }

  bool canBeAnt(SubIdx Sub) const { return Flags[Sub].CanBeAnt; }

  bool earlier(SubIdx Sub) const { return Flags[Sub].Earlier; }

  LambdaOcc(unsigned ID, BasicBlock &Block, RedIdx Class,
            unsigned NumSubclasses)
      : Occurrence{ID, Class, OccTy::Lambda}, Block(&Block), Operands(),
        NullOperands(), Uses(), LambdaUses(), AddedSuccs(),
        Flags(NumSubclasses) {}

  void addUse(RealOcc &Occ) { Uses.push_back({&Occ}); }

  void addUse(LambdaOcc &L, size_t OpIdx) { LambdaUses.emplace_back(L, OpIdx); }

  LambdaOcc &addOperand(BasicBlock &Succ, Occurrence *ReprOcc) {
    if (!AddedSuccs.count(&Succ)) {
      if (ReprOcc) {
        Operands.emplace_back(Succ, *ReprOcc);
        if (LambdaOcc *L = Operands.back().getLambda())
          L->addUse(*this, Operands.size() - 1);
      } else
        NullOperands.push_back(&Succ);
      AddedSuccs.insert(&Succ);
    }
    return *this;
  }

  void resetUpSafe() {
    for (SubFlags &F : Flags)
      F.UpSafe = false;
  }

  void resetUpSafe(SubIdx Sub) { Flags[Sub].UpSafe = false; }

  void resetUpSafeExcept(SubIdx Sub_) {
    for (SubIdx Sub = 0; Sub < Flags.size(); Sub += 1)
      if (Sub_ != Sub)
        Flags[Sub].UpSafe = false;
  }

  void resetCanBeAnt(SubIdx Sub) {
    Flags[Sub].CanBeAnt = Flags[Sub].Earlier = false;
  }

  void resetEarlier(SubIdx Sub) { Flags[Sub].Earlier = false; }

  bool willBeAnt(SubIdx Sub) const {
    return Flags[Sub].CanBeAnt && !Flags[Sub].Earlier;
  }

  raw_ostream &print(raw_ostream &O, bool UsesDefs = false,
                     SubIdx *Sub = nullptr) const {
    O << "Lambda @ " << Block->getName() << " (" << Class << ")";
    if (Sub)
      dbgs() << " [" << (upSafe(*Sub) ? "U " : "!U ")
             << (canBeAnt(*Sub) ? "C " : "!C ")
             << (earlier(*Sub) ? "E " : "!E ") << (willBeAnt(*Sub) ? "W" : "!W")
             << "]";
    if (UsesDefs) {
      O << "\n";
      for (const LambdaOcc::RealUse &Use : Uses)
        dbgs() << "\tUse: " << *Use.Occ << "\n";
      for (const LambdaOcc::Operand &Def : Operands)
        if (RealOcc *Occ = Def.hasRealUse())
          dbgs() << "\tDef: " << *Occ << "\n";
        else
          Def.getLambda()->print(dbgs() << "\tDef: ") << "\n";
      for (const BasicBlock *BB : NullOperands)
        dbgs() << "\tDef: _|_ @ " << BB->getName() << "\n";
    }
    return O;
  }
};

// Factored redundancy graph representation for each maximal group of
// must-aliasing stores.
struct RedClass {
  MemoryLocation Loc;
  // ^ The memory location that each RealOcc mods and must-alias.
  std::vector<RedIdx> Overwrites;
  // ^ Indices of redundancy classes that this class can DSE.
  std::vector<RedIdx> Interferes;
  // ^ Indices of redundancy classes that may-alias this class.
  bool KilledByThrow;
  bool DeadOnExit;
  std::vector<LambdaOcc *> Lambdas;
  std::vector<Instruction *> StoreTypes;
  // ^ TODO: store these as realocc* which are representatives of their
  // subclasses.

  DenseMap<std::pair<unsigned, Type *>, SubIdx> Subclasses;
  SmallPtrSet<BasicBlock *, 8> DefBlocks;

  RedClass(MemoryLocation Loc, bool KilledByThrow, bool DeadOnExit)
      : Loc(std::move(Loc)), KilledByThrow(KilledByThrow),
        DeadOnExit(DeadOnExit), Lambdas(), StoreTypes(), Subclasses(),
        DefBlocks() {}

private:
  using LambdaStack = SmallVector<LambdaOcc *, 16>;

  // All of the lambda occ refinement phases follow this depth-first structure
  // to propagate some lambda flag from an initial set to the rest of the graph.
  // Consult figures 8 and 10 of Kennedy et al.
  void depthFirst(std::function<void(LambdaOcc &, LambdaStack &)> push,
                  std::function<bool(LambdaOcc &)> initial,
                  std::function<bool(LambdaOcc &L)> alreadyTraversed) {
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
  //   - Q is up-unsafe (i.e., there is a reverse path from Q to function entry
  //     that doesn't cross any real occs of Q's class), and
  //   - there are no real occs from P to Q,
  // then we can conclude that P is up-unsafe too. We use this to propagate
  // up-unsafety to the rest of the FRG.
  RedClass &propagateUpUnsafe(SubIdx Sub) {
    auto push = [&](LambdaOcc &L, LambdaStack &Stack) {
      L.resetUpSafe(Sub);
      for (LambdaOcc::Operand &Op : L.Operands)
        if (!Op.hasRealUse())
          if (LambdaOcc *L = Op.getLambda())
            Stack.push_back(L);
    };
    auto initialCond = [&](LambdaOcc &L) { return !L.upSafe(Sub); };
    // If the top entry of the lambda stack is up-unsafe, then it and its
    // operands already been traversed.
    auto &alreadyTraversed = initialCond;

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

  RedClass &computeCanBeAnt(SubIdx Sub) {
    auto push = [&](LambdaOcc &L, LambdaStack &Stack) {
      L.resetCanBeAnt(Sub);
      for (LambdaOcc::LambdaUse &Use : L.LambdaUses)
        if (Use.L->canBeAnt(Sub) && !Use.getOp().hasRealUse() &&
            (!Use.L->upSafe(Sub) || cantPREInsert(*Use.getOp().Succ) ||
             (criticalSucc(*Use.L->Block, *Use.getOp().Succ) &&
              cantSplit(*Use.L->Block, *Use.getOp().Succ))))
          Stack.push_back(Use.L);
    };
    auto initialCond = [&](LambdaOcc &L) {
      assert(L.canBeAnt(Sub) && "Expected initial CanBeAnt == true.");
      return (!L.upSafe(Sub) && !L.NullOperands.empty()) ||
             any_of(L.NullOperands, [&L](const BasicBlock *Succ) {
               return cantPREInsert(*Succ) || (criticalSucc(*L.Block, *Succ) &&
                                               cantSplit(*L.Block, *Succ));
             });
    };
    auto alreadyTraversed = [&](LambdaOcc &L) { return !L.canBeAnt(Sub); };

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

  RedClass &computeEarlier(SubIdx Sub) {
    auto push = [&](LambdaOcc &L, LambdaStack &Stack) {
      L.resetEarlier(Sub);
      for (LambdaOcc::LambdaUse &Use : L.LambdaUses)
        if (Use.L->earlier(Sub))
          Stack.push_back(Use.L);
    };
    auto initialCond = [&](LambdaOcc &L) {
      return L.earlier(Sub) &&
             any_of(L.Operands, [](const LambdaOcc::Operand &Op) {
               return Op.hasRealUse();
             });
    };
    auto alreadyTraversed = [&](LambdaOcc &L) { return !L.earlier(Sub); };

    depthFirst(push, initialCond, alreadyTraversed);
    return *this;
  }

public:
  RedClass &computeWillBeAnt() {
    if (Lambdas.size() > 0) {
      DEBUG(dbgs() << "Computing willBeAnt\n");
      for (SubIdx Sub = 0; Sub < numSubclasses(); Sub += 1)
        propagateUpUnsafe(Sub).computeCanBeAnt(Sub).computeEarlier(Sub);
    }
    return *this;
  }

  SubIdx numSubclasses() const { return StoreTypes.size(); }

  friend raw_ostream &operator<<(raw_ostream &O, const RedClass &Class) {
    return O << Class.Loc;
  }
};

class EscapeTracker {
  const DataLayout &DL;
  const TargetLibraryInfo &TLI;
  DenseMap<const Value *, bool> IsLocal;

public:
  bool localAlloc(const Value *V) {
    return isa<AllocaInst>(V) || isAllocLikeFn(V, &TLI);
  }

  bool localAlloc(const MemoryLocation &Loc) {
    if (IsLocal.count(Loc.Ptr))
      return IsLocal[Loc.Ptr];
    const Value *Und = GetUnderlyingObject(Loc.Ptr, DL);
    return IsLocal[Loc.Ptr] = IsLocal[Und] = localAlloc(Und);
  }

  bool capturedSansReturned(const MemoryLocation &Loc) {
    // TODO: May need to cache PointerMayBeCaptured
    return PointerMayBeCaptured(GetUnderlyingObject(Loc.Ptr, DL), false, true);
  }

  bool capturedOrReturned(const MemoryLocation &Loc) {
    // TODO: May need to cache PointerMayBeCaptured
    return PointerMayBeCaptured(GetUnderlyingObject(Loc.Ptr, DL), true, false);
  }

  EscapeTracker(Function &F, const TargetLibraryInfo &TLI)
      : DL(F.getParent()->getDataLayout()), TLI(TLI) {
    // Record non-escaping args.
    for (Argument &Arg : F.args())
      if (Arg.hasByValOrInAllocaAttr())
        IsLocal[&Arg] = true;
  }
};

// TODO: Refactor these into member methods of RealOcc.
Value *getStoreOp(Instruction &I) {
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

Instruction &setStoreOp(Instruction &I, Value &V) {
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

Value *getWriteLoc(Instruction &I) {
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    return SI->getPointerOperand();
  } else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
    return MI->getRawDest();
  } else {
    llvm_unreachable("Unknown real occurrence type.");
  }
}

Instruction &setWriteLoc(Instruction &I, Value &V) {
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    SI->setOperand(1, &V);
  } else if (auto *MI = dyn_cast<MemIntrinsic>(&I)) {
    MI->setDest(&V);
  } else {
    llvm_unreachable("Unknown real occurrence type.");
  }
  return I;
}

// If Inst has the potential to be a DSE candidate, return its write location
// and a real occurrence wrapper. TODO: Refactor this with getWriteLoc.
Optional<std::pair<MemoryLocation, RealOcc>> makeRealOcc(Instruction &I,
                                                         unsigned &NextID) {
  using std::make_pair;
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    auto Loc = MemoryLocation::get(SI);
    assert(Loc.Size != MemoryLocation::UnknownSize &&
           "Expected all stores to have known size.");
    return make_pair(Loc, RealOcc(NextID++, I));
  } else if (auto *MI = dyn_cast<MemSetInst>(&I)) {
    auto Loc = MemoryLocation::getForDest(MI);
    if (Loc.Size != MemoryLocation::UnknownSize)
      return make_pair(Loc, RealOcc(NextID++, I));
  } else if (auto *MI = dyn_cast<MemTransferInst>(&I)) {
    // memmove, memcpy.
    auto Loc = MemoryLocation::getForDest(MI);
    if (Loc.Size != MemoryLocation::UnknownSize)
      return make_pair(Loc,
                       RealOcc(NextID++, I, MemoryLocation::getForSource(MI)));
  }
  return None;
}

class InstOrReal {
  union {
    Instruction *I;
    RealOcc Occ;
  };
  bool IsOcc;

public:
  InstOrReal(RealOcc &&Occ) : Occ(std::move(Occ)), IsOcc(true) {}

  InstOrReal(Instruction &I) : I(&I), IsOcc(false) {}

  const RealOcc *getOcc() const { return IsOcc ? &Occ : nullptr; }

  RealOcc *getOcc() { return IsOcc ? &Occ : nullptr; }

  Instruction *getInst() const { return IsOcc ? nullptr : I; }
};

struct BlockInfo {
  std::list<InstOrReal> Insts;
  std::list<LambdaOcc> Lambdas;
  std::vector<RedIdx> KilledThisBlock;
};

struct FRGAnnot final : public AssemblyAnnotationWriter {
  const std::vector<RedClass> &Worklist;
  const DenseMap<const BasicBlock *, BlockInfo> &Blocks;
  const DenseMap<const Instruction *, RealOcc *> &InstMap;

  FRGAnnot(const std::vector<RedClass> &Worklist,
           const DenseMap<const BasicBlock *, BlockInfo> &Blocks,
           const DenseMap<const Instruction *, RealOcc *> &InstMap)
      : Worklist(Worklist), Blocks(Blocks), InstMap(InstMap) {}

  raw_ostream &printID(raw_ostream &OS, const Occurrence &Occ) const {
    if (Occ.ID == 0)
      OS << "DeadOnExit";
    else
      OS << Occ.Class << "." << Occ.ID;
    return OS;
  }

  void emitBasicBlockEndAnnot(const BasicBlock *BB,
                              formatted_raw_ostream &OS) override {
    if (Blocks.count(BB)) {
      for (const LambdaOcc &L : Blocks.find(BB)->second.Lambdas) {
        assert(L.Operands.size() + L.NullOperands.size() > 1 &&
               "IDFCalculator computed an unnecessary lambda.");

        auto PrintOperand = [&](const LambdaOcc::Operand &Op) {
          assert(Op.Def && "_|_ operands belong to NullOperands.");
          OS << "{" << Op.Succ->getName() << ", ";
          if (LambdaOcc *OpL = Op.getLambda())
            printID(OS, *OpL);
          else {
            assert(Op.Def->isReal());
            printID(OS, *Op.Def);
          }
          if (Op.hasRealUse())
            OS << "*";
          OS << "}";
        };

        OS << "; Lambda(";
        for (const LambdaOcc::Operand &Op : L.Operands) {
          PrintOperand(Op);
          OS << " ";
        }
        for (const BasicBlock *Pred : L.NullOperands)
          OS << "{" << Pred->getName() << ", _|_} ";
        printID(OS << ") = ", L) << "\t";

        for (SubIdx Sub = 0; Sub < Worklist[L.Class].numSubclasses();
             Sub += 1) {
          OS << "[" << (L.upSafe(Sub) ? " U" : "~U")
             << (L.canBeAnt(Sub) ? " C" : "~C")
             << (L.earlier(Sub) ? " E" : "~E")
             << (L.willBeAnt(Sub) ? " W" : "~W") << "] ";
        }
        OS << "\n";
      }
    }
  }

  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &OS) override {
    if (InstMap.count(I)) {
      if (const RealOcc *R = InstMap.find(I)->second) {
        OS << "; ";
        if (R->Def) {
          printID(OS << "Real(", *R->Def) << ")\n";
        } else
          printID(OS, *R) << " = Repr\n";
      } else
        OS << "; Inserted\n";
    }
  }
};

struct PDSE {
  Function &F;
  AliasAnalysis &AA;
  PostDominatorTree &PDT;
  const TargetLibraryInfo &TLI;

  unsigned NextID;
  DenseMap<std::pair<RedIdx, const Instruction *>, ModRefInfo> MRI;
  // ^ Caches calls to AliasAnalysis::getModRefInfo.
  DenseMap<const BasicBlock *, BlockInfo> Blocks;
  DenseMap<const Instruction *, RealOcc *> InstMap;
  // ^ For instructions that have been inserted by PRE, this is nullptr.
  std::forward_list<Instruction *> DeadStores;
  std::vector<RedClass> Worklist;
  RealOcc DeadOnExit;
  // ^ A faux occurrence used to detect stores to non-escaping memory that are
  // redundant with respect to function exit.

  PDSE(Function &F, AliasAnalysis &AA, PostDominatorTree &PDT,
       const TargetLibraryInfo &TLI)
      : F(F), AA(AA), PDT(PDT), TLI(TLI), NextID(1),
        DeadOnExit(0, *F.getEntryBlock().getTerminator()) {}

  ModRefInfo getModRefInfo(RedIdx A, const Instruction &I) {
    auto Key = std::make_pair(A, &I);
    return MRI.count(Key) ? MRI[Key]
                          : (MRI[Key] = AA.getModRefInfo(&I, Worklist[A].Loc));
  }

  RedIdx classifyLoc(const MemoryLocation &Loc,
                     DenseMap<MemoryLocation, RedIdx> &BelongsToClass,
                     EscapeTracker &Tracker) {
    DEBUG(dbgs() << "Examining store location " << Loc << "\n");
    if (BelongsToClass.count(Loc))
      return BelongsToClass[Loc];

    std::vector<AliasResult> CachedAliases(Worklist.size());
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      RedClass &Class = Worklist[Idx];
      if ((CachedAliases[Idx] = AA.alias(Class.Loc, Loc)) == MustAlias &&
          Class.Loc.Size == Loc.Size)
        return BelongsToClass[Loc] = Idx;
    }

    // Loc doesn't belong to any existing RedClass, so start a new one.
    if (Tracker.localAlloc(Loc)) {
      bool KilledByThrow = Tracker.capturedSansReturned(Loc);
      bool DeadOnExit = !Tracker.capturedOrReturned(Loc);
      DEBUG(dbgs() << "Loc " << *Loc.Ptr << " killedbythrow/deadonexit: "
                   << KilledByThrow << "/" << DeadOnExit << "\n");
      Worklist.emplace_back(Loc, KilledByThrow, DeadOnExit);
    } else {
      DEBUG(dbgs() << "Loc " << *Loc.Ptr << " not locally allocated; assuming "
                                            "killedbythrow + !deadonexit.\n");
      Worklist.emplace_back(Loc, true, false);
    }
    RedIdx NewIdx = BelongsToClass[Worklist.back().Loc] = Worklist.size() - 1;

    // Copy must-/may-aliases into Overwrites and Interferes.
    for (RedIdx Idx = 0; Idx < CachedAliases.size(); Idx += 1) {
      if (CachedAliases[Idx] == MustAlias) {
        assert(Worklist[NewIdx].Loc.Size != Worklist[Idx].Loc.Size &&
               "Loc should have been part of redundancy class Idx.");
        if (Worklist[NewIdx].Loc.Size >= Worklist[Idx].Loc.Size)
          Worklist[NewIdx].Overwrites.push_back(Idx);
        else if (Worklist[NewIdx].Loc.Size <= Worklist[Idx].Loc.Size)
          Worklist[Idx].Overwrites.push_back(NewIdx);
      } else if (CachedAliases[Idx] != NoAlias) {
        Worklist[Idx].Interferes.push_back(NewIdx);
        Worklist[NewIdx].Interferes.push_back(Idx);
      }
    }
    return NewIdx;
  }

  struct RenameState {
    struct Incoming {
      Occurrence *ReprOcc;
    };

    std::vector<Incoming> States;

    RenameState(ArrayRef<RedClass> Worklist, RealOcc &DeadOnExit)
        : States(Worklist.size()) {
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (Worklist[Idx].DeadOnExit) {
          DEBUG(dbgs() << "Class " << Worklist[Idx] << " is dead on exit.\n");
          States[Idx] = {&DeadOnExit};
        }
    }

    bool live(RedIdx Idx) const { return States[Idx].ReprOcc; }

    LambdaOcc *exposedLambda(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isLambda() : nullptr;
    }

    RealOcc *exposedRepr(RedIdx Idx) const {
      return live(Idx) ? States[Idx].ReprOcc->isReal() : nullptr;
    }
  };

  void updateUpSafety(RedIdx Idx, RenameState &S) {
    if (LambdaOcc *L = S.exposedLambda(Idx)) {
      DEBUG(L->print(dbgs() << "Setting up-unsafe: ") << "\n");
      L->resetUpSafe();
    }
  }

  void kill(RedIdx Idx, RenameState &S) {
    DEBUG(dbgs() << "Killing class " << Worklist[Idx] << "\n");
    updateUpSafety(Idx, S);
    S.States[Idx] = {nullptr};
  }

  void handleRealOcc(RealOcc &Occ, RenameState &S) {
    DEBUG(dbgs() << "Hit a new occ: " << Occ << "\n");
    if (!S.live(Occ.Class)) {
      DEBUG(dbgs() << "Setting to new repr of " << Worklist[Occ.Class] << "\n");
      S.States[Occ.Class] = RenameState::Incoming{&Occ};
    } else if (LambdaOcc *L = S.exposedLambda(Occ.Class)) {
      L->resetUpSafeExcept(Occ.Subclass);
      L->addUse(Occ);
      S.States[Occ.Class] = RenameState::Incoming{&Occ};
    }

    // Examine interactions with its store loc.
    for (RedIdx Idx : Worklist[Occ.Class].Interferes)
      updateUpSafety(Idx, S);
    for (RedIdx Idx : Worklist[Occ.Class].Overwrites)
      if (!S.live(Idx)) {
        // Any of Idx's occs post-dommed by Occ can be DSE-ed (barring some
        // intervening load that aliases Idx). Since Idx is _|_, this occ is
        // Idx's new repr.
        DEBUG(dbgs() << "Setting to repr of smaller: " << Worklist[Idx]
                     << "\n");
        S.States[Idx] = RenameState::Incoming{&Occ};
      } else
        // Otherwise, if Idx is a lambda, this occ stomps its up-safety.
        updateUpSafety(Idx, S);

    // Find out how Occ's KillLoc, if any,  interacts with incoming occ classes.
    if (Occ.KillLoc.Ptr)
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1)
        if (S.live(Idx) &&
            AA.alias(Worklist[Idx].Loc, Occ.KillLoc) != NoAlias) {
          DEBUG(dbgs() << "KillLoc aliases: "
                       << AA.alias(Worklist[Idx].Loc, Occ.KillLoc) << " ");
          kill(Idx, S);
        }
  }

  // I could throw, or read and/or modify memory. Examine which redundancy
  // classes it kills.
  void handleMayKill(Instruction &I, RenameState &S) {
    for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1)
      if (S.live(Idx) && Worklist[Idx].KilledByThrow && I.mayThrow()) {
        kill(Idx, S);
      } else if (CallInst *F = isFreeCall(&I, &TLI)) {
        // TODO: Perhaps treat free (and lifetime_end) as real occurrences that
        // !canDSE and getStoreOp == undef.
        if (!S.live(Idx)) {
          DEBUG(dbgs() << "Found free: " << *F << "\n");
          // Top of Idx's stack is _|_, so set `free` to DeadOnExit because
          // post-dominated stores directly exposed to this are redundant.
          if (AA.isMustAlias(F->getArgOperand(0), Worklist[Idx].Loc.Ptr)) {
            DEBUG(dbgs() << "Frees " << Worklist[Idx] << "\n");
            S.States[Idx] = RenameState::Incoming{&DeadOnExit};
          }
        }
      } else if (S.live(Idx)) {
        ModRefInfo MRI = getModRefInfo(Idx, I);
        if (MRI & MRI_Ref)
          // Aliasing load
          kill(Idx, S);
        else if (MRI & MRI_Mod)
          // Aliasing store
          updateUpSafety(Idx, S);
      }
  }

  void dse(RealOcc &Occ, Occurrence &KilledBy) {
    DEBUG(dbgs() << "DSE-ing " << Occ << "\n");
    if (RealOcc *R = KilledBy.isReal())
      DEBUG(dbgs() << "\tKilled by " << *R << "\n");
    else if (LambdaOcc *L = KilledBy.isLambda())
      DEBUG(L->print(dbgs() << "\tKilled by ") << "\n");
    ++NumStores;
    DeadStores.push_front(Occ.Inst);
  }

  RenameState renameBlock(BasicBlock &BB, RenameState S) {
    DEBUG(dbgs() << "Entering block " << BB.getName() << "\n");
    // Set repr occs to lambdas, if present.
    for (LambdaOcc &L : Blocks[&BB].Lambdas)
      S.States[L.Class] = {&L};

    // Simultaneously rename and DSE in post-order.
    for (InstOrReal &I : reverse(Blocks[&BB].Insts))
      if (RealOcc *Occ = I.getOcc()) {
        // Occ's defining (representative) occurrence is the top of stack.
        Occ->Def = S.States[Occ->Class].ReprOcc;
        if (Occ->canDSE() && S.exposedRepr(Occ->Class))
          // Mark for DSE immediately if fully-post-dommed by another real occ.
          dse(*Occ, *S.States[Occ->Class].ReprOcc);
        else
          handleRealOcc(*Occ, S);
      } else
        // Not a real occ, but still a meminst that could kill or alias.
        handleMayKill(*I.getInst(), S);

    // Redundancy classes with SCC-indexed memory location whose def is in this
    // block shall be killed.
    for (RedIdx Idx : Blocks[&BB].KilledThisBlock) {
      DEBUG(dbgs() << "SCC kill: " << Worklist[Idx] << "\n");
      kill(Idx, S);
    }

    if (&BB == &BB.getParent()->getEntryBlock())
      // Lambdas directly exposed to reverse CFG exit are up-unsafe.
      for (RedIdx Idx = 0; Idx < S.States.size(); Idx += 1)
        updateUpSafety(Idx, S);
    else if (!pred_empty(&BB))
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

  using SplitEdgeMap =
      DenseMap<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>;

  void insertNewOccs(LambdaOcc &L, SubIdx Sub, Instruction &Ins,
                     SSAUpdater &StoreVals, SSAUpdater &StorePtrs,
                     SplitEdgeMap &SplitBlocks) {
    auto insert = [&](BasicBlock *Succ) {
      Instruction &I = *Ins.clone();
      setStoreOp(I, *StoreVals.GetValueAtEndOfBlock(L.Block));
      setWriteLoc(I, *StorePtrs.GetValueAtEndOfBlock(L.Block));

      if (SplitBlocks.count({L.Block, Succ}))
        Succ = SplitBlocks[{L.Block, Succ}];
      else if (criticalSucc(*L.Block, *Succ)) {
        CriticalEdgeSplittingOptions Opts;
        Opts.setMergeIdenticalEdges();
        Succ = SplitBlocks[{L.Block, Succ}] =
            SplitCriticalEdge(L.Block, Succ, Opts);
      } else
        SplitBlocks[{L.Block, Succ}] = Succ;

      // Need to insert after PHINodes.
      // - TODO: Cache insertion pos..
      // - TODO: getFirstNonPHI; cache it. and we shouldn't be breaking eh pads,
      //   or inserting into catchswitches.
      BasicBlock::iterator InsPos = find_if(*Succ, [](const Instruction &I) {
        return !isa<PHINode>(&I) && !isa<LandingPadInst>(&I);
      });
      I.insertBefore(&*InsPos);
      InstMap[&I] = nullptr;
    };

    // From Kennedy et al.: "Insert holds for a phi operand if and only if the
    // following hold:
    //
    // - the phi satisfies will_be_avail; and
    // - the operand is _|_; or has_real_use is false for the operand, and the
    //   operand is defined by a phi that does not satisfy will_be_avail."
    assert(L.willBeAnt(Sub) && "Can only PRE across willBeAnt lambdas.");
    for (BasicBlock *Succ : L.NullOperands)
      insert(Succ);
    for (LambdaOcc::Operand &Op : L.Operands)
      if (!Op.hasRealUse() && Op.getLambda() && !Op.getLambda()->willBeAnt(Sub))
        insert(Op.Succ);
  }

  void convertPartialReds() {
    SplitEdgeMap SplitBlocks;
    for (RedClass &Class : Worklist) {
      // Determine PRE-ability of this class' lambdas.
      Class.computeWillBeAnt();

      // TODO: Iterate by lambda, not subclass, as the current way will iterate
      // over the same lambda (and its defs) multiple times.
      SSAUpdater StoreVals;
      SSAUpdater StorePtrs;
      for (SubIdx Sub = 0; Sub < Class.numSubclasses(); Sub += 1) {
        assert(getStoreOp(*Class.StoreTypes[Sub]) &&
               "Expected an analyzable store instruction.");
        StoreVals.Initialize(getStoreOp(*Class.StoreTypes[Sub])->getType(),
                             Class.StoreTypes[Sub]->getName());
        StorePtrs.Initialize(getWriteLoc(*Class.StoreTypes[Sub])->getType(),
                             Class.StoreTypes[Sub]->getName());

        // Collect all possible store operand definitions that will flow into
        // the inserted stores.
        for (LambdaOcc *L : Class.Lambdas) {
          DEBUG(L->print(dbgs() << "Analyzing ", true, &Sub) << "\n");
          if (L->willBeAnt(Sub))
            for (LambdaOcc::RealUse &Use : L->Uses) {
              if (Use.Occ->Subclass == Sub) {
                DEBUG(dbgs() << "Including " << *getStoreOp(Use.getInst())
                             << "\n");
                StoreVals.AddAvailableValue(Use.getInst().getParent(),
                                            getStoreOp(Use.getInst()));
                StorePtrs.AddAvailableValue(Use.getInst().getParent(),
                                            getWriteLoc(Use.getInst()));
                if (Use.Occ->canDSE()) {
                  ++NumPartialReds;
                  dse(*Use.Occ, *L);
                }
              }
            }
        }
        for (LambdaOcc *L : Class.Lambdas) {
          if (L->willBeAnt(Sub)) {
            DEBUG(L->print(dbgs() << "Trying to PRE subclass " << Sub << " ",
                           false, &Sub)
                  << "\n");
            insertNewOccs(*L, Sub, *Class.StoreTypes[Sub], StoreVals, StorePtrs,
                          SplitBlocks);
          }
        }
      }
    }
  }

  void tagSCCIndexedLocs() {
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      DEBUG(dbgs() << "Finding SCCs for " << Worklist[Idx] << "\n");
      if (auto *Def = dyn_cast<Instruction>(Worklist[Idx].Loc.Ptr))
        // Mark every block containing a phi that partakes in a non-trivial SSA
        // def SCC, because such a phi is indexed by the SCC.
        for (scc_iterator<const Instruction *> S = scc_begin(Def); !S.isAtEnd();
             ++S)
          if (S->size() > 1) {
            DEBUG(dbgs()
                  << "Pointer def for class " << Idx << " [" << *Def
                  << "] is part of an SCC. Laying a start-of-block kill at "
                  << Def->getParent()->getName() << "\n");
            Blocks[Def->getParent()].KilledThisBlock.push_back(Idx);
            break;
          }
    }
  }

  // Insert lambdas at reverse IDF of real occs and aliasing loads.
  void insertLambdas() {
    for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
      // Find kill-only blocks.
      for (BasicBlock &BB : F)
        for (const InstOrReal &I : Blocks[&BB].Insts) {
          Instruction *II = I.getInst();
          // These checks are ordered from least to most expensive.
          if ((II && II->mayThrow() && Worklist[Idx].KilledByThrow) ||
              (I.getOcc() && I.getOcc()->KillLoc.Ptr &&
               AA.alias(Worklist[Idx].Loc, I.getOcc()->KillLoc) != NoAlias) ||
              (II && getModRefInfo(Idx, *II) & MRI_Ref)) {
            DEBUG(dbgs() << "Found kill for " << Worklist[Idx] << ": ");
            if (II)
              DEBUG(dbgs() << *II << "\n");
            else
              DEBUG(dbgs() << *I.getOcc() << "\n");
            Worklist[Idx].DefBlocks.insert(&BB);
            break;
          }
        }

      // Compute lambdas.
      ReverseIDFCalculator RIDF(PDT);
      RIDF.setDefiningBlocks(Worklist[Idx].DefBlocks);
      SmallVector<BasicBlock *, 8> LambdaBlocks;
      RIDF.calculate(LambdaBlocks);

      for (BasicBlock *BB : LambdaBlocks) {
        Blocks[BB].Lambdas.emplace_back(NextID++, *BB, Idx,
                                        Worklist[Idx].numSubclasses());
        Worklist[Idx].Lambdas.push_back(&Blocks[BB].Lambdas.back());
        DEBUG(Blocks[BB].Lambdas.back().print(dbgs() << "Inserted ") << "\n");
      }
    }
  }

  void addRealOcc(RealOcc &&Occ, RedIdx Idx) {
    // Subclasses are grouped together by identical opcode and store types. For
    // instance, `store i8*` and `store i1*`, belong to the same RedIdx (they
    // are same-sized, and assuming that they must-alias) but different
    // SubIdx.
    auto Key =
        std::make_pair(Occ.Inst->getOpcode(), getStoreOp(*Occ.Inst)->getType());

    // This will add a new subclass if `Key` isn't found.
    auto Inserted =
        Worklist[Idx].Subclasses.insert({Key, Worklist[Idx].numSubclasses()});
    if (Inserted.second)
      Worklist[Idx].StoreTypes.push_back(Occ.Inst);
    Occ.setClass(Idx, Inserted.first->second);
    DEBUG(dbgs() << "Added real occ @ " << Occ.Inst->getParent()->getName()
                 << " " << *Occ.Inst << "\n\tto subclass "
                 << *Worklist[Idx].StoreTypes[Inserted.first->second]
                 << "\n\tof " << Worklist[Idx] << "\n");

    BasicBlock *BB = Occ.Inst->getParent();
    Worklist[Idx].DefBlocks.insert(BB);
    Blocks[BB].Insts.emplace_back(std::move(Occ));

    RealOcc &R = *Blocks[BB].Insts.back().getOcc();
    InstMap[R.Inst] = &R;
  }

  // Collect real occs and track their basic blocks.
  void collectOccurrences() {
    EscapeTracker Tracker(F, TLI);
    DenseMap<MemoryLocation, RedIdx> BelongsToClass;

    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (auto LocOcc = makeRealOcc(I, NextID)) {
          // Found a real occ for this instruction. Figure out which redundancy
          // class its store locbelongs to.
          RedIdx Idx = classifyLoc(LocOcc->first, BelongsToClass, Tracker);
          addRealOcc(std::move(LocOcc->second), Idx);
        } else if (AA.getModRefInfo(&I))
          Blocks[&BB].Insts.emplace_back(I);
  }

  bool run() {
    if (!PDT.getRootNode()) {
      DEBUG(dbgs() << "FIXME: ran into the PDT bug. nothing we can do.\n");
      return false;
    }

    collectOccurrences();
    insertLambdas();
    tagSCCIndexedLocs();
    renamePass();
    convertPartialReds();

    if (PrintFRG) {
      for (RedIdx Idx = 0; Idx < Worklist.size(); Idx += 1) {
        dbgs() << "Class " << Idx << ": " << Worklist[Idx].Loc
               << "\n\tOverwrites:";
        for (RedIdx Ov : Worklist[Idx].Overwrites)
          dbgs() << " " << Ov;
        dbgs() << "\n\tInterferes:";
        for (RedIdx Intf : Worklist[Idx].Interferes)
          dbgs() << " " << Intf;
        dbgs() << "\n\tKilledByThrow: " << Worklist[Idx].KilledByThrow
               << ", DeadOnExit: " << Worklist[Idx].DeadOnExit << "\n";
      }

      FRGAnnot Annot(Worklist, Blocks, InstMap);
      F.print(dbgs(), &Annot);
    }

    // DSE.
    while (!DeadStores.empty()) {
      Instruction *Dead = DeadStores.front();
      DeadStores.pop_front();
      for (Use &U : Dead->operands()) {
        Instruction *Op = dyn_cast<Instruction>(U);
        U.set(nullptr);
        if (Op && isInstructionTriviallyDead(Op, &TLI))
          DeadStores.push_front(Op);
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
  PA.preserve<GlobalsAA>();
  return PA;
}

FunctionPass *createPDSEPass() { return new PDSELegacyPass(); }
} // end namespace llvm

// vim: set shiftwidth=2 tabstop=2:
