//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the helper functions for the LLVM-like linkage behavior.
// It is used by the LLVMLinker and other dialects that have same linkage
// semantics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LLVMLINKERMIXIN_H
#define MLIR_LINKER_LLVMLINKERMIXIN_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Linker/LinkerInterface.h"

namespace mlir::link {

using Linkage = LLVM::Linkage;

//===----------------------------------------------------------------------===//
// Linkage helpers
//===----------------------------------------------------------------------===//

static inline bool isExternalLinkage(Linkage linkage) {
  return linkage == Linkage::External;
}

static inline bool isAvailableExternallyLinkage(Linkage linkage) {
  return linkage == Linkage::AvailableExternally;
}

static inline bool isLinkOnceAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Linkonce;
}

static inline bool isLinkOnceODRLinkage(Linkage linkage) {
  return linkage == Linkage::LinkonceODR;
}

static inline bool isLinkOnceLinkage(Linkage linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}

static inline bool isWeakAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Weak;
}

static inline bool isWeakODRLinkage(Linkage linkage) {
  return linkage == Linkage::WeakODR;
}

static inline bool isWeakLinkage(Linkage linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}

static inline bool isAppendingLinkage(Linkage linkage) {
  return linkage == Linkage::Appending;
}

static inline bool isInternalLinkage(Linkage linkage) {
  return linkage == Linkage::Internal;
}

static inline bool isPrivateLinkage(Linkage linkage) {
  return linkage == Linkage::Private;
}

static inline bool isLocalLinkage(Linkage linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}

static inline bool isExternalWeakLinkage(Linkage linkage) {
  return linkage == Linkage::ExternWeak;
}

static inline bool isCommonLinkage(Linkage linkage) {
  return linkage == Linkage::Common;
}

static inline bool isWeakForLinker(Linkage linkage) {
  return linkage == Linkage::Weak || linkage == Linkage::WeakODR ||
         linkage == Linkage::Linkonce || linkage == Linkage::LinkonceODR ||
         linkage == Linkage::Common || linkage == Linkage::ExternWeak;
}

//===----------------------------------------------------------------------===//
// Visibility helpers
//===----------------------------------------------------------------------===//

using Visibility = LLVM::Visibility;

static inline bool isHiddenVisibility(Visibility visibility) {
  return visibility == Visibility::Hidden;
}

static inline bool isProtectedVisibility(Visibility visibility) {
  return visibility == Visibility::Protected;
}

static inline Visibility getMinVisibility(Visibility lhs, Visibility rhs) {
  if (isHiddenVisibility(lhs) || isHiddenVisibility(rhs))
    return Visibility::Hidden;
  if (isProtectedVisibility(lhs) || isProtectedVisibility(rhs))
    return Visibility::Protected;
  return Visibility::Default;
}

//===----------------------------------------------------------------------===//
// Unnamed_addr helpers
//===----------------------------------------------------------------------===//

using UnnamedAddr = LLVM::UnnamedAddr;

static bool isNoneUnnamedAddr(UnnamedAddr val) {
  return val == UnnamedAddr::None;
}

static bool isLocalUnnamedAddr(UnnamedAddr val) {
  return val == UnnamedAddr::Local;
}

static UnnamedAddr getMinUnnamedAddr(UnnamedAddr lhs, UnnamedAddr rhs) {
  if (isNoneUnnamedAddr(lhs) || isNoneUnnamedAddr(rhs))
    return UnnamedAddr::None;
  if (isLocalUnnamedAddr(lhs) || isLocalUnnamedAddr(rhs))
    return UnnamedAddr::Local;
  return UnnamedAddr::Global;
}

//===----------------------------------------------------------------------===//
// Comdat helpers
//===----------------------------------------------------------------------===//

using ComdatKind = LLVM::comdat::Comdat;

struct ComdatSelector {
  StringRef name;
  ComdatKind kind;
};

//===----------------------------------------------------------------------===//
// LLVMLinkerMixin
//===----------------------------------------------------------------------===//

template <typename DerivedLinkerInterface>
class LLVMLinkerMixin {
protected:
  const DerivedLinkerInterface &getDerived() const {
    return static_cast<const DerivedLinkerInterface &>(*this);
  }

public:
  bool isDeclarationForLinker(Operation *op) const {
    const DerivedLinkerInterface &derived = getDerived();
    if (isAvailableExternallyLinkage(derived.getLinkage(op)))
      return true;
    return derived.isDeclaration(op);
  }

  bool isLinkNeeded(Conflict pair, bool forDependency) const {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    if (pair.src == pair.dst)
      return false;

    if (derived.isComdat(pair.src))
      return true;

    Linkage srcLinkage = derived.getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage))
      return true;

    bool alreadyDefinedOrDeclared =
        pair.dst && !isLocalLinkage(derived.getLinkage(pair.dst));
    bool alreadyDeclared =
        alreadyDefinedOrDeclared && derived.isDeclaration(pair.dst);

    // Don't import globals that are already defined
    if (derived.shouldLinkOnlyNeeded() && !alreadyDeclared && !forDependency)
      return false;

    // Private dependencies are gonna be renamed and linked
    if (isLocalLinkage(srcLinkage))
      return forDependency;

    // Always import dependencies that are not yet defined or declared
    if (forDependency && !alreadyDefinedOrDeclared)
      return true;

    if (derived.isDeclaration(pair.src))
      return false;

    if (derived.shouldOverrideFromSrc())
      return true;

    if (pair.dst)
      return true;

    // Linkage specifies to keep operation only in source
    return !(isLinkOnceLinkage(srcLinkage) ||
             isAvailableExternallyLinkage(srcLinkage));
  }

  LogicalResult verifyLinkageCompatibility(Conflict pair) const {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    assert(derived.canBeLinked(pair.dst) && "expected linkable operation");

    auto linkError = [&](const Twine &error) -> LogicalResult {
      return pair.src->emitError(error) << " dst: " << pair.dst->getLoc();
    };

    Linkage srcLinkage = derived.getLinkage(pair.src);
    Linkage dstLinkage = derived.getLinkage(pair.dst);

    if (isAppendingLinkage(srcLinkage) != isAppendingLinkage(dstLinkage)) {
      return linkError("Mismatched appending linkage");
    }

    if (isAppendingLinkage(srcLinkage)) {
      if (derived.isConstant(pair.src) != derived.isConstant(pair.dst))
        return linkError(
            "Appending variables with different constness need to be linked!");

      if (derived.getAlignment(pair.src) != derived.getAlignment(pair.dst))
        return linkError(
            "Appending variables with different alignment need to be linked!");

      if (derived.getVisibility(pair.src) != derived.getVisibility(pair.dst))
        return linkError(
            "Appending variables with different visibility need to be linked!");

      if (derived.getUnnamedAddr(pair.src) != derived.getUnnamedAddr(pair.dst))
        return linkError("Appending variables with different unnamed_addr need "
                         "to be linked!");

      if (derived.getSection(pair.src) != derived.getSection(pair.dst))
        return linkError(
            "Appending variables with different section need to be linked!");

      if (derived.getAddressSpace(pair.src) !=
          derived.getAddressSpace(pair.dst))
        return linkError("Appending variables with different address space "
                         "need to be linked!");
    }
    return success();
  }

  ConflictResolution getConflictResolution(Conflict pair) const {
    const DerivedLinkerInterface &derived = getDerived();
    assert(derived.canBeLinked(pair.src) && "expected linkable operation");
    assert(derived.canBeLinked(pair.dst) && "expected linkable operation");

    Linkage srcLinkage = derived.getLinkage(pair.src);
    Linkage dstLinkage = derived.getLinkage(pair.dst);

    Visibility srcVisibility = derived.getVisibility(pair.src);
    Visibility dstVisibility = derived.getVisibility(pair.dst);
    Visibility visibility = getMinVisibility(srcVisibility, dstVisibility);

    derived.setVisibility(pair.src, visibility);
    derived.setVisibility(pair.dst, visibility);

    UnnamedAddr srcUnnamedAddr = derived.getUnnamedAddr(pair.src);
    UnnamedAddr dstUnnamedAddr = derived.getUnnamedAddr(pair.dst);

    UnnamedAddr unnamedAddr = getMinUnnamedAddr(srcUnnamedAddr, dstUnnamedAddr);
    derived.setUnnamedAddr(pair.src, unnamedAddr);
    derived.setUnnamedAddr(pair.dst, unnamedAddr);

    const bool srcIsDeclaration = isDeclarationForLinker(pair.src);
    const bool dstIsDeclaration = isDeclarationForLinker(pair.dst);

    if (isAppendingLinkage(dstLinkage)) {
      return ConflictResolution::LinkFromSrc;
    }

    if (isAvailableExternallyLinkage(srcLinkage) && dstIsDeclaration) {
      return ConflictResolution::LinkFromSrc;
    }

    // If both `src` and `dst` are declarations, we can ignore the conflict
    // and keep the `dst` declaration.
    if (srcIsDeclaration && dstIsDeclaration)
      return ConflictResolution::LinkFromDst;

    // If the `dst` is a declaration import `src` definition
    // Link an available_externally over a declaration.
    if (dstIsDeclaration && !srcIsDeclaration)
      return ConflictResolution::LinkFromSrc;

    // Conflicting private values are to be renamed.
    if (isLocalLinkage(dstLinkage))
      return ConflictResolution::LinkFromBothAndRenameDst;

    if (isLocalLinkage(srcLinkage))
      return ConflictResolution::LinkFromBothAndRenameSrc;

    if (isLinkOnceLinkage(srcLinkage))
      return ConflictResolution::LinkFromDst;

    if (isCommonLinkage(srcLinkage)) {
      if (isLinkOnceLinkage(dstLinkage) || isWeakLinkage(dstLinkage))
        return ConflictResolution::LinkFromSrc;
      if (!isCommonLinkage(dstLinkage))
        return ConflictResolution::LinkFromDst;
      if (derived.getBitWidth(pair.src) > derived.getBitWidth(pair.dst))
        return ConflictResolution::LinkFromSrc;
      return ConflictResolution::LinkFromDst;
    }

    if (isWeakForLinker(srcLinkage)) {
      assert(!isExternalWeakLinkage(dstLinkage));
      assert(!isAvailableExternallyLinkage(dstLinkage));
      if (isLinkOnceLinkage(dstLinkage) && isWeakLinkage(srcLinkage)) {
        return ConflictResolution::LinkFromSrc;
      }
      // No need to link the `src`
      return ConflictResolution::LinkFromDst;
    }

    if (isWeakForLinker(dstLinkage)) {
      assert(isExternalLinkage(srcLinkage));
      return ConflictResolution::LinkFromSrc;
    }

    std::optional<ComdatSelector> srcComdatSel = derived.getComdatSelector(pair.src);
    std::optional<ComdatSelector> dstComdatSel = derived.getComdatSelector(pair.dst);
    if (srcComdatSel.has_value() && dstComdatSel.has_value()) {
      auto srcComdatName = srcComdatSel->name;
      auto dstComdatName = dstComdatSel->name;
      auto srcComdat = srcComdatSel->kind;
      auto dstComdat = dstComdatSel->kind;
      if (srcComdatName != dstComdatName) {
          llvm_unreachable("Comdat selector names don't match");
      }
      if (srcComdat != dstComdat) {
          llvm_unreachable("Comdat selector kinds don't match");
      }

      if (srcComdat == mlir::LLVM::comdat::Comdat::Any) {
          return ConflictResolution::LinkFromDst;
      }
      if (srcComdat == mlir::LLVM::comdat::Comdat::NoDeduplicate) {
          return ConflictResolution::Failure;
      }
      llvm_unreachable("unimplemented comdat kind");
    }

    llvm_unreachable("unimplemented conflict resolution");
  }
};

//===----------------------------------------------------------------------===//
// SymbolAttrLLVMLinkerMixin
//===----------------------------------------------------------------------===//

template <typename DerivedLinkerInterface>
class SymbolAttrLLVMLinkerInterface
    : public SymbolAttrLinkerInterface,
      public LLVMLinkerMixin<DerivedLinkerInterface> {
public:
  using SymbolAttrLinkerInterface::SymbolAttrLinkerInterface;

  using LinkerMixin = LLVMLinkerMixin<DerivedLinkerInterface>;

  bool isLinkNeeded(Conflict pair, bool forDependency) const override {
    return LinkerMixin::isLinkNeeded(pair, forDependency);
  }

  LogicalResult verifyLinkageCompatibility(Conflict pair) const override {
    return LinkerMixin::verifyLinkageCompatibility(pair);
  }

  ConflictResolution getConflictResolution(Conflict pair) const override {
    return LinkerMixin::getConflictResolution(pair);
  }

  LogicalResult resolveConflict(Conflict pair,
                                ConflictResolution resolution) override {
    auto &derived = LinkerMixin::getDerived();
    if (resolution == ConflictResolution::LinkFromSrc &&
        isAppendingLinkage(derived.getLinkage(pair.src))) {
      auto &toAppend = append[derived.getSymbol(pair.src)];
      if (toAppend.empty())
        toAppend.push_back(pair.dst);
      if (!derived.isDeclaration(pair.src)) {
        toAppend.push_back(pair.src);
      }
    }
    return SymbolAttrLinkerInterface::resolveConflict(pair, resolution);
  }

protected:
  // Operations to append together
  llvm::StringMap<llvm::SmallVector<Operation *, 2>> append;
};
} // namespace mlir::link

#endif // MLIR_LINKER_LLVMLINKERMIXIN_H
