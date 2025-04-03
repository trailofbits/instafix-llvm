//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link llvm dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

using Linkage = LLVM::Linkage;
using Visibility = LLVM::Visibility;

static Linkage getLinkage(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getLinkage();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getLinkage();
  llvm_unreachable("unexpected operation");
}

static bool isExternalLinkage(Linkage linkage) {
  return linkage == Linkage::External;
}

static bool isAvailableExternallyLinkage(Linkage linkage) {
  return linkage == Linkage::AvailableExternally;
}

static bool isLinkOnceAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Linkonce;
}

static bool isLinkOnceODRLinkage(Linkage linkage) {
  return linkage == Linkage::LinkonceODR;
}

static bool isLinkOnceLinkage(Linkage linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}

static bool isWeakAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Weak;
}

static bool isWeakODRLinkage(Linkage linkage) {
  return linkage == Linkage::WeakODR;
}

static bool isWeakLinkage(Linkage linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}

LLVM_ATTRIBUTE_UNUSED static bool isAppendingLinkage(Linkage linkage) {
  return linkage == Linkage::Appending;
}

static bool isInternalLinkage(Linkage linkage) {
  return linkage == Linkage::Internal;
}

static bool isPrivateLinkage(Linkage linkage) {
  return linkage == Linkage::Private;
}

static bool isLocalLinkage(Linkage linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}

static bool isExternalWeakLinkage(Linkage linkage) {
  return linkage == Linkage::ExternWeak;
}

static Visibility getVisibility(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getVisibility_();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getVisibility_();
  llvm_unreachable("unexpected operation");
}

static void setVisibility(Operation *op, Visibility visibility) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setVisibility_(visibility);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setVisibility_(visibility);
  llvm_unreachable("unexpected operation");
}

static bool isHiddenVisibility(Visibility visibility) {
  return visibility == Visibility::Hidden;
}

static bool isProtectedVisibility(Visibility visibility) {
  return visibility == Visibility::Protected;
}

static Visibility getMinVisibility(Visibility lhs, Visibility rhs) {
  if (isHiddenVisibility(lhs) || isHiddenVisibility(rhs))
    return Visibility::Hidden;
  if (isProtectedVisibility(lhs) || isProtectedVisibility(rhs))
    return Visibility::Protected;
  return Visibility::Default;
}

LLVM_ATTRIBUTE_UNUSED static bool isCommonLinkage(Linkage linkage) {
  return linkage == Linkage::Common;
}

LLVM_ATTRIBUTE_UNUSED static bool isValidDeclarationLinkage(Linkage linkage) {
  return isExternalWeakLinkage(linkage) || isExternalLinkage(linkage);
}

/// Whether the definition of this global may be replaced by something
/// non-equivalent at link time. For example, if a function has weak linkage
/// then the code defining it may be replaced by different code.
LLVM_ATTRIBUTE_UNUSED static bool isInterposableLinkage(Linkage linkage) {
  switch (linkage) {
  case Linkage::Weak:
  case Linkage::Linkonce:
  case Linkage::Common:
  case Linkage::ExternWeak:
    return true;

  case Linkage::AvailableExternally:
  case Linkage::LinkonceODR:
  case Linkage::WeakODR:
    // The above three cannot be overridden but can be de-refined.

  case Linkage::External:
  case Linkage::Appending:
  case Linkage::Internal:
  case Linkage::Private:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

/// Whether the definition of this global may be discarded if it is not used
/// in its compilation unit.
LLVM_ATTRIBUTE_UNUSED static bool isDiscardableIfUnused(Linkage linkage) {
  return isLinkOnceLinkage(linkage) || isLocalLinkage(linkage) ||
         isAvailableExternallyLinkage(linkage);
}

/// Whether the definition of this global may be replaced at link time.  NB:
/// Using this method outside of the code generators is almost always a
/// mistake: when working at the IR level use isInterposable instead as it
/// knows about ODR semantics.
LLVM_ATTRIBUTE_UNUSED static bool isWeakForLinker(Linkage linkage) {
  return linkage == Linkage::Weak || linkage == Linkage::WeakODR ||
         linkage == Linkage::Linkonce || linkage == Linkage::LinkonceODR ||
         linkage == Linkage::Common || linkage == Linkage::ExternWeak;
}

LLVM_ATTRIBUTE_UNUSED static bool isValidLinkage(Linkage linkage) {
  return isExternalLinkage(linkage) || isLocalLinkage(linkage) ||
         isWeakLinkage(linkage) || isLinkOnceLinkage(linkage);
}

StringRef symbol(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getSymName();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getSymName();
  llvm_unreachable("unexpected operation");
}

StringAttr symbolAttr(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getSymNameAttr();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getSymNameAttr();
  llvm_unreachable("unexpected operation");
}

/// Return true if the primary definition of this global value is outside of the
/// current translation unit.
bool isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  llvm_unreachable("unexpected operation");
}

bool isDeclarationForLinker(Operation *op) {
  if (isAvailableExternallyLinkage(getLinkage(op)))
    return true;
  return isDeclaration(op);
}

/// Returns true if this global's definition will be the one chosen by the
/// linker.
bool isStrongDefinitionForLinker(Operation *op) {
  return !(isDeclarationForLinker(op) || isWeakForLinker(getLinkage(op)));
}

unsigned getBitWidth(LLVM::GlobalOp op) {
  return op.getType().getIntOrFloatBitWidth();
}

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//


  bool mlir::LLVM::LLVMSymbolLinkerInterface::canBeLinked(Operation *op) const  {
    return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
  }

  StringRef mlir::LLVM::LLVMSymbolLinkerInterface::getSymbol(Operation *op) const {
    return symbol(op);
  }

  Conflict mlir::LLVM::LLVMSymbolLinkerInterface::findConflict(Operation *src) const {
    assert(canBeLinked(src) && "expected linkable operation");

    if (auto it = summary.find(getSymbol(src)); it != summary.end()) {
      return {it->second, src};
    }

    return Conflict::noConflict(src);
  }

  bool mlir::LLVM::LLVMSymbolLinkerInterface::isLinkNeeded(Conflict pair, bool forDependency) const {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    if (pair.src == pair.dst)
      return false;

    Linkage srcLinkage = getLinkage(pair.src);

    // Always import variables with appending linkage.
    if (isAppendingLinkage(srcLinkage))
      return true;

    bool alreadyDeclared = pair.dst && isDeclaration(pair.dst);

    // Don't import globals that are already declared
    if (shouldLinkOnlyNeeded() && !alreadyDeclared)
      return false;

    // Private dependencies are gonna be renamed and linked
    if (isLocalLinkage(srcLinkage))
      return forDependency;

    // Always import dependencies that are not yet defined or declared
    if (forDependency && !pair.dst)
      return true;

    if (isDeclaration(pair.src))
      return false;

    if (shouldOverrideFromSrc())
      return true;

    if (pair.dst)
      return true;

    // linkage specifies to keep operation only in source
    return !(isLinkOnceLinkage(srcLinkage) ||
             isAvailableExternallyLinkage(srcLinkage));
  }

  llvm::Expected<ConflictResolution> mlir::LLVM::LLVMSymbolLinkerInterface::resolveConflict(Conflict pair) {
    assert(canBeLinked(pair.src) && "expected linkable operation");
    assert(canBeLinked(pair.dst) && "expected linkable operation");

    Linkage srcLinkage = getLinkage(pair.src);
    Linkage dstLinkage = getLinkage(pair.dst);

    Visibility visibility =
        getMinVisibility(getVisibility(pair.src), getVisibility(pair.dst));
    setVisibility(pair.src, visibility);
    setVisibility(pair.dst, visibility);

    const bool srcIsDeclaration = isDeclarationForLinker(pair.src);
    const bool dstIsDeclaration = isDeclarationForLinker(pair.dst);

    if (isAvailableExternallyLinkage(srcLinkage) && dstIsDeclaration) {
      return ConflictResolution::Import;
    }

    // If both `src` and `dst` are declarations, we can ignore the conflict.
    if (srcIsDeclaration && dstIsDeclaration) {
       return ConflictResolution::Ignore;
     }

    // If the `dst` is a declaration import `src` definition
    // Link an available_externally over a declaration.
    if (dstIsDeclaration && !srcIsDeclaration) {
      registerForLink(pair.src);
      return ConflictResolution::Import;
    }

    // Conflicting private values are to be renamed.
    if (isLocalLinkage(dstLinkage)) {
      uniqued.insert(pair.dst);
      registerForLink(pair.src);
      return ConflictResolution::RenameDst;
    }

    if (isLocalLinkage(srcLinkage)) {
      uniqued.insert(pair.src);
      return ConflictResolution::RenameSrc;
    }

    if (isLinkOnceLinkage(srcLinkage)) {
      return ConflictResolution::Ignore;
    }

    if (isLinkOnceLinkage(dstLinkage) || isWeakLinkage(dstLinkage)) {
      return ConflictResolution::Import;
    }

    if (isCommonLinkage(srcLinkage)) {
      if (!isCommonLinkage(dstLinkage))
        return ConflictResolution::Ignore;

      auto srcOp = cast<LLVM::GlobalOp>(pair.src);
      auto dstOp = cast<LLVM::GlobalOp>(pair.dst);
      if (getBitWidth(srcOp) > getBitWidth(dstOp))
        return ConflictResolution::Import;

      return ConflictResolution::Ignore;
    }

    if (isWeakForLinker(srcLinkage)) {
      assert(!isExternalWeakLinkage(dstLinkage));
      assert(!isAvailableExternallyLinkage(dstLinkage));
      if (isLinkOnceLinkage(dstLinkage) && isWeakLinkage(srcLinkage)) {
        return ConflictResolution::Import;
      } else {
        // No need to link the `src`
        return ConflictResolution::Ignore;
      }
    }

    if (isWeakForLinker(dstLinkage)) {
      assert(isExternalLinkage(srcLinkage));
      return ConflictResolution::Import;
    }

    llvm_unreachable("unimplemented conflict resolution");
    return llvm::make_error<llvm::StringError>("unimplemented conflict resolution", llvm::inconvertibleErrorCode());
  }

  LogicalResult mlir::LLVM::LLVMSymbolLinkerInterface::initialize(ModuleOp src) {
    return success();
  }

  SmallVector<Operation *> mlir::LLVM::LLVMSymbolLinkerInterface::dependencies(Operation *op) const {
    // TODO: use something like SymbolTableAnalysis
    Operation *module = op->getParentOfType<ModuleOp>();
    SymbolTable st(module);
    SmallVector<Operation *> result;
    op->walk([&](SymbolUserOpInterface user) {
      if (user.getOperation() == op)
        return;

      if (SymbolRefAttr symbol = user.getUserSymbol()) {
        if (Operation *dep = st.lookup(symbol.getRootReference())) {
          result.push_back(dep);
        }
      }
    });

    return result;
  }


//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
