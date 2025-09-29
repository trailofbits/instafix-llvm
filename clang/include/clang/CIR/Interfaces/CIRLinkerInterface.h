//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link CIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
#define CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include <optional>

namespace mlir {
class DialectRegistry;
} // namespace mlir

using namespace mlir;
using namespace mlir::link;

namespace cir {
//===----------------------------------------------------------------------===//
// CIRSymbolLinkerInterface
//===----------------------------------------------------------------------===//

class CIRSymbolLinkerInterface
    : public SymbolAttrLLVMLinkerInterface<CIRSymbolLinkerInterface> {
public:
  CIRSymbolLinkerInterface(Dialect *dialect)
      : SymbolAttrLLVMLinkerInterface(dialect) {}

  bool canBeLinked(Operation *op) const override {
    return isa<cir::GlobalOp>(op) || isa<cir::FuncOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // LLVMLinkerMixin required methods from derived linker interface
  //===--------------------------------------------------------------------===//

  // TODO: expose convertLinkage from LowerToLLVM.cpp
  static Linkage toLLVMLinkage(cir::GlobalLinkageKind linkage) {
    using CIR = cir::GlobalLinkageKind;
    using LLVM = mlir::LLVM::Linkage;

    switch (linkage) {
    case CIR::AvailableExternallyLinkage:
      return LLVM::AvailableExternally;
    case CIR::CommonLinkage:
      return LLVM::Common;
    case CIR::ExternalLinkage:
      return LLVM::External;
    case CIR::ExternalWeakLinkage:
      return LLVM::ExternWeak;
    case CIR::InternalLinkage:
      return LLVM::Internal;
    case CIR::LinkOnceAnyLinkage:
      return LLVM::Linkonce;
    case CIR::LinkOnceODRLinkage:
      return LLVM::LinkonceODR;
    case CIR::PrivateLinkage:
      return LLVM::Private;
    case CIR::WeakAnyLinkage:
      return LLVM::Weak;
    case CIR::WeakODRLinkage:
      return LLVM::WeakODR;
    };
  }

  static Linkage getLinkage(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return toLLVMLinkage(gv.getLinkage());
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return toLLVMLinkage(fn.getLinkage());
    llvm_unreachable("unexpected operation");
  }

  static bool isComdat(Operation *op) {
    // TODO(frabert): Extracting comdat info from CIR is not implemented yet
    return false;
  }

  static std::optional<mlir::link::ComdatSelector>
  getComdatSelector(Operation *op) {
    // TODO(frabert): Extracting comdat info from CIR is not implemented yet
    return std::nullopt;
  }

  // TODO: expose lowerCIRVisibilityToLLVMVisibility from LowerToLLVM.cpp
  static Visibility toLLVMVisibility(cir::VisibilityAttr visibility) {
    return toLLVMVisibility(visibility.getValue());
  }

  static Visibility toLLVMVisibility(cir::VisibilityKind visibility) {
    using CIR = cir::VisibilityKind;
    using LLVM = mlir::LLVM::Visibility;

    switch (visibility) {
    case CIR::Default:
      return LLVM::Default;
    case CIR::Hidden:
      return LLVM::Hidden;
    case CIR::Protected:
      return LLVM::Protected;
    };
  }

  static cir::VisibilityKind toCIRVisibility(Visibility visibility) {
    using CIR = cir::VisibilityKind;
    using LLVM = mlir::LLVM::Visibility;

    switch (visibility) {
    case LLVM::Default:
      return CIR::Default;
    case LLVM::Hidden:
      return CIR::Hidden;
    case LLVM::Protected:
      return CIR::Protected;
    };
  }

  static cir::VisibilityAttr toCIRVisibilityAttr(Visibility visibility,
                                                 MLIRContext *mlirContext) {
    return cir::VisibilityAttr::get(mlirContext, toCIRVisibility(visibility));
  }

  static Visibility getVisibility(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return toLLVMVisibility(gv.getGlobalVisibility());
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return toLLVMVisibility(fn.getGlobalVisibility());
    llvm_unreachable("unexpected operation");
  }

  static void setVisibility(Operation *op, Visibility visibility) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.setGlobalVisibilityAttr(
          toCIRVisibilityAttr(visibility, op->getContext()));
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return fn.setGlobalVisibilityAttr(
          toCIRVisibilityAttr(visibility, op->getContext()));
    llvm_unreachable("unexpected operation");
  }

  static bool isDeclaration(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.isDeclaration();
    if (auto fn = dyn_cast<cir::FuncOp>(op))
      return fn.isDeclaration();
    llvm_unreachable("unexpected operation");
  }

  static unsigned getBitWidth(Operation *op) { llvm_unreachable("NYI"); }

  // FIXME: CIR does not yet have UnnamedAddr attribute
  static UnnamedAddr getUnnamedAddr(Operation * /* op*/) {
    return UnnamedAddr::Global;
  }

  // FIXME: CIR does not yet have UnnamedAddr attribute
  static void setUnnamedAddr(Operation * /* op*/, UnnamedAddr addr) {}

  static std::optional<uint64_t> getAlignment(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.getAlignment();
    // FIXME: CIR does not (yet?) have alignment for functions
    llvm_unreachable("unexpected operation");
  }

  static void setAlignment(Operation *op, std::optional<uint64_t> align) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.setAlignment(align);
    // FIXME: CIR does not (yet?) have alignment for functions
    llvm_unreachable("unexpected operation");
  }

  static bool isConstant(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.getConstant();
    llvm_unreachable("unexpected operation");
  }

  static void setIsConstant(Operation *op, bool value) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op))
      return gv.setConstant(value);
    llvm_unreachable("constness setting allowed only for globals");
  }

  static bool isGlobalVar(Operation *op) { return isa<cir::GlobalOp>(op); }

  static llvm::StringRef getSection(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op)) {
      auto section = gv.getSection();
      return section ? section.value() : llvm::StringRef();
    }
    // FIXME: CIR func does not yet have section attribute
    llvm_unreachable("unexpected operation");
  }

  static std::optional<cir::AddressSpaceAttr> getAddressSpace(Operation *op) {
    if (auto gv = dyn_cast<cir::GlobalOp>(op)) {
      if (auto addrSpace = gv.getAddrSpaceAttr()) {
        return addrSpace;
      }
      return std::nullopt;
    }

    llvm_unreachable("unexpected operation");
  }

  ConflictResolution getConflictResolution(Conflict pair) const override {
    // Check if this is a cross-dialect conflict
    if (isCrossDialectConflict(pair)) {
      // always rename CIR symbols to avoid conflicts
      if (pair.src->getDialect()->getNamespace() == "cir") {
        return ConflictResolution::LinkFromBothAndRenameSrc;
      } else {
        return ConflictResolution::LinkFromBothAndRenameDst;
      }
    }

    // Handle same-dialect CIR string literal conflicts
    // This prevents the "external linkage failure" in LLVMLinkerMixin
    if (isCIRStringLiteralConflict(pair)) {
      // Treat CIR string literals as if they had internal linkage
      // (allows automatic renaming like LLVM string literals)
      return ConflictResolution::LinkFromBothAndRenameSrc;
    }

    return SymbolAttrLLVMLinkerInterface::getConflictResolution(pair);
  }

private:
  bool isCrossDialectConflict(Conflict pair) const {
    return pair.src->getDialect()->getNamespace() !=
           pair.dst->getDialect()->getNamespace();
  }

  bool isCIRStringLiteralConflict(Conflict pair) const {
    // Check if both operations are CIR globals
    auto srcGlobal = dyn_cast<cir::GlobalOp>(pair.src);
    auto dstGlobal = dyn_cast<cir::GlobalOp>(pair.dst);

    if (!srcGlobal || !dstGlobal)
      return false;

    // Check if both are from CIR dialect
    if (pair.src->getDialect()->getNamespace() != "cir" ||
        pair.dst->getDialect()->getNamespace() != "cir") {
      return false;
    }

    // Check if symbol names look like CIR string literals
    StringRef srcName = getSymbol(pair.src);
    StringRef dstName = getSymbol(pair.dst);

    // Both should be CIR string literals (cir.str, cir.str.1, cir.str.21, etc.)
    return srcName.starts_with("cir.str") && dstName.starts_with("cir.str");
  }
};

void registerLinkerInterface(mlir::DialectRegistry &registry);
} // namespace cir

#endif // CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
