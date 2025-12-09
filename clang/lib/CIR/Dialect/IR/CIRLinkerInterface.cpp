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

#include "clang/CIR/Interfaces/CIRLinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::link;
using namespace cir;

namespace {
// TODO: expose convertLinkage from LowerToLLVM.cpp
Linkage toLLVMLinkage(GlobalLinkageKind linkage) {
  using CIR = GlobalLinkageKind;
  using LLVM = LLVM::Linkage;

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

// TODO: expose lowerCIRVisibilityToLLVMVisibility from LowerToLLVM.cpp
Visibility toLLVMVisibility(VisibilityKind visibility) {
  using CIR = VisibilityKind;
  using LLVM = LLVM::Visibility;

  switch (visibility) {
  case CIR::Default:
    return LLVM::Default;
  case CIR::Hidden:
    return LLVM::Hidden;
  case CIR::Protected:
    return LLVM::Protected;
  };
}

VisibilityKind toCIRVisibility(Visibility visibility) {
  using CIR = VisibilityKind;
  using LLVM = LLVM::Visibility;

  switch (visibility) {
  case LLVM::Default:
    return CIR::Default;
  case LLVM::Hidden:
    return CIR::Hidden;
  case LLVM::Protected:
    return CIR::Protected;
  };
}

VisibilityAttr toCIRVisibilityAttr(Visibility visibility,
                                                MLIRContext *mlirContext) {
  return VisibilityAttr::get(mlirContext, toCIRVisibility(visibility));
}
} // namespace

bool CIRSymbolLinkerInterface::canBeLinked(Operation *op) const {
  return isa<GlobalOp>(op) || isa<FuncOp>(op);
}

Linkage CIRSymbolLinkerInterface::getLinkage(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return toLLVMLinkage(gv.getLinkage());
  if (auto fn = dyn_cast<FuncOp>(op))
    return toLLVMLinkage(fn.getLinkage());
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isComdat(Operation *op) {
  // TODO(frabert): Extracting comdat info from CIR is not implemented yet
  return false;
}

bool CIRSymbolLinkerInterface::hasComdat(Operation *op) {
  // TODO: Extracting comdat info from CIR is not implemented yet
  return false;
}

const link::Comdat *
CIRSymbolLinkerInterface::getComdatResolution(Operation *op) {
  return nullptr;
}

bool CIRSymbolLinkerInterface::selectedByComdat(Operation *op) {
  // TODO: Extracting comdat info from CIR is not implemented yet
  llvm_unreachable("comdat resolution not implemented for CIR");
}

void CIRSymbolLinkerInterface::updateNoDeduplicate(Operation *op) {}

Visibility CIRSymbolLinkerInterface::getVisibility(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return toLLVMVisibility(gv.getGlobalVisibility());
  if (auto fn = dyn_cast<FuncOp>(op))
    return toLLVMVisibility(fn.getGlobalVisibility());
  llvm_unreachable("unexpected operation");
}

void
CIRSymbolLinkerInterface::setVisibility(Operation *op, Visibility visibility) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setGlobalVisibilityAttr(
        toCIRVisibilityAttr(visibility, op->getContext()));
  if (auto fn = dyn_cast<FuncOp>(op))
    return fn.setGlobalVisibilityAttr(
        toCIRVisibilityAttr(visibility, op->getContext()));
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.isDeclaration();
  if (auto fn = dyn_cast<FuncOp>(op))
    return fn.isDeclaration();
  llvm_unreachable("unexpected operation");
}

unsigned CIRSymbolLinkerInterface::getBitWidth(Operation *op) {
  llvm_unreachable("NYI");
}

// FIXME: CIR does not yet have UnnamedAddr attribute
UnnamedAddr CIRSymbolLinkerInterface::getUnnamedAddr(Operation * /* op*/) {
  return UnnamedAddr::Global;
}

// FIXME: CIR does not yet have UnnamedAddr attribute
void
CIRSymbolLinkerInterface::setUnnamedAddr(Operation * /* op*/, UnnamedAddr addr) {}

std::optional<uint64_t> CIRSymbolLinkerInterface::getAlignment(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.getAlignment();
  // FIXME: CIR does not (yet?) have alignment for functions
  llvm_unreachable("unexpected operation");
}

void
CIRSymbolLinkerInterface::setAlignment(Operation *op, std::optional<uint64_t> align) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setAlignment(align);
  // FIXME: CIR does not (yet?) have alignment for functions
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isConstant(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.getConstant();
  llvm_unreachable("unexpected operation");
}

void CIRSymbolLinkerInterface::setIsConstant(Operation *op, bool value) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setConstant(value);
  llvm_unreachable("constness setting allowed only for globals");
}

bool CIRSymbolLinkerInterface::isGlobalVar(Operation *op) {
  return isa<GlobalOp>(op);
}

llvm::StringRef CIRSymbolLinkerInterface::getSection(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op)) {
    auto section = gv.getSection();
    return section ? section.value() : llvm::StringRef();
  }
  // FIXME: CIR func does not yet have section attribute
  llvm_unreachable("unexpected operation");
}

std::optional<AddressSpace>
CIRSymbolLinkerInterface::getAddressSpace(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op)) {
    if (auto addrSpace = gv.getAddrSpaceAttr()) {
      return addrSpace.getValue();
    }
    return std::nullopt;
  }

  llvm_unreachable("unexpected operation");
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void cir::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CIRDialect *dialect) {
    dialect->addInterfaces<CIRSymbolLinkerInterface>();
  });
}
