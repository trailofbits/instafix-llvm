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
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMLinkerInterface.h"
using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//




mlir::LLVM::LLVMSymbolLinkerInterface::LLVMSymbolLinkerInterface(Dialect *dialect)
    : SymbolAttrLLVMLinkerInterface(dialect) {}

bool  mlir::LLVM::LLVMSymbolLinkerInterface::canBeLinked(Operation *op) const {
  return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
}

  //===--------------------------------------------------------------------===//
  // LLVMLinkerMixin required methods from derived linker interface
  //===--------------------------------------------------------------------===//

Linkage  mlir::LLVM::LLVMSymbolLinkerInterface::getLinkage(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getLinkage();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getLinkage();
  llvm_unreachable("unexpected operation");
}

Visibility  mlir::LLVM::LLVMSymbolLinkerInterface::getVisibility(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getVisibility_();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getVisibility_();
  llvm_unreachable("unexpected operation");
}

void  mlir::LLVM::LLVMSymbolLinkerInterface::setVisibility(Operation *op, Visibility visibility) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setVisibility_(visibility);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setVisibility_(visibility);
  llvm_unreachable("unexpected operation");
}

// Return true if the primary definition of this global value is outside of
// the current translation unit.
bool  mlir::LLVM::LLVMSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  llvm_unreachable("unexpected operation");
}

unsigned  mlir::LLVM::LLVMSymbolLinkerInterface::getBitWidth(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getType().getIntOrFloatBitWidth();
  llvm_unreachable("unexpected operation");
}

UnnamedAddr mlir::LLVM::LLVMSymbolLinkerInterface::getUnnamedAddr(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    auto addr = gv.getUnnamedAddr();
    return addr ? *addr : UnnamedAddr::Global;
  }
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    auto addr = fn.getUnnamedAddr();
    return addr ? *addr : UnnamedAddr::Global;
  }
  llvm_unreachable("unexpected operation");
}

void  mlir::LLVM::LLVMSymbolLinkerInterface::setUnnamedAddr(Operation *op, UnnamedAddr val) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setUnnamedAddr(val);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setUnnamedAddr(val);
  llvm_unreachable("unexpected operation");
}


//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
