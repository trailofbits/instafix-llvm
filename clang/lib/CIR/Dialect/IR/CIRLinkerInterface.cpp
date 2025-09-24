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
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// CIRSymbolLinkerInterface Implementation
//===----------------------------------------------------------------------===//
Operation *cir::CIRSymbolLinkerInterface::cloneCIROperationPreservingAttributes(
    Operation *src, LinkState &state) const {
  // Get the mapping from the link state
  IRMapping &mapping = state.getMapping();

  // Handle different CIR operation types with specialized cloning
  if (auto globalOp = dyn_cast<cir::GlobalOp>(src)) {
    return cloneCIRGlobal(globalOp, state, mapping);
  }
  if (auto funcOp = dyn_cast<cir::FuncOp>(src)) {
    return cloneCIRFunction(funcOp, state, mapping);
  }

  // For any other operations, fall back to standard cloning with visibility filtering
  Operation *cloned = state.clone(src);

  // Apply visibility filtering to any CIR operation that might have been missed
  if (isa<cir::FuncOp, cir::GlobalOp>(src)) {
    // Set private MLIR visibility for ALL CIR operations to avoid conflicts
    // with MLIR's general verification rules that assume public visibility
    // by default. CIR has its own visibility system via global_visibility.
    cloned->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                    mlir::StringAttr::get(cloned->getContext(), "private"));

    if (auto funcOp = dyn_cast<cir::FuncOp>(cloned)) {
      // Ensure CIR global_visibility is set for declarations
      if (funcOp.isDeclaration() && !cloned->hasAttr("global_visibility")) {
        auto hiddenVisibility = cir::VisibilityAttr::get(cloned->getContext(), cir::VisibilityKind::Hidden);
        cloned->setAttr("global_visibility", hiddenVisibility);
      }
    } else if (auto globalOp = dyn_cast<cir::GlobalOp>(cloned)) {
      // Ensure CIR global_visibility is set for declarations
      if (globalOp.isDeclaration() && !cloned->hasAttr("global_visibility")) {
        auto hiddenVisibility = cir::VisibilityAttr::get(cloned->getContext(), cir::VisibilityKind::Hidden);
        cloned->setAttr("global_visibility", hiddenVisibility);
      }
    }
  }

  return cloned;
}

Operation *cir::CIRSymbolLinkerInterface::cloneCIRGlobal(
    cir::GlobalOp src, LinkState &state, IRMapping &mapping) const {
  // Extract basic attributes
  StringAttr symName = src.getSymNameAttr();
  TypeAttr symTypeAttr = src.getSymTypeAttr();
  Type symType = symTypeAttr.getValue();

  // Create new operation with minimal parameters to avoid parameter order
  // issues
  auto newGlobal =
      state.create<cir::GlobalOp>(src.getLoc(),
                                  symName, // sym_name
                                  symType  // sym_type
                                  // Let other parameters use their defaults
      );

  // Set private MLIR visibility for all CIR globals to avoid verification conflicts
  newGlobal->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                     mlir::StringAttr::get(newGlobal.getContext(), "private"));

  // Copy ALL attributes directly from source to preserve CIR attribute syntax
  // This is the key fix - no serialization means no corruption
  for (auto namedAttr : src->getAttrs()) {
    // Skip MLIR symbol visibility attribute - we've already set it to private above
    if (namedAttr.getName() == mlir::SymbolTable::getVisibilityAttrName())
      continue;

    // Handle CIR global visibility attribute specially
    if (namedAttr.getName() == "global_visibility") {
      // Always preserve global_visibility attribute as it's required by MLIR verifier
      // For declarations, ensure it's set to a safe value (hidden, not public/default)
      if (src.isDeclaration()) {
        // Set to hidden visibility for declarations to avoid verification errors
        auto hiddenVisibility = cir::VisibilityAttr::get(newGlobal.getContext(), cir::VisibilityKind::Hidden);
        newGlobal->setAttr("global_visibility", hiddenVisibility);
      } else {
        // For definitions, copy the original attribute
        newGlobal->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      continue;
    }

    newGlobal->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  // Update mapping for reference handling
  mapping.map(src.getOperation(), newGlobal.getOperation());

  return newGlobal.getOperation();
}

Operation *
cir::CIRSymbolLinkerInterface::cloneCIRFunction(FuncOp src, LinkState &state,
                                                IRMapping &mapping) const {
  // Extract basic attributes
  StringAttr symName = src.getSymNameAttr();
  TypeAttr functionTypeAttr = src.getFunctionTypeAttr();
  auto functionType = cast<cir::FuncType>(functionTypeAttr.getValue());

  // Create new FuncOp with minimal parameters to avoid parameter order issues
  auto newFunc =
      state.create<cir::FuncOp>(src.getLoc(),
                                symName,     // name
                                functionType // type
                                // Let other parameters use their defaults
      );

  // Set private MLIR visibility for all CIR functions to avoid verification conflicts
  newFunc->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                   mlir::StringAttr::get(newFunc.getContext(), "private"));

  // Copy ALL attributes directly from source to preserve CIR attribute syntax
  for (auto namedAttr : src->getAttrs()) {
    // Skip MLIR symbol visibility attribute - we've already set it to private above
    if (namedAttr.getName() == mlir::SymbolTable::getVisibilityAttrName())
      continue;

    // Handle CIR global visibility attribute specially
    if (namedAttr.getName() == "global_visibility") {
      // Always preserve global_visibility attribute as it's required by MLIR verifier
      // For declarations, ensure it's set to a safe value (hidden, not public/default)
      if (src.isDeclaration()) {
        // Set to hidden visibility for declarations to avoid verification errors
        auto hiddenVisibility = cir::VisibilityAttr::get(newFunc.getContext(), cir::VisibilityKind::Hidden);
        newFunc->setAttr("global_visibility", hiddenVisibility);
      } else {
        // For definitions, copy the original attribute
        newFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
      }
      continue;
    }

    newFunc->setAttr(namedAttr.getName(), namedAttr.getValue());
  }

  // Clone function body if it exists (for definitions, not declarations)
  if (!src.isDeclaration()) {
    newFunc.getBody().cloneInto(&src.getBody(), mapping);
  }

  // Update mapping
  mapping.map(src.getOperation(), newFunc.getOperation());

  return newFunc.getOperation();
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void cir::registerLinkerInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    dialect->addInterfaces<cir::CIRSymbolLinkerInterface>();
  });
}
