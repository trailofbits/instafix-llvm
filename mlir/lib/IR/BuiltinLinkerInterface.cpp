//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link builtin dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinLinkerInterface.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;
using namespace mlir::builtin;

//===----------------------------------------------------------------------===//
// BuiltinLinkerInterface
//===----------------------------------------------------------------------===//

  LogicalResult BuiltinLinkerInterface::initialize(ModuleOp src) {
    symbolLinkers = SymbolLinkerInterfaces(src.getContext());
    return symbolLinkers.initialize(src);
  }

  LogicalResult BuiltinLinkerInterface::summarize(ModuleOp src, unsigned flags) {
    WalkResult result = src.walk([&](Operation *op) {
      if (op == src)
        return WalkResult::advance();

      if (summarize(op, flags, /*forDependency=*/false).failed())
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    return failure(result.wasInterrupted());
  }

  LogicalResult BuiltinLinkerInterface::summarize(Operation *op, unsigned flags, bool forDependency) {
    auto* linker = dyn_cast<SymbolLinkerInterface>(op->getDialect());
    if (!linker)
      return success();

    linker->setFlags(flags);

    if (!linker->canBeLinked(op))
      return success();

    Conflict conflict = linker->findConflict(op);
    if (!linker->isLinkNeeded(conflict, forDependency))
      return success();

    if (conflict.hasConflict()) {
      auto maybeResolution = linker->resolveConflict(conflict);
      if (maybeResolution.takeError())
        return failure();

      if (linker->applyResolution(conflict, maybeResolution.get()).failed())
        return failure();
    } else {
      linker->registerForLink(op);
    }

    for (Operation *dep : linker->dependencies(op)) {
      if (summarize(dep, flags, /*forDependency=*/true).failed())
        return failure();
    }

    return success();
  }

  LogicalResult BuiltinLinkerInterface::link(LinkState &state) const {
    return symbolLinkers.link(state);
  }

  OwningOpRef<ModuleOp> BuiltinLinkerInterface::createCompositeModule(ModuleOp src) {
    return ModuleOp::create(
        FileLineColLoc::get(src.getContext(), "composite", 0, 0));
  }

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::builtin::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterface<BuiltinLinkerInterface>();
  });
}
