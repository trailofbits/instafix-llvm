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
#include "mlir/IR/Threading.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// BuiltinLinkerInterface
//===----------------------------------------------------------------------===//

class BuiltinLinkerInterface : public ModuleLinkerInterface {
public:
  using ModuleLinkerInterface::ModuleLinkerInterface;

  LogicalResult initialize(ModuleOp src) override {
    symbolLinkers = SymbolLinkerInterfaces(src.getContext());
    return symbolLinkers.initialize(src);
  }

  LogicalResult finalize(ModuleOp dst) const override {
    return symbolLinkers.finalize(dst);
  }

  LogicalResult summarize(ModuleOp src, unsigned flags,
                          SymbolTableCollection &collection) override {
    if (symbolLinkers.moduleOpSummary(src, symbolTableCollection).failed())
      return failure();
    // Collect all operations to process in parallel
    SmallVector<Operation *> ops;
    src.walk([&](Operation *op) {
      if (op != src)
        ops.push_back(op);
    });

    // Process operations in parallel
    return failableParallelForEach(
        src.getContext(), ops, [&](Operation *op) {
          return summarize(op, flags, /*forDependency=*/false, collection);
        });
  }

  LogicalResult summarize(Operation *op, unsigned flags, bool forDependency,
                          SymbolTableCollection &collection) {
    auto linker = dyn_cast<SymbolLinkerInterface>(op->getDialect());
    if (!linker)
      return success();

    linker->setFlags(flags);

    if (!linker->canBeLinked(op))
      return success();

    Conflict conflict = linker->findConflict(op, collection);
    if (!linker->isLinkNeeded(conflict, forDependency))
      return success();

    if (conflict.hasConflict()) {
      if (linker->resolveConflict(conflict, collection).failed())
        return failure();
    } else {
      linker->registerForLink(op, collection);
    }

    SmallVector<Operation *> deps = linker->dependencies(op, collection);
    auto res = failableParallelForEach(getContext(), deps, [&](Operation *dep) {
      return summarize(dep, flags, /*forDependency=*/true, collection);
    });

    return res;
  }

  LogicalResult link(LinkState &state) override {
    return symbolLinkers.link(state);
  }

  OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) override {
    return ModuleOp::create(
        FileLineColLoc::get(src.getContext(), "composite", 0, 0));
  }

private:
  SymbolLinkerInterfaces symbolLinkers;
  SymbolTableCollection symbolTableCollection;
};

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::builtin::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterface<BuiltinLinkerInterface>();
  });
}
