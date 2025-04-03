//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines registration of builtin dialect linker interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINLINKERINTERFACE_H
#define MLIR_IR_BUILTINLINKERINTERFACE_H

#include "mlir/Linker/LinkerInterface.h"

namespace mlir {
class DialectRegistry;

namespace builtin {
using namespace mlir::link;
class BuiltinLinkerInterface : public ModuleLinkerInterface {
public:
  using ModuleLinkerInterface::ModuleLinkerInterface;

    LogicalResult initialize(ModuleOp src) override;

    LogicalResult summarize(ModuleOp src, unsigned flags) override;

    LogicalResult summarize(Operation *op, unsigned flags, bool forDependency);

    LogicalResult link(LinkState &state) const override;

    OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) override;
  

private:
  SymbolLinkerInterfaces symbolLinkers;
};


void registerLinkerInterface(DialectRegistry &registry);
} // namespace builtin

} // namespace mlir
#endif // MLIR_IR_BUILTINLINKERINTERFACE_H
