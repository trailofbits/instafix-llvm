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
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "mlir/Linker/LLVMLinkerMixin.h"

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

  bool canBeLinked(Operation *op) const override;

  //===--------------------------------------------------------------------===//
  // LLVMLinkerMixin required methods from derived linker interface
  //===--------------------------------------------------------------------===//

  static Linkage getLinkage(Operation *op);

  static bool isComdat(Operation *op);

  static bool hasComdat(Operation *op);

  static const link::Comdat *getComdatResolution(Operation *op);

  static bool selectedByComdat(Operation *op);

  static void updateNoDeduplicate(Operation *op);

  static Visibility getVisibility(Operation *op);

  static void setVisibility(Operation *op, Visibility visibility);

  static bool isDeclaration(Operation *op);

  static unsigned getBitWidth(Operation *op);

  static UnnamedAddr getUnnamedAddr(Operation *op);

  static void setUnnamedAddr(Operation *op, UnnamedAddr addr);

  static std::optional<uint64_t> getAlignment(Operation *op);

  static void setAlignment(Operation *op, std::optional<uint64_t> align);

  static bool isConstant(Operation *op);

  static void setIsConstant(Operation *op, bool value);

  static bool isGlobalVar(Operation *op);

  static llvm::StringRef getSection(Operation *op);

  static std::optional<cir::AddressSpace> getAddressSpace(Operation *op);
};

void registerLinkerInterface(mlir::DialectRegistry &registry);
} // namespace cir

#endif // CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
