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

  static std::optional<mlir::link::ComdatSelector>
  getComdatSelector(Operation *op);

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

  //===--------------------------------------------------------------------===//
  // Linking overrides for handling function signature mismatches
  //===--------------------------------------------------------------------===//

  /// Finalize the linked module by updating cir.get_global types to match
  /// the linked cir.func types. This handles existing indirect calls where
  /// the get_global was created with a declaration's type that differs from
  /// the linked definition's type.
  LogicalResult finalize(ModuleOp dst) const override;

  /// Track function signature mismatches during conflict resolution.
  /// When two translation units declare the same function with different
  /// signatures (e.g., `extern int bar(int)` vs `char bar(char)`), we record
  /// the mismatch so fixMismatchedCallSites() can convert affected calls.
  LogicalResult verifyLinkageCompatibility(Conflict pair) const override;

  /// Override link to call fixMismatchedCallSites() after standard linking.
  LogicalResult link(link::LinkState &state) override;

private:
  /// Convert direct calls to functions with mismatched signatures into
  /// indirect calls through cir.get_global. This allows the call to proceed
  /// even when the caller's view of the function signature differs from the
  /// actual linked function's signature.
  LogicalResult fixMismatchedCallSites(ModuleOp module) const;

  /// Map from function name to pair of (source type, destination type) for
  /// functions that have signature mismatches between translation units.
  mutable llvm::StringMap<std::pair<mlir::Type, mlir::Type>> mismatchedFunctions;
};

void registerLinkerInterface(mlir::DialectRegistry &registry);
} // namespace cir

#endif // CLANG_INTERFACES_CIR_CIRLINKINTERFACE_H_
