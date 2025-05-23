#ifndef MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H
#define MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H

#include "mlir/Linker/LLVMLinkerMixin.h"
namespace mlir {
namespace LLVM {

class LLVMSymbolLinkerInterface
    : public link::SymbolAttrLLVMLinkerInterface<LLVMSymbolLinkerInterface> {
public:
  LLVMSymbolLinkerInterface(Dialect *dialect);

  bool canBeLinked(Operation *op) const override;
  static Linkage getLinkage(Operation *op);
  static Visibility getVisibility(Operation *op);
  static void setVisibility(Operation *op, Visibility visibility);
  static bool isDeclaration(Operation *op);
  static unsigned getBitWidth(Operation *op);
  static UnnamedAddr getUnnamedAddr(Operation *op);
  static void setUnnamedAddr(Operation *op, UnnamedAddr val);
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H
