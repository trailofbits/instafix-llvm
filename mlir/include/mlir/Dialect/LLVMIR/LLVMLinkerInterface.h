#ifndef MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H
#define MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H

#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"
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
  static std::optional<uint64_t> getAlignment(Operation *op);
  static bool isConstant(Operation *op);
  static llvm::StringRef getSection(Operation *op);
  static uint32_t getAddressSpace(Operation *op);
  Operation *materialize(Operation *src, link::LinkState &state) const override;
  Operation *appendGlobals(llvm::StringRef glob, link::LinkState &state);
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H
