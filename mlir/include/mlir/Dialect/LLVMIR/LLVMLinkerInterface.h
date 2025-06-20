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
  static bool isComdat(Operation *op);
  static std::optional<link::ComdatSelector> getComdatSelector(Operation *op);
  static bool isDeclaration(Operation *op);
  static unsigned getBitWidth(Operation *op);
  static UnnamedAddr getUnnamedAddr(Operation *op);
  static void setUnnamedAddr(Operation *op, UnnamedAddr val);
  static std::optional<uint64_t> getAlignment(Operation *op);
  static bool isConstant(Operation *op);
  static llvm::StringRef getSection(Operation *op);
  static uint32_t getAddressSpace(Operation *op);
  StringRef getSymbol(Operation *op) const override;
  Operation *materialize(Operation *src, link::LinkState &state) const override;
  SmallVector<Operation *> dependencies(Operation *op) const override;
  Operation *appendGlobals(llvm::StringRef glob, link::LinkState &state);

  template <typename structor_t>
  Operation *appendGlobalStructors(link::LinkState &state) {
    ArrayRef<Operation *> toLink{};

    if constexpr (std::is_same<LLVM::GlobalCtorsOp, structor_t>()) {
      if (auto found = append.find("llvm.global_ctors"); found != append.end())
        toLink = append.find("llvm.global_ctors")->second;
    } else if constexpr (std::is_same<LLVM::GlobalDtorsOp, structor_t>()) {
      if (auto found = append.find("llvm.global_ctors"); found != append.end())
        toLink = append.find("llvm.global_ctors")->second;
    }

    std::vector<Attribute> newStructorList;
    std::vector<Attribute> newPriorities;
    std::vector<Attribute> newData;

    for (auto op : toLink) {
      auto structor = dyn_cast<structor_t>(op);
      if (!structor)
        llvm_unreachable("invali global structor operation");

      ArrayRef<Attribute> structorList;
      if constexpr (std::is_same<LLVM::GlobalCtorsOp, structor_t>()) {
        structorList = structor.getCtors().getValue();
      } else if constexpr (std::is_same<LLVM::GlobalDtorsOp, structor_t>()) {
        structorList = structor.getDtors().getValue();
      }

      ArrayRef<Attribute> priorities = structor.getPriorities().getValue();
      ArrayRef<Attribute> data = structor.getData().getValue();

      for (auto [idx, dataAttr] : llvm::enumerate(data)) {
        // data value is either #llvm.zero or symbol ref
        // if it is zero, we always have to include the value
        // if it is a symbol ref, we have to check if the symbol
        // from the same module is being used
        //
        if (auto globalSymbol = dyn_cast<FlatSymbolRefAttr>(dataAttr)) {
          auto globalOp = summary.lookup(globalSymbol.getValue());
          assert(globalOp && "structor referenced global not in summary?");
          // globals are definde at module level
          if (globalOp->getParentOp() != op->getParentOp())
            continue;
        }

        newData.push_back(dataAttr);
        newStructorList.push_back(structorList[idx]);
        newPriorities.push_back(priorities[idx]);
      }
    }

    auto cloned = state.clone(toLink.back());
    auto newStructorsAttr =
        mlir::ArrayAttr::get(cloned->getContext(), newStructorList);
    auto newPrioritiesAttr =
        mlir::ArrayAttr::get(cloned->getContext(), newPriorities);
    auto newDataAttr = mlir::ArrayAttr::get(cloned->getContext(), newData);

    if constexpr (std::is_same<LLVM::GlobalCtorsOp, structor_t>()) {
      auto ctor = cast<LLVM::GlobalCtorsOp>(cloned);
      ctor.setCtorsAttr(newStructorsAttr);
    } else if constexpr (std::is_same<LLVM::GlobalDtorsOp, structor_t>()) {
      auto dtor = cast<LLVM::GlobalDtorsOp>(cloned);
      dtor.setDtorsAttr(newStructorsAttr);
    }

    auto structor = cast<structor_t>(cloned);
    structor.setPrioritiesAttr(newPrioritiesAttr);
    structor.setDataAttr(newDataAttr);
    return cloned;
  }
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMLINKERINTERFACE_H
