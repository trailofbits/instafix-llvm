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

#include "mlir/Dialect/LLVMIR/LLVMLinkerInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Linker/LLVMLinkerMixin.h"
#include "mlir/Linker/LinkerInterface.h"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// LLVMSymbolLinkerInterface
//===----------------------------------------------------------------------===//

LLVM::LLVMSymbolLinkerInterface::LLVMSymbolLinkerInterface(Dialect *dialect)
    : SymbolAttrLLVMLinkerInterface(dialect) {}

bool LLVM::LLVMSymbolLinkerInterface::canBeLinked(Operation *op) const {
  return isa<LLVM::GlobalOp>(op) || isa<LLVM::LLVMFuncOp>(op);
}

//===--------------------------------------------------------------------===//
// LLVMLinkerMixin required methods from derived linker interface
//===--------------------------------------------------------------------===//

Linkage LLVM::LLVMSymbolLinkerInterface::getLinkage(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getLinkage();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getLinkage();
  llvm_unreachable("unexpected operation");
}

Visibility LLVM::LLVMSymbolLinkerInterface::getVisibility(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getVisibility_();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getVisibility_();
  llvm_unreachable("unexpected operation");
}

void LLVM::LLVMSymbolLinkerInterface::setVisibility(Operation *op,
                                                    Visibility visibility) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setVisibility_(visibility);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setVisibility_(visibility);
  llvm_unreachable("unexpected operation");
}

// Return true if the primary definition of this global value is outside of
// the current translation unit.
bool LLVM::LLVMSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  llvm_unreachable("unexpected operation");
}

unsigned LLVM::LLVMSymbolLinkerInterface::getBitWidth(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    auto type = gv.getType();
    auto dataLayout = DataLayout::closest(op);
    return dataLayout.getTypeSizeInBits(type);
  }
  llvm_unreachable("unexpected operation");
}

UnnamedAddr LLVM::LLVMSymbolLinkerInterface::getUnnamedAddr(Operation *op) {
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

void LLVM::LLVMSymbolLinkerInterface::setUnnamedAddr(Operation *op,
                                                     UnnamedAddr val) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setUnnamedAddr(val);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setUnnamedAddr(val);
  llvm_unreachable("unexpected operation");
}

std::optional<uint64_t>
LLVM::LLVMSymbolLinkerInterface::getAlignment(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getAlignment();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getAlignment();
  llvm_unreachable("unexpected operation");
}

bool LLVM::LLVMSymbolLinkerInterface::isConstant(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getConstant();
  llvm_unreachable("unexpected operation");
}

llvm::StringRef LLVM::LLVMSymbolLinkerInterface::getSection(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    auto section = gv.getSection();
    return section ? section.value() : llvm::StringRef();
  }
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op)) {
    auto section = fn.getSection();
    return section ? section.value() : llvm::StringRef();
  }
  llvm_unreachable("unexpected operation");
}

uint32_t LLVM::LLVMSymbolLinkerInterface::getAddressSpace(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    return gv.getAddrSpace();
  }
  llvm_unreachable("unexpected operation");
}

Operation *
LLVM::LLVMSymbolLinkerInterface::materialize(Operation *src,
                                             LinkState &state) const {
  auto derived = LinkerMixin::getDerived();
  if (isAppendingLinkage(derived.getLinkage(src))) {
    auto dst = LinkerMixin::append.lookup(src);
    return derived.appendGlobals(dst, src, state);
  }
  return SymbolAttrLinkerInterface::materialize(src, state);
}

Operation *LLVM::LLVMSymbolLinkerInterface::appendGlobals(Operation *dst,
                                                          Operation *src,
                                                          LinkState &state) {
  auto dstGV = dyn_cast<LLVM::GlobalOp>(dst);
  auto srcGV = dyn_cast<LLVM::GlobalOp>(src);
  if (!srcGV || !dstGV)
    llvm_unreachable("unexpected operation");

  if (isDeclaration(dst))
    return state.clone(src);

  auto srcAttrs = srcGV->getAttrs();
  std::vector<NamedAttribute> attrs;
  attrs.reserve(srcAttrs.size());

  auto valueAttrName = srcGV.getValueAttrName();
  auto typeAttrName = srcGV.getGlobalTypeAttrName();
  for (auto attr : srcGV->getAttrs()) {
    auto attrName = attr.getName();
    if (attrName != typeAttrName && attrName != valueAttrName)
      attrs.push_back(attr);
  }

  if (auto dstVal =
          mlir::dyn_cast_if_present<ArrayAttr>(dstGV.getValueOrNull())) {
    auto srcVal = mlir::dyn_cast_if_present<ArrayAttr>(dstGV.getValueOrNull());
    if (!srcVal)
      llvm_unreachable("mismatech value attry type of appending variables");

    auto newVal = dstVal.getValue().vec();
    newVal.reserve(dstVal.size() + srcVal.size());
    newVal.insert(newVal.end(), srcVal.begin(), srcVal.end());
    auto newValAttr = ArrayAttr::get(dstVal.getContext(), newVal);

    auto dstType = mlir::cast<LLVM::LLVMArrayType>(dstGV.getType());
    auto srcType = mlir::cast<LLVM::LLVMArrayType>(srcGV.getType());

    attrs.emplace_back(typeAttrName, TypeAttr::get(LLVM::LLVMArrayType::get(
                                         dstType.getElementType(),
                                         dstType.getNumElements() +
                                             srcType.getNumElements())));
    attrs.emplace_back(valueAttrName, newValAttr);
    return state.remap<LLVM::GlobalOp>(src, TypeRange(), ValueRange(), attrs);
  }

  if (auto dstVal = mlir::dyn_cast<DenseElementsAttr>(dstGV.getValueOrNull())) {
    auto srcVal = mlir::dyn_cast<DenseElementsAttr>(srcGV.getValueOrNull());
    if (!srcVal)
      llvm_unreachable("mismatched value attr type of appending variables");

    auto srcData = srcVal.getRawData();
    auto dstData = dstVal.getRawData();
    std::vector<char> newData(dstData.begin(), dstData.end());
    newData.insert(newData.end(), srcData.begin(), srcData.end());

    auto numElems =
        srcVal.getType().getNumElements() + dstVal.getType().getNumElements();
    auto elemType = srcVal.getElementType();
    auto newTensorType = RankedTensorType::get(numElems, elemType);

    attrs.emplace_back(valueAttrName, DenseElementsAttr::getFromRawBuffer(
                                          newTensorType, newData));
    attrs.emplace_back(typeAttrName, TypeAttr::get(LLVM::LLVMArrayType::get(
                                         elemType, numElems)));
    return state.remap<LLVM::GlobalOp>(src, TypeRange(), ValueRange(), attrs);
  }
  llvm_unreachable("unknown value attribute type");
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
