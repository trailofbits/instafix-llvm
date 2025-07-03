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
  return isa<LLVM::GlobalOp, LLVM::LLVMFuncOp, LLVM::GlobalCtorsOp,
             LLVM::GlobalDtorsOp, LLVM::ComdatOp, LLVM::AliasOp>(op);
}

//===--------------------------------------------------------------------===//
// LLVMLinkerMixin required methods from derived linker interface
//===--------------------------------------------------------------------===//

Linkage LLVM::LLVMSymbolLinkerInterface::getLinkage(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getLinkage();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getLinkage();
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.getLinkage();
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
    return Linkage::Appending;
  llvm_unreachable("unexpected operation");
}

Visibility LLVM::LLVMSymbolLinkerInterface::getVisibility(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getVisibility_();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getVisibility_();
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.getVisibility_();

  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
    return Visibility::Default;
  llvm_unreachable("unexpected operation");
}

void LLVM::LLVMSymbolLinkerInterface::setVisibility(Operation *op,
                                                    Visibility visibility) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setVisibility_(visibility);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setVisibility_(visibility);
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.setVisibility_(visibility);
  // GlobalCotrs and Dtors are defined as operations in mlir
  // but as globals in LLVM IR
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
    return;
  llvm_unreachable("unexpected operation");
}

static bool hasComdat(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getComdat().has_value();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getComdat().has_value();
  llvm_unreachable("unexpected operation");
}

static SymbolRefAttr getComdatSymbol(Operation *op) {
  assert(hasComdat(op) && "Operation with Comdat expected");
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
      return gv.getComdat().value();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
      return fn.getComdat().value();
  llvm_unreachable("unexpected operation");
}

bool LLVM::LLVMSymbolLinkerInterface::isComdat(Operation *op) {
  return isa<LLVM::ComdatOp>(op);
}

std::optional<mlir::link::ComdatSelector> LLVM::LLVMSymbolLinkerInterface::getComdatSelector(Operation *op) {
  if (!hasComdat(op))
    return std::nullopt;

  auto symbol = getComdatSymbol(op);
  auto *symTabOp = SymbolTable::getNearestSymbolTable(op);
  auto comdatSelector = cast<mlir::LLVM::ComdatSelectorOp>(SymbolTable::lookupSymbolIn(symTabOp, symbol));
  return {{comdatSelector.getSymName(), comdatSelector.getComdat()}};
}

// Return true if the primary definition of this global value is outside of
// the current translation unit.
bool LLVM::LLVMSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp, LLVM::AliasOp>(op))
    return false;
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
  if (auto alias = dyn_cast<LLVM::AliasOp>(op)) {
    auto addr = alias.getUnnamedAddr();
    return addr ? *addr : UnnamedAddr::Global;
  }
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
    return UnnamedAddr::Global;
  llvm_unreachable("unexpected operation");
}

void LLVM::LLVMSymbolLinkerInterface::setUnnamedAddr(Operation *op,
                                                     UnnamedAddr val) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setUnnamedAddr(val);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setUnnamedAddr(val);
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.setUnnamedAddr(val);
  // GlobalCotrs and Dtors are defined as operations in mlir
  // but as globals in LLVM IR
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
    return;
  llvm_unreachable("unexpected operation");
}

std::optional<uint64_t>
LLVM::LLVMSymbolLinkerInterface::getAlignment(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getAlignment();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getAlignment();
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::AliasOp>(op))
    return {};
  llvm_unreachable("unexpected operation");
}

bool LLVM::LLVMSymbolLinkerInterface::isConstant(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getConstant();
  if (isa<LLVM::AliasOp>(op))
    return true;
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp>(op))
    return false;
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
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::AliasOp>(op))
    return llvm::StringRef();
  llvm_unreachable("unexpected operation");
}

uint32_t LLVM::LLVMSymbolLinkerInterface::getAddressSpace(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    return gv.getAddrSpace();
  }
  if (auto alias = dyn_cast<LLVM::AliasOp>(op)) {
    return alias.getAddrSpace();
  }
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp>(op))
    return 0;
  llvm_unreachable("unexpected operation");
}

StringRef LLVM::LLVMSymbolLinkerInterface::getSymbol(Operation *op) const {
  if (isa<LLVM::GlobalCtorsOp>(op))
    return "llvm.global_ctors";
  if (isa<LLVM::GlobalDtorsOp>(op))
    return "llvm.global_dtors";
  return SymbolAttrLinkerInterface::getSymbol(op);
}

Operation *
LLVM::LLVMSymbolLinkerInterface::materialize(Operation *src,
                                             LinkState &state) const {
  auto derived = LinkerMixin::getDerived();
  // empty append means that we either have single module or that something went
  // wrong
  if (isAppendingLinkage(derived.getLinkage(src)) && !append.empty()) {
    return derived.appendGlobals(derived.getSymbol(src), state);
  }
  return SymbolAttrLinkerInterface::materialize(src, state);
}

SmallVector<Operation *>
LLVM::LLVMSymbolLinkerInterface::dependencies(Operation *op) const {
  Operation *module = op->getParentOfType<ModuleOp>();
  SymbolTable st(module);
  SmallVector<Operation *> result;

  auto insertDepIfExists = [&](auto symbolRef) -> void {
    if (Operation *dep = st.lookup(symbolRef.getRootReference()))
      result.push_back(dep);
  };

  // Structor ops implement the SymbolUserOpInteface but can not provide
  // the `getUserSymbol` method correctly as they reference mutliple symbols and
  // the method allows to return only one. They also do not have any body to
  // walk and reference the symbols in an attribute. We have to intercept on
  // these operations.
  ArrayAttr structors = {};
  if (auto ctor = dyn_cast<GlobalCtorsOp>(op)) {
    structors = ctor.getCtors();
  }
  if (auto dtor = dyn_cast<GlobalDtorsOp>(op)) {
    structors = dtor.getDtors();
  }

  if (structors) {
    for (auto structor : structors)
      insertDepIfExists(cast<FlatSymbolRefAttr>(structor));
    return result;
  }

  // Functions are only defined in module, avoid unnecessary cast in every
  // analyzed op
  if (auto fn = dyn_cast<LLVMFuncOp>(op)) {
    if (FlatSymbolRefAttr personality = fn.getPersonalityAttr())
      insertDepIfExists(personality);
  }

  op->walk([&](Operation *operation) {
    if (operation == op)
      return;
    if (auto user = dyn_cast<SymbolUserOpInterface>(operation)) {
      if (SymbolRefAttr symbol = user.getUserSymbol())
        insertDepIfExists(symbol);
      return;
    }
    if (auto invoke = dyn_cast<InvokeOp>(operation)) {
      if (FlatSymbolRefAttr symbol = invoke.getCalleeAttr())
        insertDepIfExists(symbol);
    }
  });

  return result;
}

static std::pair<Attribute, Type>
getAppendedArrayAttr(llvm::ArrayRef<mlir::Operation *> globs,
                     LinkState &state) {
  std::vector<Attribute> newValue;
  // conservative estimate
  newValue.reserve(globs.size());

  for (auto op : globs) {
    auto glob = dyn_cast<LLVM::GlobalOp>(op);
    if (!glob)
      llvm_unreachable("appending non-global variable");
    // TODO: check that all element types match?

    if (auto globValue =
            dyn_cast_if_present<ArrayAttr>(glob.getValueOrNull())) {
      newValue.insert(newValue.end(), globValue.begin(), globValue.end());
    } else {
      if (glob.getInitializer().empty()) {
        // global is initialized and does not have an init region
        // -> value attribute is not an ArrayAttr
        llvm_unreachable("mismatched global variable value attributes");
      }
      llvm_unreachable(
          "linking mixed definition via an attribute and init region NYI");
    }
  }
  auto firstGV = mlir::cast<LLVM::GlobalOp>(globs.front());
  auto elemType =
      mlir::cast<LLVM::LLVMArrayType>(firstGV.getType()).getElementType();
  return {ArrayAttr::get(firstGV.getContext(), newValue),
          LLVM::LLVMArrayType::get(elemType, newValue.size())};
}

static std::pair<Attribute, Type>
getAppendedDenseAttr(llvm::ArrayRef<mlir::Operation *> globs,
                     LinkState &state) {
  std::vector<char> newValue;
  size_t numElems = 0;

  for (auto op : globs) {
    auto glob = dyn_cast<LLVM::GlobalOp>(op);
    if (!glob)
      llvm_unreachable("appending non-global variable");
    // TODO: check that all element types match?

    if (auto globValue =
            dyn_cast_if_present<DenseElementsAttr>(glob.getValueOrNull())) {
      auto data = globValue.getRawData();
      newValue.insert(newValue.end(), data.begin(), data.end());
      numElems += globValue.getNumElements();
    } else {
      if (glob.getInitializer().empty()) {
        // global is initialized and does not have an init region
        // -> value attribute is not a DenseAttr
        llvm_unreachable("mismatched global variable value attributes");
      }
      llvm_unreachable(
          "linking mixed definition via an attribute and init region NYI");
    }
  }
  auto firstGV = mlir::cast<LLVM::GlobalOp>(globs.front());
  auto elemType =
      mlir::cast<LLVM::LLVMArrayType>(firstGV.getType()).getElementType();
  auto newTensorType = RankedTensorType::get(numElems, elemType);
  return {DenseElementsAttr::getFromRawBuffer(newTensorType, newValue),
          LLVM::LLVMArrayType::get(elemType, numElems)};
}

static std::pair<Attribute, Type>
getAppendedAttr(llvm::ArrayRef<mlir::Operation *> globs, LinkState &state) {
  if (auto glob = dyn_cast<LLVM::GlobalOp>(globs.front())) {
    if (isa_and_present<ArrayAttr>(glob.getValueOrNull()))
      return getAppendedArrayAttr(globs, state);

    if (isa_and_present<DenseElementsAttr>(glob.getValueOrNull()))
      return getAppendedDenseAttr(globs, state);

    llvm_unreachable("unknown init attr kind");
  }
  llvm_unreachable("appending operation that isn't a global");
}

static Operation *
getAppendedOpWithInitRegion(llvm::ArrayRef<mlir::Operation *> globs,
                            LinkState &state) {
  auto endGV = dyn_cast<LLVM::GlobalOp>(globs.back());
  if (!endGV)
    llvm_unreachable("unexpected operation");

  auto targetGV = cast<LLVM::GlobalOp>(state.cloneWithoutRegions(globs.back()));
  auto &targetRegion = targetGV.getInitializer();
  // Without a block the builder does not know where to set the insertion point
  targetRegion.emplaceBlock();

  auto originalType = dyn_cast<LLVM::LLVMArrayType>(endGV.getType());
  if (!originalType)
    llvm_unreachable("unexpected global type");
  auto elemType = originalType.getElementType();
  size_t elemCount = 0;

  IRMapping &mapping = state.getMapping();
  auto builder = OpBuilder(targetRegion);
  std::vector<Value> values;
  std::vector<std::vector<int64_t>> positions;

  for (auto globalOp : globs) {
    auto glob = dyn_cast<LLVM::GlobalOp>(globalOp);
    auto globType = dyn_cast<LLVM::LLVMArrayType>(glob.getType());
    if (!globType || globType.getElementType() != elemType)
      llvm_unreachable("appending globals with mismatched types");
    for (auto &op : glob.getInitializer().getOps()) {
      // skip ops that will be crated manually later
      if (isa<LLVM::UndefOp, LLVM::ReturnOp, LLVM::InsertValueOp>(op))
        continue;

      Operation *cloned = builder.clone(op, mapping);
      mapping.map(&op, cloned);
      // LLVM dialect does not have multiple result operations
      // zero result operation should not appear in this context
      // unless its a return which we skip
      Value clonedRes = cloned->getResult(0);
      for (Operation *user : op.getUsers()) {
        if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(user)) {
          llvm::ArrayRef<int64_t> currentPositions = insertValue.getPosition();
          std::vector<int64_t> offsetPositions;
          offsetPositions.reserve(positions.size());

          for (auto pos : currentPositions) {
            offsetPositions.push_back(pos + elemCount);
          }
          positions.push_back(std::move(offsetPositions));
        }
      }
      values.push_back(clonedRes);
    }
    elemCount += globType.getNumElements();
  }
  Type resType = LLVM::LLVMArrayType::get(elemType, elemCount);
  targetGV->setAttr(targetGV.getGlobalTypeAttrName(), TypeAttr::get(resType));

  Value currentValue =
      builder.create<LLVM::UndefOp>(targetGV.getLoc(), resType);
  for (auto [idx, value] : llvm::enumerate(values)) {
    currentValue = builder.create<LLVM::InsertValueOp>(
        targetGV.getLoc(), currentValue, value, positions[idx]);
  }

  builder.create<LLVM::ReturnOp>(targetGV.getLoc(), currentValue);

  return targetGV;
}

static Operation *appendGlobalOps(ArrayRef<Operation *> globs, LLVM::GlobalOp lastGV, LinkState &state) {
  // Src ops that are declarations are ignored in favour of dst operation
  // This mimics the behaviour of linkAppendingVarProto in llvm-link
  if (globs.size() == 1)
      return state.clone(globs.front());

  if (!lastGV.getInitializer().empty()) {
      return getAppendedOpWithInitRegion(globs, state);
  } else {
      auto [value, type] = getAppendedAttr(globs, state);

      auto valueAttrName = lastGV.getValueAttrName();
      auto typeAttrName = lastGV.getGlobalTypeAttrName();

      auto cloned = state.clone(globs.back());
      cloned->setAttr(valueAttrName, value);
      cloned->setAttr(typeAttrName, TypeAttr::get(type));
      return cloned;
  }
  llvm_unreachable("unknown value attribute type");
}

static Operation *appendComdatOps(ArrayRef<Operation *> globs, LLVM::ComdatOp comdat, LinkState &state) {
  auto result = cast<LLVM::ComdatOp>(state.clone(comdat));
  llvm::StringMap<Operation *> selectors;

  for (auto selector : result.getOps<LLVM::ComdatSelectorOp>()) {
      selectors[selector.getSymName()] = selector;
  }

  for (auto *glob : globs) {
    comdat = dyn_cast<LLVM::ComdatOp>(glob);
      for (auto &op : comdat.getBody().getOps()) {
        auto selector = cast<LLVM::ComdatSelectorOp>(op);
        auto selectorName = selector.getSymName();
        if (selectors.contains(selectorName)) {
            continue;
        }
        auto *cloned = state.clone(selector);
        cloned->moveBefore(&result.getBody().front().back());
        selectors[selectorName] = cloned;
      }
  }
  return result;
}

Operation *LLVM::LLVMSymbolLinkerInterface::appendGlobals(llvm::StringRef glob,
                                                          LinkState &state) {
  if (glob == "llvm.global_ctors")
    return appendGlobalStructors<LLVM::GlobalCtorsOp>(state);
  if (glob == "llvm.global_dtors")
    return appendGlobalStructors<LLVM::GlobalDtorsOp>(state);

  const auto &globs = append.lookup(glob);
  if (auto lastGV = dyn_cast<LLVM::GlobalOp>(globs.back()))
    return appendGlobalOps(globs, lastGV, state);
  if (auto comdat = dyn_cast<LLVM::ComdatOp>(globs.back()))
    return appendComdatOps(globs, comdat, state);
  llvm_unreachable("unexpected operation");
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
