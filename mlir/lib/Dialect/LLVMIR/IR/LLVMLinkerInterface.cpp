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
#include "mlir/Dialect/DLTI/DLTI.h"
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
  // Only link operations that are direct children of a top-level ModuleOp.
  // This filters out:
  // 1. Operations nested inside regions (e.g., inside an AliasOp's initializer)
  // 2. Operations inside nested named modules
  Operation *parent = op->getParentOp();
  if (!isa_and_nonnull<ModuleOp>(parent))
    return false;

  // Check that the parent module is a top-level module (has no parent module)
  if (parent->getParentOfType<ModuleOp>())
    return false;

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

bool LLVM::LLVMSymbolLinkerInterface::hasComdat(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getComdat().has_value();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getComdat().has_value();
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.getComdat().has_value();
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp>(op))
    return false;
  llvm_unreachable("unexpected operation");
}

SymbolRefAttr LLVM::LLVMSymbolLinkerInterface::getComdatSymbol(Operation *op) {
  assert(hasComdat(op) && "Operation with Comdat expected");
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getComdat().value();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getComdat().value();
  if (auto alias = dyn_cast<LLVM::AliasOp>(op))
    return alias.getComdat().value();
  llvm_unreachable("unexpected operation");
}

bool LLVM::LLVMSymbolLinkerInterface::isComdat(Operation *op) {
  return isa<LLVM::ComdatOp>(op);
}

LLVM::comdat::Comdat
LLVM::LLVMSymbolLinkerInterface::getComdatSelectionKind(Operation *op) {
  if (auto selector = dyn_cast<LLVM::ComdatSelectorOp>(op))
    return selector.getComdat();
  llvm_unreachable("expected selector op");
}

// Return true if the primary definition of this global value is outside of
// the current translation unit.
bool LLVM::LLVMSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getInitializerRegion().empty() && !gv.getValue();
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.getBody().empty();
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp,
          LLVM::AliasOp>(op))
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
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::AliasOp,
          LLVM::ComdatOp>(op))
    return std::nullopt;
  llvm_unreachable("unexpected operation");
}

void LLVM::LLVMSymbolLinkerInterface::setAlignment(
    Operation *op, std::optional<uint64_t> align) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.setAlignment(align);
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(op))
    return fn.setAlignment(align);
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::AliasOp,
          LLVM::ComdatOp>(op))
    return;
  llvm_unreachable("unexpected operation");
}

bool LLVM::LLVMSymbolLinkerInterface::isConstant(Operation *op) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op))
    return gv.getConstant();
  if (isa<LLVM::AliasOp, LLVM::ComdatOp>(op))
    return true;
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp>(op))
    return false;
  llvm_unreachable("unexpected operation");
}

void LLVM::LLVMSymbolLinkerInterface::setIsConstant(Operation *op, bool value) {
  if (auto gv = dyn_cast<LLVM::GlobalOp>(op)) {
    gv.setConstant(value);
    return;
  }
  llvm_unreachable("constness setting allowed only for globals");
}

bool LLVM::LLVMSymbolLinkerInterface::isGlobalVar(Operation *op) {
  return isa<LLVM::GlobalOp>(op);
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
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::AliasOp,
          LLVM::ComdatOp>(op))
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
  if (isa<LLVM::GlobalCtorsOp, LLVM::GlobalDtorsOp, LLVM::ComdatOp>(op))
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
                                             LinkState &state) {
  auto &derived = LinkerMixin::getDerived();
  if (isAppendingLinkage(derived.getLinkage(src))) {
    return derived.appendGlobals(derived.getSymbol(src), state);
  }
  return SymbolAttrLinkerInterface::materialize(src, state);
}

Conflict LLVM::LLVMSymbolLinkerInterface::findConflict(
    Operation *src, SymbolTableCollection &collection) const {
  // First, use the base class to find conflicts
  auto conflict = SymbolAttrLinkerInterface::findConflict(src, collection);

  // If there's a conflict and both are functions, check for type mismatches
  if (conflict.hasConflict()) {
    auto srcFunc = dyn_cast<LLVM::LLVMFuncOp>(src);
    auto dstFunc = dyn_cast<LLVM::LLVMFuncOp>(conflict.dst);

    if (srcFunc && dstFunc) {
      auto srcType = srcFunc.getFunctionType();
      auto dstType = dstFunc.getFunctionType();

      if (srcType != dstType) {
        // Record the mismatch for later fixing
        StringRef funcName = srcFunc.getSymName();
        mismatchedFunctions[funcName] = {srcType, dstType};
      }
    }
  }

  return conflict;
}

SmallVector<Operation *> LLVM::LLVMSymbolLinkerInterface::dependencies(
    Operation *op, SymbolTableCollection &collection) const {
  Operation *module = op->getParentOfType<ModuleOp>();
  // If this operation is nested inside a region (e.g., inside an AliasOp's
  // initializer) or inside a nested module, it won't have dependencies that
  // need linking.
  if (!module || module->getParentOfType<ModuleOp>())
    return {};
  SymbolTable &st = collection.getSymbolTable(module);
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
  ArrayAttr data = {};
  if (auto ctor = dyn_cast<GlobalCtorsOp>(op)) {
    structors = ctor.getCtors();
    data = ctor.getData();
  }
  if (auto dtor = dyn_cast<GlobalDtorsOp>(op)) {
    structors = dtor.getDtors();
    data = dtor.getData();
  }

  if (structors) {
    for (auto structor : structors)
      insertDepIfExists(cast<FlatSymbolRefAttr>(structor));
    for (auto dataAttr : data)
      if (auto symbolRef = dyn_cast<FlatSymbolRefAttr>(dataAttr))
        insertDepIfExists(symbolRef);

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

LogicalResult LLVM::LLVMSymbolLinkerInterface::initialize(ModuleOp src) {
  dtla = src.getDataLayoutSpec();
  targetSys = src.getTargetSystemSpec();
  mismatchedFunctions.clear();
  return success();
}

LogicalResult
LLVM::LLVMSymbolLinkerInterface::verifyLinkageCompatibility(Conflict pair) const {
  // First, run the base class verification
  auto &mixin = static_cast<const link::SymbolAttrLLVMLinkerInterface<LLVMSymbolLinkerInterface>&>(*this);
  if (failed(mixin.LinkerMixin::verifyLinkageCompatibility(pair)))
    return failure();

  // Check if both operations are functions
  auto srcFunc = dyn_cast<LLVM::LLVMFuncOp>(pair.src);
  auto dstFunc = dyn_cast<LLVM::LLVMFuncOp>(pair.dst);

  if (!srcFunc || !dstFunc)
    return success();

  // Compare function types
  auto srcType = srcFunc.getFunctionType();
  auto dstType = dstFunc.getFunctionType();

  if (srcType != dstType) {
    // Types mismatch - record this for later fixing
    StringRef funcName = srcFunc.getSymName();
    mismatchedFunctions[funcName] = {srcType, dstType};

    // Emit a warning but don't fail
    pair.src->emitWarning()
        << "function '" << funcName << "' has mismatched signatures: "
        << srcType << " vs " << dstType
        << " - call sites will be converted to indirect calls";
  }

  return success();
}

LogicalResult
LLVM::LLVMSymbolLinkerInterface::fixMismatchedCallSites(ModuleOp module) const {
  if (mismatchedFunctions.empty())
    return success();

  // Walk through all functions and fix call sites
  auto result = module.walk([&](LLVM::CallOp callOp) -> WalkResult {
    // Only handle direct calls (not indirect calls)
    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return WalkResult::advance();

    StringRef calleeName = calleeAttr.getValue();
    auto it = mismatchedFunctions.find(calleeName);
    if (it == mismatchedFunctions.end())
      return WalkResult::advance();

    // Found a call to a function with mismatched signature
    // Convert it to an indirect call through a function pointer cast

    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    // Get the function being called
    auto funcOp = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        callOp, calleeAttr);
    if (!funcOp) {
      callOp.emitError("cannot find function ") << calleeName;
      return WalkResult::interrupt();
    }

    // Create the expected function type based on the call operands and results
    SmallVector<Type> argTypes;
    for (Value arg : callOp.getArgOperands())
      argTypes.push_back(arg.getType());

    Type resultType;
    if (callOp.getNumResults() == 0) {
      resultType = LLVM::LLVMVoidType::get(builder.getContext());
    } else {
      resultType = callOp.getResult().getType();
    }

    auto expectedFuncType = LLVM::LLVMFunctionType::get(
        resultType, argTypes, callOp.getVarCalleeType().has_value());

    // Get the address of the function
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto addressOfOp = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, calleeAttr.getValue());

    // For indirect calls, the function pointer is passed as the first argument
    // followed by the actual call arguments
    SmallVector<Value> indirectCallArgs;
    indirectCallArgs.push_back(addressOfOp.getResult());
    indirectCallArgs.append(callOp.getArgOperands().begin(),
                           callOp.getArgOperands().end());

    // Create an indirect call with the same attributes as the original call
    // but using the function pointer and expected type
    auto newCallOp = builder.create<LLVM::CallOp>(loc, expectedFuncType,
                                                   indirectCallArgs);

    // Copy over relevant attributes
    if (auto fmf = callOp.getFastmathFlagsAttr())
      newCallOp.setFastmathFlagsAttr(fmf);
    if (auto cconv = callOp.getCConvAttr())
      newCallOp.setCConvAttr(cconv);
    if (auto tailcall = callOp.getTailCallKindAttr())
      newCallOp.setTailCallKindAttr(tailcall);
    if (auto memeff = callOp.getMemoryEffectsAttr())
      newCallOp.setMemoryEffectsAttr(memeff);
    if (callOp.getConvergent())
      newCallOp.setConvergentAttr(builder.getUnitAttr());
    if (callOp.getNoUnwind())
      newCallOp.setNoUnwindAttr(builder.getUnitAttr());
    if (callOp.getWillReturn())
      newCallOp.setWillReturnAttr(builder.getUnitAttr());
    if (auto accessGroups = callOp.getAccessGroupsAttr())
      newCallOp.setAccessGroupsAttr(accessGroups);
    if (auto aliasScopes = callOp.getAliasScopesAttr())
      newCallOp.setAliasScopesAttr(aliasScopes);
    if (auto noaliasScopes = callOp.getNoaliasScopesAttr())
      newCallOp.setNoaliasScopesAttr(noaliasScopes);
    if (auto tbaa = callOp.getTbaaAttr())
      newCallOp.setTbaaAttr(tbaa);

    // Replace the old call with the new indirect call
    callOp.replaceAllUsesWith(newCallOp);
    callOp.erase();

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult LLVM::LLVMSymbolLinkerInterface::link(link::LinkState &state) {
  // First, perform the normal linking process
  if (failed(SymbolAttrLinkerInterface::link(state)))
    return failure();

  // After linking but before verification, fix call sites for functions with
  // mismatched signatures by converting them to indirect calls
  auto dst = dyn_cast<ModuleOp>(state.getDestinationOp());
  if (!dst)
    return success();

  if (failed(fixMismatchedCallSites(dst)))
    return failure();

  return success();
}

LogicalResult LLVM::LLVMSymbolLinkerInterface::finalize(ModuleOp dst) const {
  SmallVector<NamedAttribute, 2> newAttrs;
  // The names are currently hardcoded for dlti dialect
  // Nice solution would be preferable
  if (dtla)
    dst->setAttr(DataLayoutSpecAttr::name, dyn_cast<Attribute>(dtla));
  if (targetSys)
    dst->setAttr(TargetSystemSpecAttr::name, dyn_cast<Attribute>(targetSys));
  return success();
}

LogicalResult LLVM::LLVMSymbolLinkerInterface::moduleOpSummary(
    ModuleOp src, SymbolTableCollection &collection) {
  return resolveComdats(src, collection);
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

  auto [mapping, mutex] = state.getMapping();
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

      Operation *cloned;
      {
        std::lock_guard<std::mutex> lock(mutex);
        cloned = builder.clone(op, mapping);
        mapping.map(&op, cloned);
      }
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

static Operation *appendGlobalOps(ArrayRef<Operation *> globs,
                                  LLVM::GlobalOp lastGV, LinkState &state) {
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

Operation *LLVM::LLVMSymbolLinkerInterface::appendComdatOps(
    ArrayRef<Operation *> globs, LLVM::ComdatOp comdat, LinkState &state) {
  auto result =
      state.create<LLVM::ComdatOp>(comdat.getLoc(), comdat.getSymName());

  auto guard = OpBuilder::InsertionGuard(state.getBuilder());
  state.getBuilder().setInsertionPointToStart(&result.getBody().front());

  for (auto &&[name, comdatResPair] : comdatResolution)
    state.clone(comdatResPair.selectorOp);

  return result;
}

Operation *LLVM::LLVMSymbolLinkerInterface::appendGlobals(llvm::StringRef glob,
                                                          LinkState &state) {
  if (glob == "llvm.global_ctors")
    return appendGlobalStructors<LLVM::GlobalCtorsOp>(state);
  if (glob == "llvm.global_dtors")
    return appendGlobalStructors<LLVM::GlobalDtorsOp>(state);

  const auto &globs = append.lookup(glob);
  if (globs.empty()) {
    if (auto found = summary.find(glob); found != summary.end())
      return state.clone(found->second);
    return nullptr;
  }
  if (auto lastGV = dyn_cast<LLVM::GlobalOp>(globs.back()))
    return appendGlobalOps(globs, lastGV, state);
  if (auto comdat = dyn_cast<LLVM::ComdatOp>(globs.back()))
    return appendComdatOps(globs, comdat, state);
  llvm_unreachable("unexpected operation");
}

ComdatResolution LLVM::LLVMSymbolLinkerInterface::computeComdatResolution(
    Operation *srcSelector, SymbolTableCollection &collection,
    Comdat *dstComdat) {
  // For reference check llvm/lib/Linker/LinkModules.cpp
  // computeResultingSelectionKind
  ComdatKind srcKind = getComdatSelectionKind(srcSelector);
  ComdatKind dstKind = dstComdat->kind;
  bool dstAnyOrLargest =
      dstKind == ComdatKind::Any || dstKind == ComdatKind::Largest;
  bool srcAnyOrLargest =
      srcKind == ComdatKind::Any || srcKind == ComdatKind::Largest;

  ComdatKind resolutionKind;
  if (dstAnyOrLargest && srcAnyOrLargest) {
    if (dstKind == ComdatKind::Largest || srcKind == ComdatKind::Largest) {
      resolutionKind = ComdatKind::Largest;
    } else {
      resolutionKind = ComdatKind::Any;
    }
  } else if (srcKind == dstKind) {
    resolutionKind = dstKind;
  } else {
    return ComdatResolution::Failure;
  }

  auto computeSize = [&](GlobalOp op) -> llvm::TypeSize {
    auto dataLayout = DataLayout(op->getParentOfType<ModuleOp>());
    return dataLayout.getTypeSize(op.getType());
  };

  auto getComdatLeader = [&](Operation *selector) -> GlobalOp {
    SymbolTable &st =
        collection.getSymbolTable(selector->getParentOfType<ModuleOp>());
    Operation *leader = st.lookup(getSymbol(selector));

    while (auto alias = dyn_cast_if_present<AliasOp>(leader))
      for (AddressOfOp addrOf : alias.getInitializer().getOps<AddressOfOp>())
        leader = st.lookup(addrOf.getGlobalName());

    if (hasComdat(leader) &&
        getComdatSymbol(leader).getLeafReference() == getSymbol(leader))
      return mlir::dyn_cast<GlobalOp>(leader);

    return {};
  };

  switch (resolutionKind) {
  case ComdatKind::Any:
    return ComdatResolution::LinkFromDst;
  case ComdatKind::NoDeduplicate:
    return ComdatResolution::LinkFromBoth;
  case ComdatKind::ExactMatch: {
    GlobalOp srcLeader = getComdatLeader(srcSelector);
    GlobalOp dstLeader = getComdatLeader(dstComdat->selectorOp);
    assert(srcLeader && dstLeader && "Couldn't find comdat leader");
    return OperationEquivalence::isEquivalentTo(
               dstLeader, srcLeader,
               OperationEquivalence::Flags::IgnoreLocations)
               ? ComdatResolution::Failure
               : ComdatResolution::LinkFromDst;
  }
  case ComdatKind::Largest: {
    GlobalOp srcLeader = getComdatLeader(srcSelector);
    GlobalOp dstLeader = getComdatLeader(dstComdat->selectorOp);
    assert(srcLeader && dstLeader && "Size based comdat without valid leader");
    return computeSize(srcLeader) > computeSize(dstLeader)
               ? ComdatResolution::LinkFromSrc
               : ComdatResolution::LinkFromDst;
  }
  case ComdatKind::SameSize:
    GlobalOp srcLeader = getComdatLeader(srcSelector);
    GlobalOp dstLeader = getComdatLeader(dstComdat->selectorOp);
    assert(srcLeader && dstLeader && "Size based comdat without valid leader");
    return computeSize(srcLeader) == computeSize(dstLeader)
               ? ComdatResolution::Failure
               : ComdatResolution::LinkFromDst;
  }
}

void LLVM::LLVMSymbolLinkerInterface::dropReplacedComdat(Operation *op) const {
  if (auto global = mlir::dyn_cast<LLVM::GlobalOp>(op)) {
    global.removeValueAttr();

    Region &initializer = global.getInitializer();
    initializer.dropAllReferences();
    initializer.getBlocks().clear();

    global.removeComdatAttr();
    global.setLinkage(Linkage::AvailableExternally);
  }

  if (auto func = mlir::dyn_cast<LLVM::LLVMFuncOp>(op)) {
    Region &body = func.getBody();
    body.dropAllReferences();
    body.getBlocks().clear();

    func.removeComdatAttr();
    func.setLinkage(Linkage::External);
  }

  // Handle aliases: just erase them.
  // Aliases inherit COMDAT from their aliasee, so if the aliasee's COMDAT
  // is dropped, the alias should be dropped too.
  if (auto alias = mlir::dyn_cast<LLVM::AliasOp>(op)) {
    alias.erase();
  }
}

void LLVM::LLVMSymbolLinkerInterface::updateNoDeduplicate(Operation *op) {
  if (auto global = mlir::dyn_cast<LLVM::GlobalOp>(op)) {
    global.setVisibility_(LLVM::Visibility::Default);
    global.setDsoLocal(true);
    global.setLinkage(LLVM::Linkage::Private);
  } else {
    llvm_unreachable("Only globals should have NoDeduplicate comdat");
  }
}

LogicalResult LLVM::LLVMSymbolLinkerInterface::resolveComdats(
    ModuleOp srcMod, SymbolTableCollection &collection) {
  LLVM::ComdatOp srcComdatOp =
      collection.getSymbolTable(srcMod).lookup<LLVM::ComdatOp>(
          "__llvm_global_comdat");

  // Nothing to do
  if (!srcComdatOp)
    return success();

  // TODO: Figure out how to share this map with the rest of the linker
  SymbolUserMap srcSymbolUsers(collection,
                               srcComdatOp->getParentOfType<ModuleOp>());

  for (Operation &op : srcComdatOp.getBody().front()) {
    auto srcSelector = cast<LLVM::ComdatSelectorOp>(op);
    auto dstComdatIt = comdatResolution.find(getSymbol(&op));
    link::Comdat *dstComdat =
        dstComdatIt == comdatResolution.end() ? nullptr : &dstComdatIt->second;

    // If no conflict choose src
    ComdatResolution res =
        dstComdat ? computeComdatResolution(srcSelector, collection, dstComdat)
                  : ComdatResolution::LinkFromSrc;

    switch (res) {
    case ComdatResolution::LinkFromSrc: {
      // COMDAT group is used or dropped as a whole, remove all users of dropped
      // COMDAT if present
      // Drop all users before replacing the value
      if (dstComdat)
        for (Operation *dstUser : dstComdat->users)
          dropReplacedComdat(dstUser);
      ArrayRef<Operation *> users = srcSymbolUsers.getUsers(&op);
      comdatResolution[getSymbol(srcSelector)] =
          link::Comdat{getComdatSelectionKind(srcSelector),
                       srcSelector,
                       {users.begin(), users.end()}};
      break;
    }
    case ComdatResolution::LinkFromDst:
      continue;
    case ComdatResolution::LinkFromBoth: {
      ArrayRef<Operation *> users = srcSymbolUsers.getUsers(&op);
      dstComdat->users.insert(users.begin(), users.end());
      break;
    }
    case ComdatResolution::Failure:
      return failure();
    }
  }
  return success();
}

const link::Comdat *
LLVM::LLVMSymbolLinkerInterface::getComdatResolution(Operation *op) const {
  if (hasComdat(op)) {
    auto resolutionIt = comdatResolution.find(
        getComdatSymbol(op).getLeafReference().getValue());
    return resolutionIt != comdatResolution.end() ? &resolutionIt->second
                                                  : nullptr;
  }
  return nullptr;
}

bool LLVM::LLVMSymbolLinkerInterface::selectedByComdat(Operation *op) const {
  assert(hasComdat(op) && "expected operation with comdat");

  if (auto comdatIt = comdatResolution.find(
          getComdatSymbol(op).getLeafReference().getValue());
      comdatIt != comdatResolution.end()) {
    return comdatIt->second.users.contains(op);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void mlir::LLVM::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    dialect->addInterfaces<LLVMSymbolLinkerInterface>();
  });
}
