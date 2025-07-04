//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/LinkerInterface.h"

#define DEBUG_TYPE "mlir-linker-interface"

using namespace mlir;
using namespace mlir::link;

//===----------------------------------------------------------------------===//
// LinkState
//===----------------------------------------------------------------------===//

template <typename CloneFunc>
Operation *cloneImpl(Operation *src, std::shared_ptr<IRMapping> &mapping,
                     CloneFunc cloneFunc) {
  assert(!mapping->contains(src));
  Operation *dst = cloneFunc(src);
  mapping->map(src, dst);
  return dst;
}

Operation *LinkState::clone(Operation *src) {
  return cloneImpl(src, mapping, [this](Operation *op) {
    return builder.clone(*op, *mapping);
  });
}

Operation *LinkState::cloneWithoutRegions(Operation *src) {
  return cloneImpl(src, mapping, [this](Operation *op) {
    return builder.cloneWithoutRegions(*op, *mapping);
  });
}

Operation *LinkState::getDestinationOp() const {
  return builder.getInsertionBlock()->getParentOp();
}

Operation *LinkState::remapped(Operation *src) const {
  return mapping->lookupOrNull(src);
}

LinkState LinkState::nest(ModuleOp submod) const {
  assert(submod->getParentOfType<mlir::ModuleOp>().getOperation() ==
             getDestinationOp() &&
         "Submodule should be directly nested in the current state");
  return LinkState(submod, mapping);
}

IRMapping &LinkState::getMapping() { return *mapping.get(); }

//===----------------------------------------------------------------------===//
// SymbolAttrLinkerInterface
//===----------------------------------------------------------------------===//

static StringAttr symbolAttr(Operation *op) {
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

static StringAttr getUniqueNameIn(SymbolTable &st, StringAttr name) {
  MLIRContext *context = name.getContext();
  int uniqueId = 0;
  Twine prefix = name.getValue() + ".";
  while (st.lookup(name))
    name = StringAttr::get(context, prefix + Twine(uniqueId++));
  return name;
}

static void renameSymbolRefIn(Operation *op, StringAttr newName) {
  AttrTypeReplacer replacer;
  replacer.addReplacement([&](SymbolRefAttr attr) {
    return SymbolRefAttr::get(newName, attr.getNestedReferences());
  });
  replacer.replaceElementsIn(op);
}

static LogicalResult renameRemappedUsersOf(Operation *op, StringAttr newName,
                                           LinkState &state) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  // TODO: use something like SymbolTableAnalysis
  SymbolTable src(module);
  if (auto uses = src.getSymbolUses(op, module)) {
    for (SymbolTable::SymbolUse use : *uses) {
      // TODO: add test where user is not remapped
      Operation *dstUser = state.remapped(use.getUser());
      if (!dstUser)
        continue;
      renameSymbolRefIn(dstUser, newName);
    }

    return success();
  }

  return op->emitError("failed to rename symbol to a unique name");
}

StringRef SymbolAttrLinkerInterface::getSymbol(Operation *op) const {
  return symbolAttr(op).getValue();
}

Conflict SymbolAttrLinkerInterface::findConflict(Operation *src) const {
  assert(canBeLinked(src) && "expected linkable operation");
  StringRef symbol = getSymbol(src);
  auto it = summary.find(symbol);
  if (it == summary.end())
    return Conflict::noConflict(src);
  return {it->second, src};
}

void SymbolAttrLinkerInterface::registerForLink(Operation *op) {
  assert(canBeLinked(op) && "expected linkable operation");
  summary[getSymbol(op)] = op;
}

LogicalResult SymbolAttrLinkerInterface::link(LinkState &state) const {
  SymbolTable st(state.getDestinationOp());

  auto materializeError = [&](Operation *op) {
    return op->emitError("failed to materialize symbol for linking");
  };

  for (const auto &[symbol, op] : summary) {
    Operation *materialized = materialize(op, state);
    if (!materialized)
      return materializeError(op);

    if (isa<SymbolOpInterface>(op))
      st.insert(materialized);
  }

  std::vector<std::pair<Operation *, StringAttr>> toRenameUsers;

  for (Operation *op : uniqued) {
    Operation *materialized = materialize(op, state);
    if (!materialized)
      return materializeError(op);

    StringAttr name = symbolAttr(materialized);
    if (st.lookup(name)) {
      StringAttr newName = getUniqueNameIn(st, name);
      st.setSymbolName(materialized, newName);
      toRenameUsers.push_back({op, newName});
    }

    st.insert(materialized);
  }

  for (auto &[op, newName] : toRenameUsers) {
    if (failed(renameRemappedUsersOf(op, newName, state)))
      return failure();
  }

  return success();
}

SmallVector<Operation *>
SymbolAttrLinkerInterface::dependencies(Operation *op) const {
  // TODO: use something like SymbolTableAnalysis
  Operation *module = op->getParentOfType<ModuleOp>();
  SymbolTable st(module);
  SmallVector<Operation *> result;
  op->walk([&](SymbolUserOpInterface user) {
    if (user.getOperation() == op)
      return;

    if (SymbolRefAttr symbol = user.getUserSymbol()) {
      if (Operation *dep = st.lookup(symbol.getRootReference())) {
        result.push_back(dep);
      }
    }
  });

  return result;
}

LogicalResult
SymbolAttrLinkerInterface::resolveConflict(Conflict pair,
                                           ConflictResolution resolution) {

  switch (resolution) {
  case ConflictResolution::LinkFromSrc:
    registerForLink(pair.src);
    return success();
  case ConflictResolution::LinkFromDst:
    return success();
  case ConflictResolution::LinkFromBothAndRenameDst:
    uniqued.insert(pair.dst);
    registerForLink(pair.src);
    return success();
  case ConflictResolution::LinkFromBothAndRenameSrc:
    uniqued.insert(pair.src);
    return success();
  case ConflictResolution::Failure:
    return pair.src->emitError("Linker error");
  }

  llvm_unreachable("unimplemented conflict resolution");
}

LogicalResult SymbolAttrLinkerInterface::resolveConflict(Conflict pair) {
  if (failed(this->verifyLinkageCompatibility(pair)))
    return failure();
  ConflictResolution resolution = this->getConflictResolution(pair);
  return resolveConflict(pair, resolution);
}
