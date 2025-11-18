//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/LinkerInterface.h"

#include "mlir/IR/Threading.h"

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
  std::lock_guard<std::mutex> lock(*mutex);
  return cloneImpl(src, mapping, [this](Operation *op) {
    return builder.clone(*op, *mapping);
  });
}

Operation *LinkState::cloneWithoutRegions(Operation *src) {
  std::lock_guard<std::mutex> lock(*mutex);
  return cloneImpl(src, mapping, [this](Operation *op) {
    return builder.cloneWithoutRegions(*op, *mapping);
  });
}

Operation *LinkState::getDestinationOp() const {
  std::lock_guard<std::mutex> lock(*mutex);
  return builder.getInsertionBlock()->getParentOp();
}

Operation *LinkState::remapped(Operation *src) const {
  std::lock_guard<std::mutex> lock(*mutex);
  return mapping->lookupOrNull(src);
}

LinkState LinkState::nest(ModuleOp submod) const {
  assert(submod->getParentOfType<mlir::ModuleOp>().getOperation() ==
             getDestinationOp() &&
         "Submodule should be directly nested in the current state");
  return LinkState(submod, mapping, mutex, symbolTableCollection);
}

std::pair<IRMapping &, std::mutex &> LinkState::getMapping() { return {*mapping, *mutex}; }
SymbolUserMap &LinkState::getSymbolUserMap(ModuleOp mod) {
  std::lock_guard<std::mutex> lock(*mutex);
  return moduleMaps.try_emplace(mod, symbolTableCollection, mod).first->second;
}

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

  // Get a mapping of all the users
  SymbolUserMap &userMap = state.getSymbolUserMap(module);

  auto users = userMap.getUsers(op);
  for (Operation *user : users) {
    Operation *dstUser = state.remapped(user);
    if (!dstUser)
      continue;

    renameSymbolRefIn(dstUser, newName);
  }
  return success();
}

StringRef SymbolAttrLinkerInterface::getSymbol(Operation *op) const {
  return symbolAttr(op).getValue();
}

Conflict SymbolAttrLinkerInterface::findConflict(Operation *src,
                                                 SymbolTableCollection &collection) const {
  assert(canBeLinked(src) && "expected linkable operation");
  StringRef symbol = getSymbol(src);
  std::lock_guard<std::mutex> lock(summaryMutex);
  auto it = summary.find(symbol);
  if (it == summary.end())
    return Conflict::noConflict(src);
  return {it->second, src};
}

void SymbolAttrLinkerInterface::registerForLink(Operation *op,
                                                SymbolTableCollection &collection) {
  assert(canBeLinked(op) && "expected linkable operation");
  std::lock_guard<std::mutex> lock(summaryMutex);
  summary[getSymbol(op)] = op;
}

LogicalResult SymbolAttrLinkerInterface::link(LinkState &state) {
  SymbolTable &st =
      state.getSymbolTableCollection().getSymbolTable(state.getDestinationOp());

  auto materializeError = [&](Operation *op) {
    return op->emitError("failed to materialize symbol for linking");
  };

  // Materialize symbols from summary in parallel
  std::mutex insertMutex;
  std::vector<std::pair<StringRef, Operation *>> summaryVec;
  summaryVec.reserve(summary.size());
  for (const auto &entry : summary) {
    summaryVec.emplace_back(entry.getKey(), entry.getValue());
  }

  LogicalResult result = failableParallelForEach(
      state.getDestinationOp()->getContext(), summaryVec,
      [&](const std::pair<StringRef, Operation *> &pair) -> LogicalResult {
        Operation *op = pair.second;
        Operation *materialized = materialize(op, state);
        if (!materialized)
          return materializeError(op);

        if (isa<SymbolOpInterface>(op)) {
          std::lock_guard<std::mutex> lock(insertMutex);
          st.insert(materialized);
        }
        return success();
      });

  if (failed(result))
    return failure();

  // Materialize uniqued symbols in parallel
  std::vector<std::pair<Operation *, StringAttr>> toRenameUsers;
  std::mutex renameMutex;

  SmallVector<Operation *> uniquedVec(uniqued.begin(), uniqued.end());

  result = failableParallelForEach(
      state.getDestinationOp()->getContext(), uniquedVec,
      [&](Operation *op) -> LogicalResult {
        Operation *materialized = materialize(op, state);
        if (!materialized)
          return materializeError(op);

        StringAttr name = symbolAttr(materialized);
        StringAttr newName = name;
        bool needsRename = false;

        {
          std::lock_guard<std::mutex> lock(insertMutex);
          if (st.lookup(name)) {
            newName = getUniqueNameIn(st, name);
            st.setSymbolName(materialized, newName);
            needsRename = true;
          }
          st.insert(materialized);
        }

        if (needsRename) {
          std::lock_guard<std::mutex> lock(renameMutex);
          toRenameUsers.emplace_back(op, newName);
        }

        return success();
      });

  if (failed(result))
    return failure();

  // Rename users (sequential, but typically small)
  for (auto &[op, newName] : toRenameUsers) {
    if (failed(renameRemappedUsersOf(op, newName, state)))
      return failure();
  }

  return success();
}

SmallVector<Operation *> SymbolAttrLinkerInterface::dependencies(
    Operation *op, SymbolTableCollection &collection) const {
  // TODO: use something like SymbolTableAnalysis
  Operation *module = op->getParentOfType<ModuleOp>();
  SymbolTable &st = collection.getSymbolTable(module);
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
                                           ConflictResolution resolution,
                                           SymbolTableCollection &collection) {

  switch (resolution) {
  case ConflictResolution::LinkFromSrc:
    registerForLink(pair.src, collection);
    return success();
  case ConflictResolution::LinkFromDst:
    return success();
  case ConflictResolution::LinkFromBothAndRenameDst:
    {
      std::lock_guard<std::mutex> lock(summaryMutex);
      uniqued.insert(pair.dst);
    }
    registerForLink(pair.src, collection);
    return success();
  case ConflictResolution::LinkFromBothAndRenameSrc:
    {
      std::lock_guard<std::mutex> lock(summaryMutex);
      uniqued.insert(pair.src);
    }
    return success();
  case ConflictResolution::Failure:
    return pair.src->emitError("Linker error");
  }

  llvm_unreachable("unimplemented conflict resolution");
}

LogicalResult SymbolAttrLinkerInterface::resolveConflict(Conflict pair, SymbolTableCollection &collection) {
  if (failed(this->verifyLinkageCompatibility(pair)))
    return failure();
  ConflictResolution resolution = this->getConflictResolution(pair);
  return resolveConflict(pair, resolution, collection);
}
