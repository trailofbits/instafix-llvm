//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Linker/LinkerInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"

#define DEBUG_TYPE "mlir-linker-interface"

using namespace mlir;
using namespace mlir::link;

template <typename CloneFunc>
Operation *cloneImpl(Operation *src, IRMapping &mapping, CloneFunc cloneFunc) {
  assert(!mapping.contains(src));
  Operation *dst = cloneFunc(src);
  mapping.map(src, dst);
  return dst;
}

Operation *LinkState::clone(Operation *src) {
    return cloneImpl(src, mapping, [this](Operation *op) {
        return builder.clone(*op, mapping);
    });
}

Operation *LinkState::cloneWithoutRegions(Operation *src) {
    return cloneImpl(src, mapping, [this](Operation *op) {
        return builder.cloneWithoutRegions(*op, mapping);
    });
}

Operation *LinkState::getDestinationOp() const {
  return builder.getInsertionBlock()->getParentOp();
}

Operation *LinkState::remapped(Operation *src) const {
  return mapping.lookupOrNull(src);
}


LinkState LinkState::nest(ModuleOp submod) const {
  assert(submod->getParentOfType<mlir::ModuleOp>().getOperation() == getDestinationOp() && "Submodule should be directly nested in the current state");
  LinkState submodState(submod);
  submodState.mapping = mapping;
  return submodState;
}

void LinkState::updateState(const LinkState &submodState) {
mapping = submodState.mapping;
}



  StringAttr UniqueableSymbolLinker::getUniqueNameIn(SymbolTable &st, StringAttr name) const {
    MLIRContext *context = name.getContext();
    int uniqueId = 0;
    Twine prefix = name.getValue() + ".";
    while (st.lookup(name))
      name = StringAttr::get(context, prefix + Twine(uniqueId++));
    return name;
  }

  void UniqueableSymbolLinker::renameSymbolRefIn(Operation *op, StringAttr newName) const {
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](SymbolRefAttr attr) {
      return SymbolRefAttr::get(newName, attr.getNestedReferences());
    });
    replacer.replaceElementsIn(op);
  }

  LogicalResult UniqueableSymbolLinker::renameRemappedUsersOf(Operation *op, StringAttr newName,
                                      LinkState &state) const {
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



  LogicalResult UniqueableSymbolLinker::link(LinkState &state) const {
    SymbolTable st(state.getDestinationOp());

    auto materializeError = [&](Operation *op) {
      return op->emitError("failed to materialize symbol for linking");
    };

    for (const auto &[symbol, op] : summary) {
      Operation *materialized = materialize(op, state);
      if (!materialized)
        return materializeError(op);

      st.insert(materialized);
    }

    std::vector<std::pair<Operation *, StringAttr>> toRenameUsers;

    for (Operation *op : uniqued) {
      Operation *materialized = materialize(op, state);
      if (!materialized)
        return materializeError(op);

      StringAttr name = StringAttr::get(materialized->getContext(), this->getSymbol(materialized));
    
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


mlir::Operation *UniqueableSymbolLinker::materialize(mlir::Operation *src, LinkState &state) const {
  return state.clone(src);
}


void UniqueableSymbolLinker::registerForLink(Operation *op) {
  this->summary[this->getSymbol(op)] = op;
}

llvm::LogicalResult UniqueableSymbolLinker::applyResolution(link::Conflict pair, link::ConflictResolution resolution) {
  switch (resolution) {
    case ConflictResolution::Ignore:
      return success();
    case ConflictResolution::Import:
      this->registerForLink(pair.src);
      return success();
    case ConflictResolution::RenameDst:
      this->uniqued.insert(pair.dst);
      this->registerForLink(pair.src);
      return success();
    case ConflictResolution::RenameSrc:
      this->uniqued.insert(pair.src);
      return success();
      
  }
}
