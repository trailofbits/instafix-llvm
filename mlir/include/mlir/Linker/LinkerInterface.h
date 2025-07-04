//===- LinkerInterface.h - MLIR Linker Interface ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines interfaces and utilities necessary for dialects
// to hook into mlir linker.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKERINTERFACE_H
#define MLIR_LINKER_LINKERINTERFACE_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace mlir::link {

//===----------------------------------------------------------------------===//
// LinkerInterface
//===----------------------------------------------------------------------===//

enum LinkerFlags {
  None = 0,
  OverrideFromSrc = (1 << 0),
  LinkOnlyNeeded = (1 << 1),
};

class LinkState {
public:
  LinkState(ModuleOp dst)
      : mapping(std::make_shared<IRMapping>()), builder(dst.getBodyRegion()) {}

  Operation *clone(Operation *src);
  Operation *cloneWithoutRegions(Operation *src);

  Operation *getDestinationOp() const;

  Operation *remapped(Operation *src) const;

  LinkState nest(ModuleOp submod) const;

  IRMapping &getMapping();

private:
  // Private constructor used by nest()
  LinkState(ModuleOp dst, std::shared_ptr<IRMapping> mapping)
      : mapping(std::move(mapping)), builder(dst.getBodyRegion()) {}

  std::shared_ptr<IRMapping> mapping;
  OpBuilder builder;
};

struct Conflict {
  Operation *dst;
  Operation *src;

  bool hasConflict() const { return dst; }

  static Conflict noConflict(Operation *src) { return {nullptr, src}; }
};

template <typename ConcreteType>
class LinkerInterface : public DialectInterface::Base<ConcreteType> {
public:
  LinkerInterface(Dialect *dialect)
      : DialectInterface::Base<ConcreteType>(dialect) {}

  /// Runs initialization of alinker before summarization for the given module
  virtual LogicalResult initialize(ModuleOp src) { return success(); }

  /// Link operations from current summary using state builder
  virtual LogicalResult link(LinkState &state) const = 0;
};

//===----------------------------------------------------------------------===//
// ModuleLinkerInterface
//===----------------------------------------------------------------------===//

class ModuleLinkerInterface : public LinkerInterface<ModuleLinkerInterface> {
public:
  using LinkerInterface::LinkerInterface;

  /// TODO comment
  virtual LogicalResult summarize(ModuleOp src, unsigned flags) = 0;

  /// TODO comment
  virtual OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) = 0;
};

//===----------------------------------------------------------------------===//
// SymbolLinkerInterface
//===----------------------------------------------------------------------===//

class SymbolLinkerInterface : public LinkerInterface<SymbolLinkerInterface> {
public:
  using LinkerInterface::LinkerInterface;

  /// Determines if the given operation is eligible for linking.
  virtual bool canBeLinked(Operation *op) const = 0;

  /// Returns the symbol for the given operation.
  virtual StringRef getSymbol(Operation *op) const = 0;

  /// Determines if an operation should be linked into the destination module.
  virtual bool isLinkNeeded(Conflict pair, bool forDependency) const = 0;

  /// Checks if an operation conflicts with existing linked operations.
  virtual Conflict findConflict(Operation *src) const = 0;

  /// Resolves a conflict between an existing operation and a new one.
  virtual LogicalResult resolveConflict(Conflict pair) = 0;

  /// Records a non-conflicting operation for linking.
  virtual void registerForLink(Operation *op) = 0;

  /// Materialize new operation for the given conflict src operation.
  virtual Operation *materialize(Operation *src, LinkState &state) const {
    return state.clone(src);
  }

  /// Dependencies of the given operation required to be linked.
  virtual SmallVector<Operation *> dependencies(Operation *op) const = 0;

  void setFlags(unsigned flags) { this->flags = flags; }

  bool shouldLinkOnlyNeeded() const {
    return flags & LinkerFlags::LinkOnlyNeeded;
  }

  bool shouldOverrideFromSrc() const {
    return flags & LinkerFlags::OverrideFromSrc;
  }

protected:
  unsigned flags = LinkerFlags::None;
};

//===----------------------------------------------------------------------===//
// SymbolAttrLinkerInterface
//===----------------------------------------------------------------------===//
enum class ConflictResolution {
  LinkFromSrc,
  LinkFromDst,
  LinkFromBothAndRenameDst,
  LinkFromBothAndRenameSrc,
  Failure,
};

class SymbolAttrLinkerInterface : public SymbolLinkerInterface {
public:
  using SymbolLinkerInterface::SymbolLinkerInterface;

  /// Link operations from current summary using state builder
  LogicalResult link(LinkState &state) const override;

  /// Returns the symbol for the given operation.
  StringRef getSymbol(Operation *op) const override;

  /// Checks if an operation conflicts with existing linked operations.
  Conflict findConflict(Operation *src) const override;

  /// Records a non-conflicting operation for linking.
  void registerForLink(Operation *op) override;

  /// Resolves a conflict between an existing operation and a new one.
  LogicalResult resolveConflict(Conflict pair) override;

  virtual LogicalResult resolveConflict(Conflict pair,
                                        ConflictResolution resolution);

  /// Gets the conflict resolution for a given conflict
  virtual ConflictResolution getConflictResolution(Conflict pair) const = 0;

  virtual LogicalResult verifyLinkageCompatibility(Conflict pair) const = 0;

  /// Dependencies of the given operation required to be linked.
  SmallVector<Operation *> dependencies(Operation *op) const override;

protected:
  // Operations that are to be linked with the original name.
  llvm::StringMap<Operation *> summary;

  // Operations that are to be linked with unique names.
  SetVector<Operation *> uniqued;
};

//===----------------------------------------------------------------------===//
// SymbolLinkerInterfaceCollection
//===----------------------------------------------------------------------===//

// TODO: Fix DialectInterfaceCollection to allow non-const access to interfaces.
struct InterfaceKeyInfo : public DenseMapInfo<DialectInterface *> {
  using DenseMapInfo<DialectInterface *>::isEqual;

  static unsigned getHashValue(Dialect *key) { return llvm::hash_value(key); }
  static unsigned getHashValue(DialectInterface *key) {
    return getHashValue(key->getDialect());
  }

  static bool isEqual(Dialect *lhs, DialectInterface *rhs) {
    if (rhs == getEmptyKey() || rhs == getTombstoneKey())
      return false;
    return lhs == rhs->getDialect();
  }
};

class SymbolLinkerInterfaces {
public:
  SymbolLinkerInterfaces() = default;

  SymbolLinkerInterfaces(MLIRContext *ctx) {
    for (auto *dialect : ctx->getLoadedDialects()) {
      if (auto *iface =
              dialect->getRegisteredInterface<SymbolLinkerInterface>()) {
        interfaces.insert(iface);
      }
    }
  }

  LogicalResult initialize(ModuleOp src) {
    for (SymbolLinkerInterface *linker : interfaces) {
      if (failed(linker->initialize(src)))
        return failure();
    }
    return success();
  }

  LogicalResult link(LinkState &state) const {
    for (SymbolLinkerInterface *linker : interfaces) {
      if (failed(linker->link(state)))
        return failure();
    }
    return success();
  }

  Conflict findConflict(Operation *src) const {
    for (SymbolLinkerInterface *linker : interfaces) {
      if (auto pair = linker->findConflict(src); pair.hasConflict())
        return pair;
    }
    return Conflict::noConflict(src);
  }

private:
  SetVector<SymbolLinkerInterface *> interfaces;
};

} // namespace mlir::link

#endif // MLIR_LINKER_LINKERINTERFACE_H
