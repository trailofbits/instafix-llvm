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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectInterface.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir::link {

//===----------------------------------------------------------------------===//
// LinkerStateBase
//===----------------------------------------------------------------------===//

struct LinkerState {
  virtual ~LinkerState() = default;

  TypeID getID() const { return id; }

  template <typename ConcreteType, typename BaseType>
  struct LinkerStateBase : BaseType {
    using Base = LinkerStateBase<ConcreteType, BaseType>;

    static TypeID getTypeID() { return TypeID::get<ConcreteType>(); }

    static bool classof(const LinkerState *state) {
      return state->getID() == TypeID::get<ConcreteType>();
    }

  protected:
    LinkerStateBase() : BaseType(getTypeID()) {}
  };

  template <typename ConcreteType>
  using Base = LinkerStateBase<ConcreteType, LinkerState>;

  /// Lookup the operation in the linker state of the destination module.
  virtual Operation *lookup(Operation *op) const { return nullptr; }

protected:
  explicit LinkerState(TypeID id) : id(id) {}

private:
  TypeID id;
};

//===----------------------------------------------------------------------===//
// LinkerInterface
//===----------------------------------------------------------------------===//

enum LinkerFlags {
  None = 0,
  OverrideFromSrc = (1 << 0),
  LinkOnlyNeeded = (1 << 1),
};

struct LinkerInterface : DialectInterface::Base<LinkerInterface> {
  using DialectInterface::Base<LinkerInterface>::Base;

  template <typename ConcreteType, typename BaseType, typename StateType>
  struct LinkerInterfaceBase : BaseType {
    using BaseType::BaseType;

    const StateType &getLinkerState() const { return cast<StateType>(*state); }
    StateType &getLinkerState() { return cast<StateType>(*state); }

    static bool classof(const LinkerInterface *iface) {
      return iface->getInterfaceID() == TypeID::get<ConcreteType>();
    }

    LogicalResult initialize(ModuleOp src) override {
      if (!state)
        state = init(src);
      return success();
    }

    Operation *lookup(Operation *op) const override {
      return getLinkerState().lookup(op);
    }

  private:
    std::unique_ptr<LinkerState> state = nullptr;
  };

  template <typename ConcreteType, typename StateType>
  using Base = LinkerInterfaceBase<ConcreteType, LinkerInterface, StateType>;

  virtual LogicalResult initialize(ModuleOp src) = 0;

  virtual LogicalResult materialize(ModuleOp dst) const = 0;

  virtual Operation *lookup(Operation *op) const = 0;

protected:
  virtual std::unique_ptr<LinkerState> init(ModuleOp src) const = 0;
};

//===----------------------------------------------------------------------===//
// ModuleLinkerInterface
//===----------------------------------------------------------------------===//

struct ModuleLinkerInterface : LinkerInterface {
  template <typename ConcreteType, typename StateType>
  using Base =
      LinkerInterfaceBase<ConcreteType, ModuleLinkerInterface, StateType>;

  using LinkerInterface::LinkerInterface;

  virtual LogicalResult process(ModuleOp src, unsigned flags) = 0;

  virtual OwningOpRef<ModuleOp> createCompositeModule(ModuleOp src) = 0;
};

//===----------------------------------------------------------------------===//
// SymbolLinkerInterface
//===----------------------------------------------------------------------===//

struct SymbolLinkerInterface : LinkerInterface {
  template <typename ConcreteType, typename StateType>
  using Base =
      LinkerInterfaceBase<ConcreteType, SymbolLinkerInterface, StateType>;

  using LinkerInterface::LinkerInterface;

  /// Determines if the given operation is eligible for linking.
  virtual bool canBeLinked(Operation *op) const = 0;

  /// Checks if an operation conflicts with existing linked operations.
  /// @param src The operation to check for conflicts.
  /// @return Pointer to the conflicting operation if found, nullptr otherwise.
  virtual Operation *findConflict(Operation *src) const = 0;

  /// Determines if an operation should be linked into the destination module.
  /// @param dst An optional conflicting operation in the destination
  /// module.
  /// @param src The operation from the source module to be evaluated.
  /// @return True if src should be linked, false otherwise.
  virtual bool isLinkNeeded(Operation *dst, Operation *src) const = 0;

  /// Resolves a conflict between an existing operation and a new one.
  /// @param dst The operation already present in the destination module.
  /// @param src The conflicting operation from the source module.
  /// @return Success if resolved, failure otherwise.
  virtual LogicalResult resolveConflict(Operation *dst, Operation *src) = 0;

  /// Records a non-conflicting operation for linking.
  virtual void registerOperation(Operation *op) = 0;

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

} // namespace mlir::link

#endif // MLIR_LINKER_LINKERINTERFACE_H
