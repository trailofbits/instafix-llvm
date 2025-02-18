//===- BytecodeImplementation.h - MLIR Bytecode Implementation --*- C++ -*-===//
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

#ifndef MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
#define MLIR_LINKER_LINKAGEDIALECTINTERFACE_H

#include "mlir/IR/DialectInterface.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir::link {

//===----------------------------------------------------------------------===//
// LinkerInterface
//===----------------------------------------------------------------------===//

using ComdatPair = std::pair<StringRef, ::mlir::link::ComdatSelectionKind>;

class LinkerInterface : public DialectInterface::Base<LinkerInterface> {
public:
  LinkerInterface(Dialect *dialect) : Base(dialect) {}

  virtual bool isDeclaration(GlobalValueLinkageOpInterface op) const {
    return false;
  }

  bool isDeclarationForLinker(GlobalValueLinkageOpInterface op) const {
    if (op.hasAvailableExternallyLinkage())
      return true;
    return isDeclaration(op);
  }
};

struct LinkableOp {
  LinkableOp() = default;

  explicit LinkableOp(Operation *op)
      : op(op), linker(cast<LinkerInterface>(op->getDialect())) {}

  operator bool() const { return op; }

  Operation *getOperation() const { return op; }

protected:
  Operation *op;
  LinkerInterface *linker;
};

template <typename Interface>
struct SpecificLinkableOp : LinkableOp {
  SpecificLinkableOp() = default;
  SpecificLinkableOp(Interface op) : LinkableOp(op) {}

  Interface interface() const { return cast<Interface>(op); }
  Interface operator*() const { return interface(); }

  operator Interface() { return cast<Interface>(op); }
  operator Interface() const { return cast<Interface>(op); }
};

struct GlobalValue : SpecificLinkableOp<GlobalValueLinkageOpInterface> {
  using SpecificLinkableOp::SpecificLinkableOp;

  using SpecificLinkableOp::interface;

  bool isDeclaration() const { return linker->isDeclaration(interface()); }

  bool isDeclarationForLinker() const {
    return linker->isDeclarationForLinker(interface());
  }

  bool hasExternalLinkage() const { return interface().hasExternalLinkage(); }

  bool hasAvailableExternallyLinkage() const {
    return interface().hasAvailableExternallyLinkage();
  }

  bool hasLinkOnceLinkage() const { return interface().hasLinkOnceLinkage(); }

  bool hasLinkOnceAnyLinkage() const {
    return interface().hasLinkOnceAnyLinkage();
  }

  bool hasLinkOnceODRLinkage() const {
    return interface().hasLinkOnceODRLinkage();
  }

  bool hasWeakLinkage() const { return interface().hasWeakLinkage(); }

  bool hasWeakAnyLinkage() const { return interface().hasWeakAnyLinkage(); }

  bool hasWeakODRLinkage() const { return interface().hasWeakODRLinkage(); }

  bool hasAppendingLinkage() const { return interface().hasAppendingLinkage(); }

  bool hasInternalLinkage() const { return interface().hasInternalLinkage(); }

  bool hasPrivateLinkage() const { return interface().hasPrivateLinkage(); }

  bool hasLocalLinkage() const { return interface().hasLocalLinkage(); }

  bool hasExternalWeakLinkage() const {
    return interface().hasExternalWeakLinkage();
  }

  bool hasCommonLinkage() const { return interface().hasCommonLinkage(); }

  ::mlir::link::Linkage getLinkage() const { return interface().getLinkage(); }

  FailureOr<StringRef> getLinkedName() const {
    return interface().getLinkedName();
  }

  std::optional<StringRef> getComdatName() const {
    return interface().getComdatName();
  }

  std::optional<::mlir::link::ComdatPair> getComdatPair() const {
    return interface().getComdatPair();
  }
};

struct GlobalVariable : SpecificLinkableOp<GlobalVariableLinkageOpInterface> {
  using SpecificLinkableOp::SpecificLinkableOp;

  bool isConstant() const { return interface().isConstant(); }
};

struct Function : SpecificLinkableOp<FunctionLinkageOpInterface> {
  using SpecificLinkableOp::SpecificLinkableOp;
};

} // namespace mlir::link

namespace llvm {

///
/// LinkableOp
///
template <typename T>
struct CastInfo<T, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, ::mlir::link::LinkableOp &,
                                     CastInfo<T, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return T::classof(op.getOperation());
  }

  static T doCast(::mlir::link::LinkableOp &op) { return T(op.getOperation()); }
};

template <>
struct CastInfo<::mlir::link::GlobalVariable, ::mlir::link::LinkableOp>
    : public NullableValueCastFailed<::mlir::link::GlobalVariable>,
      public DefaultDoCastIfPossible<
          ::mlir::link::GlobalVariable, ::mlir::link::LinkableOp &,
          CastInfo<::mlir::link::GlobalVariable, ::mlir::link::LinkableOp>> {

  static bool isPossible(::mlir::link::LinkableOp &op) {
    return ::mlir::GlobalVariableLinkageOpInterface::classof(op.getOperation());
  }

  static ::mlir::link::GlobalVariable doCast(::mlir::link::LinkableOp &op) {
    return ::mlir::link::GlobalVariable(
        cast<::mlir::GlobalVariableLinkageOpInterface>(op.getOperation()));
  }
};

template <typename T>
struct CastInfo<T, const ::mlir::link::LinkableOp>
    : public ConstStrippingForwardingCast<
          T, const ::mlir::link::LinkableOp,
          CastInfo<T, ::mlir::link::LinkableOp>> {};

template <>
struct DenseMapInfo<::mlir::link::LinkableOp>
    : public DenseMapInfo<::mlir::Operation *> {};

///
/// GlobalValue
///

template <>
struct CastInfo<::mlir::link::GlobalValue, ::mlir::Operation *>
    : public CastInfo<::mlir::GlobalValueLinkageOpInterface,
                      ::mlir::Operation *> {};

template <typename T>
struct CastInfo<T, ::mlir::link::GlobalValue>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::GlobalValue>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::GlobalValue>
    : public DenseMapInfo<::mlir::GlobalValueLinkageOpInterface> {};

///
/// GlobalVariable
///

template <>
struct CastInfo<::mlir::link::GlobalVariable, ::mlir::Operation *>
    : public CastInfo<::mlir::GlobalVariableLinkageOpInterface,
                      ::mlir::Operation *> {};

template <typename T>
struct CastInfo<T, ::mlir::link::GlobalVariable>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::GlobalVariable>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::GlobalVariable>
    : public DenseMapInfo<::mlir::GlobalVariableLinkageOpInterface> {};

///
/// Function
///

template <>
struct CastInfo<::mlir::link::Function, ::mlir::Operation *>
    : public CastInfo<::mlir::FunctionLinkageOpInterface, ::mlir::Operation *> {
};

template <typename T>
struct CastInfo<T, ::mlir::link::Function>
    : public CastInfo<T, ::mlir::link::LinkableOp> {};

template <typename T>
struct CastInfo<T, const ::mlir::link::Function>
    : public CastInfo<T, const ::mlir::link::LinkableOp> {};

template <>
struct DenseMapInfo<::mlir::link::Function>
    : public DenseMapInfo<::mlir::FunctionLinkageOpInterface> {};

} // namespace llvm

#endif // MLIR_LINKER_LINKAGEDIALECTINTERFACE_H
