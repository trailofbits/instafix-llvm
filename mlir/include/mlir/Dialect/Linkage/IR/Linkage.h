//===- Linkage.h - Linkage Dialect ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINKAGE_IR_LINKAGE_H
#define MLIR_DIALECT_LINKAGE_IR_LINKAGE_H

#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/Linkage/IR/Linkage.h.inc"
#include "mlir/Dialect/Linkage/IR/LinkageEnums.h.inc"

namespace mlir::linkage {

inline constexpr bool isExternalLinkage(Linkage linkage) {
  return linkage == Linkage::External;
}
inline constexpr bool isAvailableExternallyLinkage(Linkage linkage) {
  return linkage == Linkage::AvailableExternally;
}
inline constexpr bool isLinkOnceAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Linkonce;
}
inline constexpr bool isLinkOnceODRLinkage(Linkage linkage) {
  return linkage == Linkage::LinkonceODR;
}
inline constexpr bool isLinkOnceLinkage(Linkage linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}
inline constexpr bool isWeakAnyLinkage(Linkage linkage) {
  return linkage == Linkage::Weak;
}
inline constexpr bool isWeakODRLinkage(Linkage linkage) {
  return linkage == Linkage::WeakODR;
}
inline constexpr bool isWeakLinkage(Linkage linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}
inline constexpr bool isInternalLinkage(Linkage linkage) {
  return linkage == Linkage::Internal;
}
inline constexpr bool isPrivateLinkage(Linkage linkage) {
  return linkage == Linkage::Private;
}
inline constexpr bool isLocalLinkage(Linkage linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}
inline constexpr bool isExternalWeakLinkage(Linkage linkage) {
  return linkage == Linkage::ExternWeak;
}
inline constexpr bool isCommonLinkage(Linkage linkage) {
  return linkage == Linkage::Common;
}
inline constexpr bool isValidDeclarationLinkage(Linkage linkage) {
  return isExternalWeakLinkage(linkage) || isExternalLinkage(linkage);
}

/// Whether the definition of this global may be replaced by something
/// non-equivalent at link time. For example, if a function has weak linkage
/// then the code defining it may be replaced by different code.
inline constexpr bool isInterposableLinkage(Linkage linkage) {
  switch (linkage) {
  case Linkage::Weak:
  case Linkage::Linkonce:
  case Linkage::Common:
  case Linkage::ExternWeak:
    return true;

  case Linkage::AvailableExternally:
  case Linkage::LinkonceODR:
  case Linkage::WeakODR:
    // The above three cannot be overridden but can be de-refined.

  case Linkage::External:
  case Linkage::Appending:
  case Linkage::Internal:
  case Linkage::Private:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

/// Whether the definition of this global may be discarded if it is not used
/// in its compilation unit.
inline constexpr bool isDiscardableIfUnused(Linkage linkage) {
  return isLinkOnceLinkage(linkage) || isLocalLinkage(linkage) ||
         isAvailableExternallyLinkage(linkage);
}

/// Whether the definition of this global may be replaced at link time.  NB:
/// Using this method outside of the code generators is almost always a
/// mistake: when working at the IR level use isInterposable instead as it
/// knows about ODR semantics.
inline constexpr bool isWeakForLinker(Linkage linkage) {
  return linkage == Linkage::Weak ||
         linkage == Linkage::WeakODR ||
         linkage == Linkage::Linkonce ||
         linkage == Linkage::LinkonceODR ||
         linkage == Linkage::Common ||
         linkage == Linkage::ExternWeak;
}

} // namespace mlir::linkage

#endif // MLIR_DIALECT_LINKAGE_IR_LINKAGE_H
