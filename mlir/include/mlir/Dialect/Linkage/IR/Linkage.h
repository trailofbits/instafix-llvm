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

inline constexpr bool isExternalLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::ExternalLinkage;
}
inline constexpr bool isAvailableExternallyLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::AvailableExternallyLinkage;
}
inline constexpr bool isLinkOnceAnyLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::LinkOnceAnyLinkage;
}
inline constexpr bool isLinkOnceODRLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::LinkOnceODRLinkage;
}
inline constexpr bool isLinkOnceLinkage(LinkageKind linkage) {
  return isLinkOnceAnyLinkage(linkage) || isLinkOnceODRLinkage(linkage);
}
inline constexpr bool isWeakAnyLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::WeakAnyLinkage;
}
inline constexpr bool isWeakODRLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::WeakODRLinkage;
}
inline constexpr bool isWeakLinkage(LinkageKind linkage) {
  return isWeakAnyLinkage(linkage) || isWeakODRLinkage(linkage);
}
inline constexpr bool isInternalLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::InternalLinkage;
}
inline constexpr bool isPrivateLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::PrivateLinkage;
}
inline constexpr bool isLocalLinkage(LinkageKind linkage) {
  return isInternalLinkage(linkage) || isPrivateLinkage(linkage);
}
inline constexpr bool isExternalWeakLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::ExternalWeakLinkage;
}
inline constexpr bool isCommonLinkage(LinkageKind linkage) {
  return linkage == LinkageKind::CommonLinkage;
}
inline constexpr bool isValidDeclarationLinkage(LinkageKind linkage) {
  return isExternalWeakLinkage(linkage) || isExternalLinkage(linkage);
}

/// Whether the definition of this global may be replaced by something
/// non-equivalent at link time. For example, if a function has weak linkage
/// then the code defining it may be replaced by different code.
inline constexpr bool isInterposableLinkage(LinkageKind linkage) {
  switch (linkage) {
  case LinkageKind::WeakAnyLinkage:
  case LinkageKind::LinkOnceAnyLinkage:
  case LinkageKind::CommonLinkage:
  case LinkageKind::ExternalWeakLinkage:
    return true;

  case LinkageKind::AvailableExternallyLinkage:
  case LinkageKind::LinkOnceODRLinkage:
  case LinkageKind::WeakODRLinkage:
    // The above three cannot be overridden but can be de-refined.

  case LinkageKind::ExternalLinkage:
  case LinkageKind::AppendingLinkage:
  case LinkageKind::InternalLinkage:
  case LinkageKind::PrivateLinkage:
    return false;
  }
  llvm_unreachable("Fully covered switch above!");
}

/// Whether the definition of this global may be discarded if it is not used
/// in its compilation unit.
inline constexpr bool isDiscardableIfUnused(LinkageKind linkage) {
  return isLinkOnceLinkage(linkage) || isLocalLinkage(linkage) ||
         isAvailableExternallyLinkage(linkage);
}

/// Whether the definition of this global may be replaced at link time.  NB:
/// Using this method outside of the code generators is almost always a
/// mistake: when working at the IR level use isInterposable instead as it
/// knows about ODR semantics.
inline constexpr bool isWeakForLinker(LinkageKind linkage) {
  return linkage == LinkageKind::WeakAnyLinkage ||
         linkage == LinkageKind::WeakODRLinkage ||
         linkage == LinkageKind::LinkOnceAnyLinkage ||
         linkage == LinkageKind::LinkOnceODRLinkage ||
         linkage == LinkageKind::CommonLinkage ||
         linkage == LinkageKind::ExternalWeakLinkage;
}

} // namespace mlir::linkage

#endif // MLIR_DIALECT_LINKAGE_IR_LINKAGE_H
