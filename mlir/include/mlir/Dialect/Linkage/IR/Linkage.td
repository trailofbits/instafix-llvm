//===- Linkage.td - Linkage dialect ------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINKAGE
#define LINKAGE

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Linkage dialect definition.
//===----------------------------------------------------------------------===//

def Linkage_Dialect : Dialect {
  let name = "linkage";

  let summary = "The Linkage dialect";
  let description = [{
    TBD
  }];

  let cppNamespace = "::mlir::linkage";

  let useDefaultAttributePrinterParser = 1;
}


// Linkage types: mirrors linkage types from llvm/IR/GlobalValue.h.
// Used for forwarding linkage types during LLVM lowering and for MLIR linker.

// Externally visible function
def Linkage_ExternalLinkage :
  I32EnumAttrCase<"ExternalLinkage", 0, "external">;
// Available for inspection, not emission.
def Linkage_AvailableExternallyLinkage :
  I32EnumAttrCase<"AvailableExternallyLinkage", 1, "available_externally">;
// Keep one copy of function when linking (inline)
def Linkage_LinkOnceAnyLinkage :
  I32EnumAttrCase<"LinkOnceAnyLinkage", 2, "linkonce">;
// Same, but only replaced by something equivalent.
def Linkage_LinkOnceODRLinkage :
  I32EnumAttrCase<"LinkOnceODRLinkage", 3, "linkonce_odr">;
// Keep one copy of named function when linking (weak)
def Linkage_WeakAnyLinkage :
  I32EnumAttrCase<"WeakAnyLinkage", 4, "weak">;
// Same, but only replaced by something equivalent.
def Linkage_WeakODRLinkage :
  I32EnumAttrCase<"WeakODRLinkage", 5, "weak_odr">;
// Special purpose, only applies to global arrays
def Linkage_AppendingLinkage :
  I32EnumAttrCase<"AppendingLinkage", 6, "appending">;
// Rename collisions when linking (static functions).
def Linkage_InternalLinkage :
  I32EnumAttrCase<"InternalLinkage", 7, "internal">;
// Like Internal, but omit from symbol table, prefix it with
// "linkage_" to prevent clash with MLIR's symbol "private".
def Linkage_PrivateLinkage :
  I32EnumAttrCase<"PrivateLinkage", 8, "linkage_private">;
// ExternalWeak linkage description.
def Linkage_ExternalWeakLinkage :
  I32EnumAttrCase<"ExternalWeakLinkage", 9, "extern_weak">;
// Tentative definitions.
def Linkage_CommonLinkage :
  I32EnumAttrCase<"CommonLinkage", 10, "common">;

/// An enumeration for the kinds of linkage for global values.
def Linkage_LinkageKind : I32EnumAttr<
    "LinkageKind",
    "Linkage type/kind",
    [Linkage_ExternalLinkage, Linkage_AvailableExternallyLinkage,
     Linkage_LinkOnceAnyLinkage, Linkage_LinkOnceODRLinkage,
     Linkage_WeakAnyLinkage, Linkage_WeakODRLinkage,
     Linkage_AppendingLinkage,
     Linkage_InternalLinkage, Linkage_PrivateLinkage,
     Linkage_ExternalWeakLinkage, Linkage_CommonLinkage
     ]> {
  let cppNamespace = "::mlir::linkage";
}

#endif // LINKAGE
