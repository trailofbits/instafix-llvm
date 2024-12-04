//===- Linkage.td - Linkage dialect ------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LINKAGE_DIALECT
#define LINKAGE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/LLVMIR/Utils.td"

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

// Linkage attribute is used on functions and globals. The order follows that of
// llvm::GlobalValue::LinkageTypes from llvm/IR/GlobalValue.h. The names are
// equivalent to visible names in the IR rather than to enum values names in
// llvm::GlobalValue since the latter is easier to change.
def Linkage_LinkageExternal
    : LLVM_EnumAttrCase<"External", "external", "ExternalLinkage", 0>;
def Linkage_LinkageAvailableExternally
    : LLVM_EnumAttrCase<"AvailableExternally", "available_externally",
                        "AvailableExternallyLinkage", 1>;
def Linkage_LinkageLinkonce
    : LLVM_EnumAttrCase<"Linkonce", "linkonce", "LinkOnceAnyLinkage", 2>;
def Linkage_LinkageLinkonceODR
    : LLVM_EnumAttrCase<"LinkonceODR", "linkonce_odr", "LinkOnceODRLinkage", 3>;
def Linkage_LinkageWeak
    : LLVM_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage", 4>;
def Linkage_LinkageWeakODR
    : LLVM_EnumAttrCase<"WeakODR", "weak_odr", "WeakODRLinkage", 5>;
def Linkage_LinkageAppending
    : LLVM_EnumAttrCase<"Appending", "appending", "AppendingLinkage", 6>;
def Linkage_LinkageInternal
    : LLVM_EnumAttrCase<"Internal", "internal", "InternalLinkage", 7>;
def Linkage_LinkagePrivate
    : LLVM_EnumAttrCase<"Private", "private", "PrivateLinkage", 8>;
def Linkage_LinkageExternWeak
   : LLVM_EnumAttrCase<"ExternWeak", "extern_weak", "ExternalWeakLinkage", 9>;
def Linkage_LinkageCommon
    : LLVM_EnumAttrCase<"Common", "common", "CommonLinkage", 10>;

def Linkage_LinkageEnum : LLVM_EnumAttr<
    "Linkage",
    "::llvm::GlobalValue::LinkageTypes",
    "Linkage types",
    [Linkage_LinkageExternal, Linkage_LinkageAvailableExternally,
     Linkage_LinkageLinkonce, Linkage_LinkageLinkonceODR,
     Linkage_LinkageWeak, Linkage_LinkageWeakODR, Linkage_LinkageAppending,
     Linkage_LinkageInternal, Linkage_LinkagePrivate, Linkage_LinkageExternWeak,
     Linkage_LinkageCommon]> {
  let cppNamespace = "::mlir::linkage";
}

#endif // LINKAGE_DIALECT
