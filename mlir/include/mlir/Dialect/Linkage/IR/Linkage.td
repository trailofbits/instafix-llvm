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

#endif // LINKAGE
