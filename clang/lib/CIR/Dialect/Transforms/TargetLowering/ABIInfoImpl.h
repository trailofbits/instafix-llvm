//===- ABIInfoImpl.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/ABIInfoImpl.h. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H
#define LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H

#include "ABIInfo.h"
#include "CIRCXXABI.h"
#include "LowerFunctionInfo.h"

namespace cir {

bool classifyReturnType(const CIRCXXABI &CXXABI, LowerFunctionInfo &FI,
                        const ABIInfo &Info);

bool isAggregateTypeForABI(mlir::Type T);

mlir::Value emitRoundPointerUpToAlignment(cir::CIRBaseBuilderTy &builder,
                                          mlir::Value ptr, unsigned alignment);

/// Pass transparent unions as if they were the type of the first element. Sema
/// should ensure that all elements of the union have the same "machine type".
mlir::Type useFirstFieldIfTransparentUnion(mlir::Type Ty);

CIRCXXABI::RecordArgABI getRecordArgABI(const RecordType RT, CIRCXXABI &CXXABI);
CIRCXXABI::RecordArgABI getRecordArgABI(mlir::Type ty, CIRCXXABI &CXXABI);

} // namespace cir

#endif // LLVM_CLANG_LIB_CIR_DIALECT_TRANSFORMS_TARGETLOWERING_ABIINFOIMPL_H
