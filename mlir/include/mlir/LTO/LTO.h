//===- LTO.h - MLIR Link Time Optimizer -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions and classes used to support MLIR LTO. It is
// intended to be used both by LTO classes as well as by clients (gold-plugin)
// that don't utilize the LTO code generator interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LTO_LTO_H
#define MLIR_LTO_LTO_H

namespace mlir {
namespace lto {

class LTO {};

} // namespace lto
} // namespace mlir

#endif // MLIR_LTO_LTO_H
