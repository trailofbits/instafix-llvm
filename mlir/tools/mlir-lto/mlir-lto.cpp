//===-- mlir-lto: test harness for the resolution-based LTO interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lto/MlirLTOMain.h"

using namespace mlir;

int main(int argc, char **argv) {
    DialectRegistry registry;
    registerAllDialects(registry);
    registerAllExtensions(registry);

    return failed(MlirLTOMain(argc, argv, registry));
}
