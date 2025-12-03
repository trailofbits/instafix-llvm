// RUN: mlir-link -sort-symbols %s %S/Inputs/ctors.mlir | FileCheck --check-prefix=ALL --check-prefix=CHECK1 %s
// RUN: mlir-link -sort-symbols %S/Inputs/ctors.mlir %s | FileCheck --check-prefix=ALL --check-prefix=CHECK2 %s

// ALL: llvm.mlir.global_ctors ctors = [@f], priorities = [65535 : i32], data = [@v]
// CHECK1: llvm.mlir.global weak @v(0 : i8)
// CHECK2: llvm.mlir.global weak @v(1 : i8)
module {
  llvm.mlir.global weak @v(0 : i8) {addr_space = 0 : i32} : i8
  llvm.mlir.global_ctors ctors = [@f], priorities = [65535 : i32], data = [@v]
  llvm.func weak @f() {
    llvm.return
  }
}
