// RUN: mlir-link %s | FileCheck %s

// CHECK: llvm.mlir.global_ctors ctors = [@f], priorities = [65535 : i32], data = [@v]

module {
  llvm.mlir.global linkonce @v(42 : i8) {addr_space = 0 : i32} : i8
  llvm.mlir.global_ctors ctors = [@f], priorities = [65535 : i32], data = [@v]
  llvm.func @f() {
    llvm.return
  }
}
