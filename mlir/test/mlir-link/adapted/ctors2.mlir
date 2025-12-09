// RUN: mlir-link %s %S/Inputs/ctors2.mlir | FileCheck %s

// CHECK: llvm.mlir.global_ctors ctors = [], priorities = [], data = []
// CHECK: llvm.mlir.global external @foo(0 : i8) comdat

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo any
  }
  llvm.mlir.global external @foo(0 : i8) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i8
}
