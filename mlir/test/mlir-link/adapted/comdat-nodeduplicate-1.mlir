// RUN: not mlir-link -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: Linker error

llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @foo nodeduplicate
}
llvm.mlir.global external @foo(43 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64

// -----

llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @foo nodeduplicate
}
llvm.mlir.global external @foo(43 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64
