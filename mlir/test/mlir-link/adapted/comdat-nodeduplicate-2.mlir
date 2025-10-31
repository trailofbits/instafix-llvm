// RUN: mlir-link %s -split-input-file | FileCheck %s

// CHECK-DAG: llvm.mlir.global private @foo.0(0 : i64)
// CHECK-DAG: llvm.mlir.global private @bar.0(0 : i64)
// CHECK-DAG: llvm.mlir.global external hidden @foo(2 : i64)
// CHECK-DAG: llvm.mlir.global external @bar(3 : i64)
// CHECK-DAG: llvm.mlir.global weak_odr @qux(4 : i64)
// CHECK-DAG: llvm.mlir.global linkonce @fred(5 : i64)

llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @foo nodeduplicate
}
llvm.mlir.global external @foo(2 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32, alignment = 8 : i64, section = "data"} : i64
llvm.mlir.global weak @bar(0 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32, section = "cnts"} : i64
llvm.mlir.global weak_odr @qux(4 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64

// -----

llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @foo nodeduplicate
}
llvm.mlir.global weak hidden @foo(0 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32, dso_local, section = "data"} : i64
llvm.mlir.global external @bar(3 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32, alignment = 16 : i64, dso_local, section = "cnts"} : i64
llvm.mlir.global linkonce @fred(5 : i64) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i64
