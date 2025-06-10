// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global weak @v(0 : i8)

llvm.mlir.global weak @v(0 : i8) {addr_space = 0 : i32} : i8

llvm.func @use_v1() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @v : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

llvm.mlir.global weak @v(1 : i8) {addr_space = 0 : i32} : i8

llvm.func @use_v2() -> !llvm.ptr {
  %0 = llvm.mlir.addressof @v : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
