// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK: appending @X(dense<[7, 4, 8, 9]> : tensor<4xi32>){{.*}} : !llvm.array<4 x i32>

llvm.mlir.global appending @X(dense<[7, 4]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>
llvm.mlir.global external @Y() {addr_space = 0 : i32} : !llvm.ptr {
  %0 = llvm.mlir.addressof @X : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}
llvm.func @foo(%arg0: i64) {
  %0 = llvm.mlir.addressof @X : !llvm.ptr
  %1 = llvm.mlir.constant(0 : i64) : i64
  %2 = llvm.getelementptr %0[%1, %arg0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<2 x i32>
  llvm.return
}

// -----

llvm.mlir.global appending @X(dense<8> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>

// -----

llvm.mlir.global appending @X(dense<9> : tensor<1xi32>) {addr_space = 0 : i32} : !llvm.array<1 x i32>
