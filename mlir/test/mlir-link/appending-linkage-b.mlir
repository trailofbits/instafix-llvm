// RUN: mlir-link -split-input-file %s | FileCheck %s

// CHECK: appending @X(dense<[7, 4]> : tensor<2xi32>){{.*}} : !llvm.array<2 x i32>

llvm.mlir.global appending @X(dense<[7, 4]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>

