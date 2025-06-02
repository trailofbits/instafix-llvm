// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global common @common_arr{{.*}}!llvm.array<18 x i32>

llvm.mlir.global common @common_arr(dense<0> : tensor<17xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<17 x i32>

// -----

llvm.mlir.global common local_unnamed_addr @common_arr(dense<0> : tensor<18xi32>) {addr_space = 0 : i32, alignment = 16 : i64, dso_local} : !llvm.array<18 x i32>
