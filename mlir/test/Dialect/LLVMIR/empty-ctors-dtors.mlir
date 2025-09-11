// RUN: mlir-opt %s | FileCheck %s

// Test that modules with empty constructor/destructor arrays can be parsed without crashing
// This tests the fix for ArrayRef bounds checking in LLVMLinkerInterface.h when linking
// modules with empty ctor/dtor arrays.

// CHECK-LABEL: llvm.mlir.global external @llvm.global_ctors()
llvm.mlir.global external @llvm.global_ctors() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<0 x struct<(i32, ptr, ptr)>>

// CHECK-LABEL: llvm.mlir.global external @llvm.global_dtors()  
llvm.mlir.global external @llvm.global_dtors() {addr_space = 0 : i32, alignment = 8 : i64} : !llvm.array<0 x struct<(i32, ptr, ptr)>>

// CHECK-LABEL: llvm.func @test()
llvm.func @test() {
  llvm.return
}