// RUN: mlir-link -split-input-file %s | FileCheck %s

// Test that when two modules define the same function with different signatures,
// the linker converts direct calls to indirect calls to handle the mismatch.
// This mimics the scenario where foo.c defines: int foo(int x) { return x; }
// and main.c declares: int foo(void); and calls foo();

// CHECK: llvm.func @main() -> i16 {
// CHECK-NEXT:   %[[ADDR:.+]] = llvm.mlir.addressof @foo : !llvm.ptr
// CHECK-NEXT:   %[[RESULT:.+]] = llvm.call %[[ADDR]]() : !llvm.ptr, () -> i16
// CHECK-NEXT:   llvm.return %[[RESULT]] : i16
// CHECK-NEXT: }
// CHECK: llvm.func @foo(%arg0: i32) -> i32

// Module 1: defines foo(i32) -> i32
llvm.func @foo(%arg0: i32) -> i32 {
  llvm.return %arg0 : i32
}

// -----

// Module 2: declares foo() -> i32 (no parameters) and calls it
llvm.func @foo() -> i16

llvm.func @main() -> i16 {
  %result = llvm.call @foo() : () -> i16
  llvm.return %result : i16
}
