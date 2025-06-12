// RUN: mlir-link -sort-symbols %S/Inputs/2003-05-31-LinkerRename.mlir %s -o - | FileCheck %s

// CHECK:       llvm.mlir.global external @bar() {addr_space = 0 : i32} : !llvm.ptr {
// CHECK-NEXT:    %0 = llvm.mlir.addressof @foo.0 : !llvm.ptr
// CHECK-NEXT:    llvm.return %0 : !llvm.ptr
// CHECK-NEXT:  }

// CHECK: llvm.func @foo() -> i32

// CHECK:      llvm.func internal @foo.0() -> i32 attributes {dso_local} {
// CHECK-NEXT:   %0 = llvm.mlir.constant(7 : i32) : i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }

// CHECK:      llvm.func @test() -> i32 {
// CHECK-NEXT:   %0 = llvm.call @foo() : () -> i32
// CHECK-NEXT:   llvm.return %0 : i32
// CHECK-NEXT: }



module {
  llvm.func @foo() -> i32
  llvm.func @test() -> i32 {
    %0 = llvm.call @foo() : () -> i32
    llvm.return %0 : i32
  }
}
