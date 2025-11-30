// RUN: mlir-link -split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Cross-TU external declaration type mismatch (reverse order)
//===----------------------------------------------------------------------===//
//
// Same as external-decl-type-mismatch.mlir but with modules in reverse order.
// This verifies that the larger type wins regardless of linking order.
//===----------------------------------------------------------------------===//

// The call site expecting i8 should call the i32 version with truncation
// CHECK-LABEL: llvm.func @caller_expecting_i8() -> i8
// CHECK:         %[[CALL:.*]] = llvm.call @getpid() : () -> i32
// CHECK-NEXT:    %[[TRUNC:.*]] = llvm.trunc %[[CALL]] : i32 to i8
// CHECK-NEXT:    llvm.return %[[TRUNC]] : i8

// The canonical declaration should have the larger return type (i32)
// CHECK: llvm.func @getpid() -> i32

//===----------------------------------------------------------------------===//
// Module 1: Correct declaration (i32 return type) - linked first
//===----------------------------------------------------------------------===//

llvm.func @getpid() -> i32

// -----

//===----------------------------------------------------------------------===//
// Module 2: Incorrect declaration (i8 return type) with call site
//===----------------------------------------------------------------------===//

llvm.func @getpid() -> i8

llvm.func @caller_expecting_i8() -> i8 {
  %0 = llvm.call @getpid() : () -> i8
  llvm.return %0 : i8
}
