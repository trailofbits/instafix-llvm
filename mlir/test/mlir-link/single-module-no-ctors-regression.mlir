// RUN: mlir-link %s | FileCheck %s

// Test for single-module case when NO global constructors are present
// This test exercises the appendGlobalStructors template when the summary
// map has no "llvm.global_ctors" entry. The linker should handle this
// gracefully without attempting to access non-existent summary entries.
//
// Before the fix, this could potentially cause issues when the template
// tries to find constructors that don't exist.
// After the fix, it should handle the missing entry case cleanly.

// Another function to make this meaningful
// CHECK: llvm.func @main()
llvm.func @main() {
  llvm.call @regular_function() : () -> ()
  llvm.return
}

// Regular function (no constructor attributes)
// CHECK: llvm.func @regular_function()
llvm.func @regular_function() {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.return
}

// Verify no global constructors are present in the output
// The linker should not crash when no ctors are found
// CHECK-NOT: llvm.mlir.global_ctors