// RUN: mlir-link -sort-symbols %s | FileCheck %s

// Test for single-module global constructor linking regression
// This test exercises the specific fix in LLVMLinkerInterface.h where
// appendGlobalStructors template handles single-module case by looking
// for "llvm.global_ctors" in the summary map rather than iterating 
// over an empty toLink ArrayRef.
//
// Before the fix, this would crash with ArrayRef bounds error.
// After the fix, it should correctly find and link the global constructor.

// This global constructor table tests the single-module case
// The linker should find this in the summary and properly link it
llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_exp], priorities = [65535 : i32], data = [#llvm.zero]

// CHECK: llvm.func internal @_GLOBAL__sub_I_exp()
llvm.func internal @_GLOBAL__sub_I_exp() {
  // Simulate global constructor initialization
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c42 = llvm.mlir.constant(42 : i32) : i32
  llvm.return
}

// CHECK: llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_exp], priorities = [65535 : i32], data = [#llvm.zero]

// Test function to make this a meaningful module
// CHECK: llvm.func @main()
llvm.func @main() {
  llvm.return
}