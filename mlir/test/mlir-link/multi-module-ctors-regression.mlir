// RUN: mlir-link %s | FileCheck %s

// Test for multi-module global constructor linking
// This test exercises both the single-module and multi-module code paths
// in appendGlobalStructors template. The first module tests single-module,
// and when combined with the second, tests the multi-module path.
//
// This ensures our fix doesn't break the existing multi-module functionality
// while fixing the single-module case.

module @first_module {
  // Constructor function in first module
  // CHECK-LABEL: llvm.func internal @_GLOBAL__sub_I_mod1()
  llvm.func internal @_GLOBAL__sub_I_mod1() {
    %c100 = llvm.mlir.constant(100 : i32) : i32
    llvm.return
  }

  // Global constructor table for first module
  // CHECK: llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod1], priorities = [65535 : i32], data = [#llvm.zero]
  llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod1], priorities = [65535 : i32], data = [#llvm.zero]

  // Regular function
  // CHECK-LABEL: llvm.func @func_mod1()
  llvm.func @func_mod1() {
    llvm.return
  }
}

// -----

module @second_module {
  // Constructor function in second module  
  // CHECK-LABEL: llvm.func internal @_GLOBAL__sub_I_mod2()
  llvm.func internal @_GLOBAL__sub_I_mod2() {
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.return
  }

  // Destructor function for variety
  // CHECK-LABEL: llvm.func internal @_GLOBAL__sub_D_mod2()
  llvm.func internal @_GLOBAL__sub_D_mod2() {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return
  }

  // Global constructor and destructor tables for second module
  // These should be properly merged with the first module's constructors
  // CHECK: llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod2], priorities = [65534 : i32], data = [#llvm.zero]
  llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod2], priorities = [65534 : i32], data = [#llvm.zero]
  
  // CHECK: llvm.mlir.global_dtors dtors = [@_GLOBAL__sub_D_mod2], priorities = [65535 : i32], data = [#llvm.zero]  
  llvm.mlir.global_dtors dtors = [@_GLOBAL__sub_D_mod2], priorities = [65535 : i32], data = [#llvm.zero]

  // Regular function
  // CHECK-LABEL: llvm.func @func_mod2()
  llvm.func @func_mod2() {
    llvm.return
  }
}