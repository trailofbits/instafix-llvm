// RUN: mlir-link %s | FileCheck %s

// Test for multi-module global constructor linking
// This test exercises both the single-module and multi-module code paths
// in appendGlobalStructors template. The first module tests single-module,
// and when combined with the second, tests the multi-module path.
//
// This ensures our fix doesn't break the existing multi-module functionality
// while fixing the single-module case.

// Check that all the structors are present
// CHECK-DAG: llvm.func internal @_GLOBAL__sub_I_mod1()
// CHECK-DAG: llvm.func internal @_GLOBAL__sub_I_mod2()
// CHECK-DAG: llvm.func internal @_GLOBAL__sub_D_mod2()

// Check global_{c,d}tors ops have correct lists
// CHECK-DAG: llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod1, @_GLOBAL__sub_I_mod2], priorities = [65535 : i32, 65534 : i32], data = [#llvm.zero, #llvm.zero]
// CHECK-DAG: llvm.mlir.global_dtors dtors = [@_GLOBAL__sub_D_mod2], priorities = [65535 : i32], data = [#llvm.zero]

// Check regular functions
// CHECK-DAG: llvm.func @func_mod1
// CHECK-DAG: llvm.func @func_mod2

module @first_module {
  // Constructor function in first module
  llvm.func internal @_GLOBAL__sub_I_mod1() {
    %c100 = llvm.mlir.constant(100 : i32) : i32
    llvm.return
  }

  // Global constructor table for first module
  llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod1], priorities = [65535 : i32], data = [#llvm.zero]

  // Regular function
  llvm.func @func_mod1() {
    llvm.return
  }
}

// -----

module @second_module {
  // Constructor function in second module
  llvm.func internal @_GLOBAL__sub_I_mod2() {
    %c200 = llvm.mlir.constant(200 : i32) : i32
    llvm.return
  }

  // Destructor function for variety
  llvm.func internal @_GLOBAL__sub_D_mod2() {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    llvm.return
  }

  // Global constructor and destructor tables for second module
  // These should be properly merged with the first module's constructors
  llvm.mlir.global_ctors ctors = [@_GLOBAL__sub_I_mod2], priorities = [65534 : i32], data = [#llvm.zero]

  llvm.mlir.global_dtors dtors = [@_GLOBAL__sub_D_mod2], priorities = [65535 : i32], data = [#llvm.zero]

  // Regular function
  llvm.func @func_mod2() {
    llvm.return
  }
}
