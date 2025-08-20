// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.func internal @foo
// CHECK: llvm.mlir.global_ctors ctors = [@foo]

llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo any
}

llvm.mlir.global_ctors ctors = [@foo], priorities = [65535 : i32], data = [#llvm.zero]
llvm.func internal @foo() -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}

// -----

llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @bar any
}
