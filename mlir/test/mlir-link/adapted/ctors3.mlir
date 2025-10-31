// RUN: mlir-link %s %S/Inputs/ctors3.mlir | FileCheck %s

// CHECK: llvm.mlir.global_ctors ctors = [], priorities = [], data = []
// CHECK llvm.mlir.global external @foo() comdat

module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo any
  }
  llvm.mlir.global external @foo() comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : !llvm.struct<"t", (i8)> {
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.undef : !llvm.struct<"t", (i8)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<"t", (i8)>
    llvm.return %2 : !llvm.struct<"t", (i8)>
  }
}
