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
  llvm.mlir.global_ctors ctors = [@bar], priorities = [65535 : i32], data = [@foo]
  llvm.func internal @bar() comdat(@__llvm_global_comdat::@foo) attributes {dso_local} {
    llvm.return
  }
}
