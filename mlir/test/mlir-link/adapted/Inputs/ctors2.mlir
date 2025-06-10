module {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @foo any
  }
  llvm.mlir.global external @foo(1 : i8) comdat(@__llvm_global_comdat::@foo) {addr_space = 0 : i32} : i8
  llvm.mlir.global_ctors ctors = [@bar], priorities = [65535 : i32], data = [@foo]
  llvm.func @bar() comdat(@__llvm_global_comdat::@foo) {
    llvm.return
  }
}
