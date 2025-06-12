module {
  llvm.mlir.global external @g2() {addr_space = 0 : i32} : !llvm.struct<"A", ()>
  llvm.mlir.global external @g3() {addr_space = 0 : i32} : !llvm.struct<"B", (struct<"D", (struct<"E", opaque>)>, struct<"E", opaque>, ptr)>
  llvm.func @f1() {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.getelementptr %0[%1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"A", ()>
    llvm.return
  }
  llvm.func @use_g2() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @g2 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
  llvm.func @use_g3() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @g3 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
