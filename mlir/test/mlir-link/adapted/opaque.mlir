// RUN: mlir-link %s %S/Inputs/opaque.mlir | FileCheck %s
// CHECK-DAG: llvm.mlir.global external @g1() {{.*}} : !llvm.struct<"B", (struct<"C", (struct<"A", ()>)>, struct<"C", (struct<"A", ()>)>, ptr)>
// CHECK-DAG: llvm.mlir.global external @g2() {{.*}} : !llvm.struct<"A", ()>
// CHECK-DAG: llvm.mlir.global external @g3() {{.*}} : !llvm.struct<"B.1", (struct<"D", (struct<"E", opaque>)>, struct<"E", opaque>, ptr)>

module {
  llvm.mlir.global external @g1() {addr_space = 0 : i32} : !llvm.struct<"B", (struct<"C", (struct<"A", opaque>)>, struct<"C", (struct<"A", opaque>)>, ptr)>
  llvm.func @use_g1() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @g1 : !llvm.ptr
    llvm.return %0 : !llvm.ptr
  }
}
