// RUN: mlir-opt %s -o - | FileCheck %s

module {
  llvm.func @foo(%arg0: i32) -> i32 attributes {dso_local} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: !llvm.struct<"struct.S", (i32)>
    %2 = llvm.alloca %0 x !llvm.struct<"struct.S", (i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.S", (i32)>
    llvm.store %arg0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.S", (i32)>
    %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %5 : i32
  }
}

module {
  llvm.func @getX(%arg0: !llvm.struct<"struct.S", (i32, i1, i32)>) -> i32 attributes {dso_local} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: !llvm.struct<"struct.S.1", (i32, i1, i32)>
    %2 = llvm.alloca %0 x !llvm.struct<"struct.S", (i32, i1, i32)> {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.S", (i32, i1, i32)>
    llvm.store %arg0, %3 {alignment = 4 : i64} : !llvm.struct<"struct.S", (i32, i1, i32)>, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[%1, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.S", (i32, i1, i32)>
    %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.return %5 : i32
  }
}
