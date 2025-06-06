// RUN: mlir-link -sort-symbols -split-input-file %s | FileCheck %s

// CHECK: llvm.mlir.global appending @llvm.used() {{.*}} : !llvm.array<4 x ptr>
// CHECK: [[V1:%[0-9]+]] = llvm.mlir.addressof @V1
// CHECK: [[V2:%[0-9]+]] = llvm.mlir.addressof @V2
// CHECK: [[V3:%[0-9]+]] = llvm.mlir.addressof @V3
// CHECK: [[V4:%[0-9]+]] = llvm.mlir.addressof @V4
// CHECK: [[UNDEF:%[0-9]+]] = llvm.mlir.undef : !llvm.array<4 x ptr>
// CHECK: [[INIT_1:%[0-9]+]] = llvm.insertvalue [[V1]], [[UNDEF]][1] : !llvm.array<4 x ptr>
// CHECK: [[INIT_2:%[0-9]+]] = llvm.insertvalue [[V2]], [[INIT_1]][0] : !llvm.array<4 x ptr>
// CHECK: [[INIT_3:%[0-9]+]] = llvm.insertvalue [[V3]], [[INIT_2]][2] : !llvm.array<4 x ptr>
// CHECK: [[INIT_4:%[0-9]+]] = llvm.insertvalue [[V4]], [[INIT_3]][3] : !llvm.array<4 x ptr>
// CHECK: llvm.return [[INIT_4]] : !llvm.array<4 x ptr>

llvm.mlir.global @V1(32.0 : f32) : f32
llvm.mlir.global @V2(33.0 : f32) : f32

llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<2 x ptr> {
  %0 = llvm.mlir.addressof @V1 : !llvm.ptr
  %1 = llvm.mlir.addressof @V2 : !llvm.ptr
  %2 = llvm.mlir.undef : !llvm.array<2 x ptr>
  %3 = llvm.insertvalue %1, %2[0] : !llvm.array<2 x ptr>
  %4 = llvm.insertvalue %0, %3[1] : !llvm.array<2 x ptr>
  llvm.return %4 : !llvm.array<2 x ptr>
}

// -----

llvm.mlir.global @V3(34.0 : f32) : f32

llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
  %0 = llvm.mlir.addressof @V3 : !llvm.ptr
  %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr>
  llvm.return %2 : !llvm.array<1 x ptr>
}

// -----

llvm.mlir.global @V4(34.0 : f32) : f32
llvm.mlir.global appending @llvm.used() {addr_space = 0 : i32, section = "llvm.metadata"} : !llvm.array<1 x ptr> {
  %0 = llvm.mlir.addressof @V4 : !llvm.ptr
  %1 = llvm.mlir.undef : !llvm.array<1 x ptr>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.array<1 x ptr>
  llvm.return %2 : !llvm.array<1 x ptr>
}
