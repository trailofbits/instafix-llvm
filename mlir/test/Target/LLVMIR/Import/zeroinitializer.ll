; RUN: mlir-translate --import-llvm %s | FileCheck %s

%Domain = type { ptr, ptr }

; CHECK: llvm.mlir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: %[[RES:.+]] = llvm.mlir.zero : !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: llvm.return %[[RES]]
@D = global %Domain zeroinitializer
