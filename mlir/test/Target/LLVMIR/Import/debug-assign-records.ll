; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Test debug record format (new debug info format with #dbg_assign records)
; This file uses the new debug record format which cannot be mixed with debug intrinsics

; CHECK-LABEL: @test_debug_record_format
define dso_local void @test_debug_record_format() local_unnamed_addr !dbg !10 {
  ; Test undef value generation
  ; CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i1
  ; Test alloca with DIAssignID metadata
  ; CHECK: %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i32 {DIAssignID = #[[ASSIGN_ID:llvm.di_assign_id<id = distinct\[[0-9]+\]<>>]], alignment = 4 : i64} : (i32) -> !llvm.ptr
  %1 = alloca i32, align 4, !DIAssignID !16

  ; Test dbg_assign record conversion (new debug record format)
  ; CHECK: llvm.intr.dbg.assign #[[VAR:.*]] #[[ASSIGN_ID]] = %[[UNDEF]], %[[ALLOCA]] : i1, !llvm.ptr
  #dbg_assign(i1 undef, !14, !DIExpression(), !16, ptr %1, !DIExpression(), !17)

  call void @func1(ptr noundef nonnull %1)
  ret void
}

declare void @func1(ptr noundef) local_unnamed_addr

; -----

; CHECK-LABEL: @test_debug_record_shared_ids
define void @test_debug_record_shared_ids() !dbg !20 {
  ; Test undef value generation
  ; CHECK: %[[UNDEF2:.*]] = llvm.mlir.undef : i1
  ; Two allocas with same assignment ID should get same MLIR attribute
  ; CHECK: %[[ALLOCA1:.*]] = llvm.alloca %{{.*}} x i32 {DIAssignID = #[[SHARED_ID:llvm.di_assign_id<id = distinct\[[0-9]+\]<>>]], alignment = 4 : i64} : (i32) -> !llvm.ptr
  %1 = alloca i32, align 4, !DIAssignID !26
  ; CHECK: %[[ALLOCA2:.*]] = llvm.alloca %{{.*}} x i32 {DIAssignID = #[[SHARED_ID]], alignment = 4 : i64} : (i32) -> !llvm.ptr  
  %2 = alloca i32, align 4, !DIAssignID !26

  ; Both dbg_assign records should reference the same assignment ID
  ; CHECK: llvm.intr.dbg.assign #[[VAR2:.*]] #[[SHARED_ID]] = %[[UNDEF2]], %[[ALLOCA1]] : i1, !llvm.ptr
  #dbg_assign(i1 undef, !24, !DIExpression(), !26, ptr %1, !DIExpression(), !27)
  ; CHECK: llvm.intr.dbg.assign #[[VAR2]] #[[SHARED_ID]] = %[[UNDEF2]], %[[ALLOCA2]] : i1, !llvm.ptr
  #dbg_assign(i1 undef, !24, !DIExpression(), !26, ptr %2, !DIExpression(), !27)

  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "test_debug_record_format", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !{null})
!14 = !DILocalVariable(name: "host", scope: !10, file: !1, line: 4, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = distinct !DIAssignID()
!17 = !DILocation(line: 4, column: 7, scope: !10)
!20 = distinct !DISubprogram(name: "test_debug_record_shared_ids", scope: !1, file: !1, line: 8, type: !11, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!24 = !DILocalVariable(name: "shared", scope: !20, file: !1, line: 9, type: !15)
!26 = distinct !DIAssignID()
!27 = !DILocation(line: 9, column: 7, scope: !20)
