; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; Test basic llvm.dbg.assign intrinsic import
; CHECK-LABEL: @test_dbg_assign_basic
define void @test_dbg_assign_basic() !dbg !6 {
entry:
  %var = alloca i32, align 4
  %value = add i32 1, 2

  ; Store with assignment tracking
  store i32 %value, ptr %var, !DIAssignID !8

  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : i32, !llvm.ptr
  call void @llvm.dbg.assign(metadata i32 %value, metadata !7, metadata !DIExpression(), metadata !8, metadata ptr %var, metadata !DIExpression()), !dbg !9

  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #0

; Test store instruction with DIAssignID metadata
; CHECK-LABEL: @test_store_with_assignid
define void @test_store_with_assignid() !dbg !10 {
entry:
  %var = alloca i32, align 4
  %value = add i32 3, 4

  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : i32, !llvm.ptr
  store i32 %value, ptr %var, !DIAssignID !11

  ret void
}

; Test multiple assignments with same ID (assignment tracking relationship)
; CHECK-LABEL: @test_multiple_assignments_same_id
define void @test_multiple_assignments_same_id() !dbg !12 {
entry:
  %var = alloca i32, align 4
  %value1 = add i32 5, 6
  %value2 = add i32 7, 8

  ; First dbg.assign and store with same ID
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : i32, !llvm.ptr
  call void @llvm.dbg.assign(metadata i32 %value1, metadata !14, metadata !DIExpression(), metadata !13, metadata ptr %var, metadata !DIExpression()), !dbg !15
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : i32, !llvm.ptr
  call void @llvm.dbg.assign(metadata i32 %value2, metadata !14, metadata !DIExpression(), metadata !13, metadata ptr %var, metadata !DIExpression()), !dbg !16

  ; Stores with assignment tracking (should reuse the same MLIR attribute)
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : i32, !llvm.ptr
  store i32 %value1, ptr %var, !DIAssignID !13
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : i32, !llvm.ptr
  store i32 %value2, ptr %var, !DIAssignID !13

  ret void
}

%struct.Point = type { i32, i32 }

; Test complex debug expressions in assignments
; CHECK-LABEL: @test_complex_expressions
define void @test_complex_expressions() !dbg !17 {
entry:
  %struct_var = alloca %struct.Point, align 4
  %field_ptr = getelementptr inbounds %struct.Point, ptr %struct_var, i32 0, i32 1
  %value = add i32 9, 10

  ; dbg.assign with complex location and address expressions
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : i32, !llvm.ptr
  call void @llvm.dbg.assign(metadata i32 %value, metadata !19, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !18, metadata ptr %struct_var, metadata !DIExpression(DW_OP_deref)), !dbg !20

  ; Store to struct field with assignment tracking
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : i32, !llvm.ptr
  store i32 %value, ptr %field_ptr, !DIAssignID !18

  ret void
}

; Test assignment tracking with different variable types
; CHECK-LABEL: @test_different_types
define void @test_different_types() !dbg !21 {
entry:
  %int_var = alloca i32, align 4
  %float_var = alloca float, align 4
  %ptr_var = alloca ptr, align 8

  ; Integer assignment
  %int_val = add i32 11, 12
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : i32, !llvm.ptr
  call void @llvm.dbg.assign(metadata i32 %int_val, metadata !23, metadata !DIExpression(), metadata !22, metadata ptr %int_var, metadata !DIExpression()), !dbg !24
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : i32, !llvm.ptr
  store i32 %int_val, ptr %int_var, !DIAssignID !22

  ; Float assignment
  %float_val = fadd float 1.0, 2.0
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : f32, !llvm.ptr
  call void @llvm.dbg.assign(metadata float %float_val, metadata !26, metadata !DIExpression(), metadata !25, metadata ptr %float_var, metadata !DIExpression()), !dbg !27
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : f32, !llvm.ptr
  store float %float_val, ptr %float_var, !DIAssignID !25

  ; Pointer assignment
  %ptr_val = getelementptr i32, ptr %int_var, i32 0
  ; CHECK: llvm.intr.dbg.assign #{{.*}} #{{.*}} = %{{.*}}, %{{.*}} : !llvm.ptr, !llvm.ptr
  call void @llvm.dbg.assign(metadata ptr %ptr_val, metadata !29, metadata !DIExpression(), metadata !28, metadata ptr %ptr_var, metadata !DIExpression()), !dbg !30
  ; CHECK: llvm.store %{{.*}}, %{{.*}} {DIAssignID = #{{.*}}} : !llvm.ptr, !llvm.ptr
  store ptr %ptr_val, ptr %ptr_var, !DIAssignID !28

  ret void
}



attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "debug-assign.c", directory: "/test")
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)

; Function: test_dbg_assign_basic
!6 = distinct !DISubprogram(name: "test_dbg_assign_basic", scope: !2, file: !2, line: 5, type: !31, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocalVariable(name: "var", scope: !6, file: !2, line: 6, type: !3)
!8 = distinct !DIAssignID()
!9 = !DILocation(line: 7, column: 3, scope: !6)

; Function: test_store_with_assignid
!10 = distinct !DISubprogram(name: "test_store_with_assignid", scope: !2, file: !2, line: 10, type: !31, scopeLine: 10, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DIAssignID()

; Function: test_multiple_assignments_same_id
!12 = distinct !DISubprogram(name: "test_multiple_assignments_same_id", scope: !2, file: !2, line: 15, type: !31, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !0)
!13 = distinct !DIAssignID()
!14 = !DILocalVariable(name: "var", scope: !12, file: !2, line: 16, type: !3)
!15 = !DILocation(line: 17, column: 3, scope: !12)
!16 = !DILocation(line: 18, column: 3, scope: !12)

; Function: test_complex_expressions
!17 = distinct !DISubprogram(name: "test_complex_expressions", scope: !2, file: !2, line: 25, type: !31, scopeLine: 25, spFlags: DISPFlagDefinition, unit: !0)
!18 = distinct !DIAssignID()
!19 = !DILocalVariable(name: "struct_var", scope: !17, file: !2, line: 26, type: !32)
!20 = !DILocation(line: 27, column: 3, scope: !17)

; Function: test_different_types
!21 = distinct !DISubprogram(name: "test_different_types", scope: !2, file: !2, line: 35, type: !31, scopeLine: 35, spFlags: DISPFlagDefinition, unit: !0)
!22 = distinct !DIAssignID()
!23 = !DILocalVariable(name: "int_var", scope: !21, file: !2, line: 36, type: !3)
!24 = !DILocation(line: 37, column: 3, scope: !21)
!25 = distinct !DIAssignID()
!26 = !DILocalVariable(name: "float_var", scope: !21, file: !2, line: 38, type: !4)
!27 = !DILocation(line: 39, column: 3, scope: !21)
!28 = distinct !DIAssignID()
!29 = !DILocalVariable(name: "ptr_var", scope: !21, file: !2, line: 40, type: !5)
!30 = !DILocation(line: 41, column: 3, scope: !21)

; Types
!31 = !DISubroutineType(types: !{null})
!32 = !DICompositeType(tag: DW_TAG_structure_type, name: "Point", file: !2, line: 1, size: 64, elements: !33)
!33 = !{!34, !35}
!34 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !32, file: !2, line: 2, baseType: !3, size: 32)
!35 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !32, file: !2, line: 3, baseType: !3, size: 32, offset: 32)
