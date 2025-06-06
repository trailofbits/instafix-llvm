; RUN: not mlir-translate -import-llvm -emit-expensive-warnings -split-input-file %s 2>&1 -o /dev/null | FileCheck %s

; CHECK:      <unknown>
; CHECK-SAME: error: unhandled instruction: indirectbr ptr %dst, [label %bb1, label %bb2]
define i32 @unhandled_instruction(ptr %dst) {
  indirectbr ptr %dst, [label %bb1, label %bb2]
bb1:
  ret i32 0
bb2:
  ret i32 1
}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: unhandled constant: ptr blockaddress(@unhandled_constant, %bb1) since blockaddress(...) is unsupported
; CHECK:      <unknown>
; CHECK-SAME: error: unhandled instruction: ret ptr blockaddress(@unhandled_constant, %bb1)
define ptr @unhandled_constant() {
  br label %bb1
bb1:
  ret ptr blockaddress(@unhandled_constant, %bb1)
}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: unhandled constant: ptr blockaddress(@unhandled_global, %bb1) since blockaddress(...) is unsupported
; CHECK:      <unknown>
; CHECK-SAME: error: unhandled global variable: @private = private global ptr blockaddress(@unhandled_global, %bb1)
@private = private global ptr blockaddress(@unhandled_global, %bb1)

define void @unhandled_global() {
  br label %bb1
bb1:
  ret void
}

; // -----

; Check that debug intrinsics with an unsupported argument are dropped.

declare void @llvm.dbg.value(metadata, metadata, metadata)

; CHECK:      import-failure.ll
; CHECK-SAME: warning: dropped intrinsic: tail call void @llvm.dbg.value(metadata !DIArgList(i64 %{{.*}}, i64 undef), metadata !3, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_stack_value))
; CHECK:      import-failure.ll
; CHECK-SAME: warning: dropped intrinsic: tail call void @llvm.dbg.value(metadata !6, metadata !3, metadata !DIExpression())
define void @unsupported_argument(i64 %arg1) {
  tail call void @llvm.dbg.value(metadata !DIArgList(i64 %arg1, i64 undef), metadata !3, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_mul, DW_OP_plus, DW_OP_stack_value)), !dbg !5
  tail call void @llvm.dbg.value(metadata !6, metadata !3, metadata !DIExpression()), !dbg !5
  ret void
}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2)
!2 = !DIFile(filename: "import-failure.ll", directory: "/")
!3 = !DILocalVariable(scope: !4, name: "arg1", file: !2, line: 1, arg: 1, align: 64);
!4 = distinct !DISubprogram(name: "intrinsic", scope: !2, file: !2, spFlags: DISPFlagDefinition, unit: !1)
!5 = !DILocation(line: 1, column: 2, scope: !4)
!6 = !{}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: unsupported TBAA node format: !{{.*}} = !{!{{.*}}, i64 1, !"omnipotent char"}
define dso_local void @tbaa(ptr %0) {
  store i32 1, ptr %0, align 4, !tbaa !2
  ret void
}

!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!4, i64 4, !"int"}
!4 = !{!5, i64 1, !"omnipotent char"}
!5 = !{!"Simple C++ TBAA"}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: error: has cycle in TBAA graph: ![[ID:.*]] = distinct !{![[ID]], i64 4, !"int"}
define dso_local void @tbaa(ptr %0) {
  store i32 1, ptr %0, align 4, !tbaa !2
  ret void
}

!2 = !{!3, !3, i64 0, i64 4}
!3 = !{!3, i64 4, !"int"}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: warning: expected an access group node to be empty and distinct
; CHECK:      error: unsupported access group node: !0 = !{}
define void @access_group(ptr %arg1) {
  %1 = load i32, ptr %arg1, !llvm.access.group !0
  ret void
}

!0 = !{}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected all loop properties to be either debug locations or metadata nodes
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, i32 42}
define void @invalid_loop_node(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, i32 42}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: cannot import empty loop property
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @invalid_loop_node(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = distinct !{}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: cannot import loop property without a name
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @invalid_loop_node(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = distinct !{i1 0}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: cannot import loop properties with duplicated names llvm.loop.disable_nonforced
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1, !1}
!1 = !{!"llvm.loop.disable_nonforced"}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata node llvm.loop.disable_nonforced to hold no value
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.disable_nonforced", i1 0}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata nodes llvm.loop.unroll.enable and llvm.loop.unroll.disable to be mutually exclusive
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1, !2}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.unroll.enable"}
!2 = !{!"llvm.loop.unroll.disable"}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata node llvm.loop.vectorize.enable to hold a boolean value
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable"}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata node llvm.loop.vectorize.width to hold an i32 value
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.width", !0}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata node llvm.loop.vectorize.followup_all to hold an MDNode
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.followup_all", i32 42}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected metadata node llvm.loop.parallel_accesses to hold one or multiple MDNodes
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.parallel_accesses", i32 42}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: unknown loop annotation llvm.loop.typo
; CHECK:      <unknown>
; CHECK-SAME: warning: unhandled metadata: !0 = distinct !{!0, !1, !2}
define void @unsupported_loop_annotation(i64 %n, ptr %A) {
entry:
  br label %end, !llvm.loop !0
end:
  ret void
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.disable_nonforced"}
!2 = !{!"llvm.loop.typo"}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: could not lookup access group
define void @unused_access_group(ptr %arg) {
entry:
  %0 = load i32, ptr %arg, !llvm.access.group !0
  br label %end, !llvm.loop !1
end:
  ret void
}

!0 = distinct !{}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.parallel_accesses", !0, !3}
!3 = distinct !{}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: expected function_entry_count to be attached to a function
; CHECK:      warning: unhandled metadata: !0 = !{!"function_entry_count", i64 42}
define void @cond_br(i1 %arg) {
entry:
  br i1 %arg, label %bb1, label %bb2, !prof !0
bb1:
  ret void
bb2:
  ret void
}

!0 = !{!"function_entry_count", i64 42}

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: dropped instruction: call void @llvm.experimental.noalias.scope.decl(metadata !0)
define void @unused_scope() {
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)

!0 = !{!1}
!1 = !{!1, !2}
!2 = distinct !{!2, !"The domain"}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: dereferenceable metadata operand must be a non-negative constant integer
define void @deref(i64 %0) {
  %2 = inttoptr i64 %0 to ptr, !dereferenceable !0
  ret void
}

!0 = !{i64 -4}

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: warning: unhandled data layout token: ni:42
target datalayout = "e-ni:42-i64:64"

; // -----

; CHECK:      import-failure.ll
; CHECK-SAME: malformed specification, must be of the form "m:<mangling>"
target datalayout = "e-m-i64:64"

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: incompatible call and callee types: '!llvm.func<void (i64)>' and '!llvm.func<void (ptr)>'
define void @incompatible_call_and_callee_types() {
  call void @callee(i64 0)
  ret void
}

declare void @callee(ptr)

; // -----

; CHECK:      <unknown>
; CHECK-SAME: warning: incompatible call and callee types: '!llvm.func<void ()>' and '!llvm.func<i32 ()>'
define void @f() personality ptr @__gxx_personality_v0 {
entry:
  invoke void @g() to label %bb1 unwind label %bb2
bb1:
  ret void
bb2:
  %0 = landingpad i32 cleanup
  unreachable
}

declare i32 @g()

declare i32 @__gxx_personality_v0(...)
