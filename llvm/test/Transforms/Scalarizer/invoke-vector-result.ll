; RUN: opt %s -passes='function(scalarizer<load-store>)' -S | FileCheck %s
; RUN: opt %s -passes='function(scalarizer<load-store>,dce)' -S | FileCheck %s --check-prefix=CHECK-DCE

; Test that scalarizing a store of a vector value returned by an invoke
; instruction works correctly. Previously, this would trigger undefined
; behavior by trying to insert instructions after the invoke terminator.

declare <4 x float> @may_throw()
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @test_invoke_vector_result(
; CHECK: entry:
; CHECK:   %dest.i1 = getelementptr float, ptr %dest, i32 1
; CHECK:   %dest.i2 = getelementptr float, ptr %dest, i32 2
; CHECK:   %dest.i3 = getelementptr float, ptr %dest, i32 3
; CHECK:   %result = invoke <4 x float> @may_throw()
; CHECK:           to label %cont unwind label %lpad
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   store float %result.i0, ptr %dest, align 16
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   store float %result.i1, ptr %dest.i1, align 4
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   store float %result.i2, ptr %dest.i2, align 8
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   store float %result.i3, ptr %dest.i3, align 4
; CHECK:   ret void
; CHECK: lpad:
; CHECK:   %lp = landingpad { ptr, i32 }
; CHECK:           cleanup
; CHECK:   ret void

; CHECK-DCE-LABEL: @test_invoke_vector_result(
; CHECK-DCE: cont:
; CHECK-DCE:   extractelement <4 x float> %result
; CHECK-DCE:   store float
define void @test_invoke_vector_result(ptr %dest) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  store <4 x float> %result, ptr %dest
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Test with multiple uses of the invoke result in the same block
; CHECK-LABEL: @test_invoke_multiple_uses(
; CHECK: entry:
; CHECK:   %dest2.i1 = getelementptr float, ptr %dest2, i32 1
; CHECK:   %dest2.i2 = getelementptr float, ptr %dest2, i32 2
; CHECK:   %dest2.i3 = getelementptr float, ptr %dest2, i32 3
; CHECK:   %dest1.i1 = getelementptr float, ptr %dest1, i32 1
; CHECK:   %dest1.i2 = getelementptr float, ptr %dest1, i32 2
; CHECK:   %dest1.i3 = getelementptr float, ptr %dest1, i32 3
; CHECK:   %result = invoke <4 x float> @may_throw()
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   store float %result.i0, ptr %dest1
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   store float %result.i1, ptr %dest1.i1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   store float %result.i2, ptr %dest1.i2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   store float %result.i3, ptr %dest1.i3
; CHECK:   store float %result.i0, ptr %dest2
; CHECK:   store float %result.i1, ptr %dest2.i1
; CHECK:   store float %result.i2, ptr %dest2.i2
; CHECK:   store float %result.i3, ptr %dest2.i3
define void @test_invoke_multiple_uses(ptr %dest1, ptr %dest2) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  store <4 x float> %result, ptr %dest1
  store <4 x float> %result, ptr %dest2
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Test with invoke result used in a different successor block
; CHECK-LABEL: @test_invoke_different_block(
; CHECK: entry:
; CHECK:   %dest.i1 = getelementptr float, ptr %dest, i32 1
; CHECK:   %dest.i2 = getelementptr float, ptr %dest, i32 2
; CHECK:   %dest.i3 = getelementptr float, ptr %dest, i32 3
; CHECK:   %result = invoke <4 x float> @may_throw()
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   br label %use_block
; CHECK: use_block:
; CHECK:   store float %result.i0, ptr %dest
; CHECK:   store float %result.i1, ptr %dest.i1
; CHECK:   store float %result.i2, ptr %dest.i2
; CHECK:   store float %result.i3, ptr %dest.i3
define void @test_invoke_different_block(ptr %dest) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  br label %use_block

use_block:
  store <4 x float> %result, ptr %dest
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; Test with invoke result used in arithmetic before store
; CHECK-LABEL: @test_invoke_with_arithmetic(
; CHECK: entry:
; CHECK:   %dest.i1 = getelementptr float, ptr %dest, i32 1
; CHECK:   %dest.i2 = getelementptr float, ptr %dest, i32 2
; CHECK:   %dest.i3 = getelementptr float, ptr %dest, i32 3
; CHECK:   %result = invoke <4 x float> @may_throw()
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %add.i0 = fadd float %result.i0, %result.i0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %add.i1 = fadd float %result.i1, %result.i1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %add.i2 = fadd float %result.i2, %result.i2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   %add.i3 = fadd float %result.i3, %result.i3
; CHECK:   store float %add.i0, ptr %dest
; CHECK:   store float %add.i1, ptr %dest.i1
; CHECK:   store float %add.i2, ptr %dest.i2
; CHECK:   store float %add.i3, ptr %dest.i3
define void @test_invoke_with_arithmetic(ptr %dest) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  %add = fadd <4 x float> %result, %result
  store <4 x float> %add, ptr %dest
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; =============================================================================
; EDGE CASE TESTS - These test cases expose potential issues with the current fix
; =============================================================================

; EDGE CASE 1: Invoke result used in non-dominating sibling blocks
; With the fix, extracts are placed at the start of %cont (the normal destination),
; which correctly dominates both %then and %else branches.
;
; CHECK-LABEL: @test_invoke_sibling_blocks(
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   br i1 %cond, label %then, label %else
; CHECK: then:
; CHECK:   store float %result.i0, ptr %dest1
; CHECK: else:
; CHECK:   store float %result.i0, ptr %dest2
define void @test_invoke_sibling_blocks(ptr %dest1, ptr %dest2, i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  br i1 %cond, label %then, label %else

then:
  store <4 x float> %result, ptr %dest1
  br label %merge

else:
  store <4 x float> %result, ptr %dest2
  br label %merge

merge:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 2: PHI node using invoke result
; With the fix, extracts are placed at the start of %cont, making them
; available from the %cont incoming edge to the PHI.
;
; CHECK-LABEL: @test_invoke_phi_user(
; CHECK: entry:
; CHECK:   %result = invoke <4 x float> @may_throw()
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   br label %merge
; CHECK: other:
; CHECK:   br label %merge
; CHECK: merge:
; CHECK:   %phi.i0 = phi float
; CHECK:   %phi.i1 = phi float
; CHECK:   %phi.i2 = phi float
; CHECK:   %phi.i3 = phi float
define void @test_invoke_phi_user(ptr %dest, ptr %other_src) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  br label %merge

other:
  %other_vec = load <4 x float>, ptr %other_src
  br label %merge

merge:
  %phi = phi <4 x float> [ %result, %cont ], [ %other_vec, %other ]
  store <4 x float> %phi, ptr %dest
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 3: Invoke result used in PHI and also directly
; With the fix, extracts are placed at the start of %cont, enabling both
; direct use and PHI scalarization to work correctly.
;
; CHECK-LABEL: @test_invoke_phi_and_direct_use(
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   store float %result.i0, ptr %dest1
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   store float %result.i1, ptr %dest1.i1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   store float %result.i2, ptr %dest1.i2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   store float %result.i3, ptr %dest1.i3
; CHECK: use_phi:
; CHECK:   %phi.i0 = phi float [ %result.i0, %cont ]
define void @test_invoke_phi_and_direct_use(ptr %dest1, ptr %dest2, i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  store <4 x float> %result, ptr %dest1  ; Direct use
  br i1 %cond, label %use_phi, label %exit

use_phi:
  %phi = phi <4 x float> [ %result, %cont ]
  store <4 x float> %phi, ptr %dest2
  br label %exit

exit:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 4: Multiple invokes with vector results
; Tests that caching doesn't incorrectly share between different invokes.
; Extracts for %result1 are placed in %cont1, extracts for %result2 in %cont2.
;
; CHECK-LABEL: @test_multiple_invokes(
; CHECK: entry:
; CHECK:   %result1 = invoke <4 x float> @may_throw()
; CHECK: cont1:
; CHECK:   %result1.i0 = extractelement <4 x float> %result1
; CHECK:   %result1.i1 = extractelement <4 x float> %result1
; CHECK:   %result1.i2 = extractelement <4 x float> %result1
; CHECK:   %result1.i3 = extractelement <4 x float> %result1
; CHECK:   %result2 = invoke <4 x float> @may_throw()
; CHECK: cont2:
; CHECK:   %result2.i0 = extractelement <4 x float> %result2
; CHECK:   %result2.i1 = extractelement <4 x float> %result2
; CHECK:   %result2.i2 = extractelement <4 x float> %result2
; CHECK:   %result2.i3 = extractelement <4 x float> %result2
define void @test_multiple_invokes(ptr %dest1, ptr %dest2) personality ptr @__gxx_personality_v0 {
entry:
  %result1 = invoke <4 x float> @may_throw() to label %cont1 unwind label %lpad

cont1:
  %result2 = invoke <4 x float> @may_throw() to label %cont2 unwind label %lpad

cont2:
  store <4 x float> %result1, ptr %dest1
  store <4 x float> %result2, ptr %dest2
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 5: Invoke result used in loop
; With the fix, extracts are placed at the start of %loop (after PHI nodes).
; The extracts are inside the loop but are loop-invariant, so LICM could
; hoist them if needed.
;
; CHECK-LABEL: @test_invoke_in_loop(
; CHECK: loop:
; CHECK:   %i = phi i32
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
define void @test_invoke_in_loop(ptr %dest, i32 %n) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %loop unwind label %lpad

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ptr = getelementptr <4 x float>, ptr %dest, i32 %i
  store <4 x float> %result, ptr %ptr
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 6: Diamond CFG with invoke result used in merge block
; With the fix, extracts are placed at the start of %cont, which dominates
; all subsequent blocks including the merge block.
;
; CHECK-LABEL: @test_invoke_diamond_merge(
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   br i1 %cond
; CHECK: left:
; CHECK:   br label %merge
; CHECK: right:
; CHECK:   br label %merge
; CHECK: merge:
; CHECK:   store float %result.i0
define void @test_invoke_diamond_merge(ptr %dest, i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  br i1 %cond, label %left, label %right

left:
  br label %merge

right:
  br label %merge

merge:
  store <4 x float> %result, ptr %dest
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}

; EDGE CASE 7: Invoke result used in both branches AND merge block
; With the fix, extracts are placed at the start of %cont and correctly
; reused in all three use sites (left, right, merge).
;
; CHECK-LABEL: @test_invoke_diamond_all_uses(
; CHECK: cont:
; CHECK:   %result.i0 = extractelement <4 x float> %result, i64 0
; CHECK:   %result.i1 = extractelement <4 x float> %result, i64 1
; CHECK:   %result.i2 = extractelement <4 x float> %result, i64 2
; CHECK:   %result.i3 = extractelement <4 x float> %result, i64 3
; CHECK:   br i1 %cond, label %left, label %right
; CHECK: left:
; CHECK:   store float %result.i0, ptr %dest1
; CHECK: right:
; CHECK:   store float %result.i0, ptr %dest2
; CHECK: merge:
; CHECK:   store float %result.i0, ptr %dest3
define void @test_invoke_diamond_all_uses(ptr %dest1, ptr %dest2, ptr %dest3, i1 %cond) personality ptr @__gxx_personality_v0 {
entry:
  %result = invoke <4 x float> @may_throw() to label %cont unwind label %lpad

cont:
  br i1 %cond, label %left, label %right

left:
  store <4 x float> %result, ptr %dest1
  br label %merge

right:
  store <4 x float> %result, ptr %dest2
  br label %merge

merge:
  store <4 x float> %result, ptr %dest3
  ret void

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  ret void
}
