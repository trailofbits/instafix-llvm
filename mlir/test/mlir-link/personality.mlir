// RUN: mlir-link --sort-symbols %s | FileCheck %s

// CHECK: llvm.func @__gxx_personality_v0(...) -> i32
// CHECK: llvm.func @bar() attributes {personality = @__gxx_personality_v0}
// CHECK: llvm.func @foo() -> i32

llvm.func @bar() attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.zero : !llvm.ptr
  %1 = llvm.invoke @foo() to ^bb1 unwind ^bb2 : () -> i32
^bb1:  // pred: ^bb0
  llvm.return
^bb2:  // pred: ^bb0
  %2 = llvm.landingpad (catch %0 : !llvm.ptr) : !llvm.struct<(ptr, i32)>
  llvm.return
}
llvm.func @foo() -> i32
llvm.func @__gxx_personality_v0(...) -> i32
