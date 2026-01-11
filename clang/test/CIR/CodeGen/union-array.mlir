!s32i = !cir.int<s, 32>
!s64i = !cir.int<s, 64>
!s8i = !cir.int<s, 8>
!u8i = !cir.int<u, 8>
#fn_attr = #cir<extra({inline = #cir.inline<no>, nothrow = #cir.nothrow, optnone = #cir.optnone})>
!rec_S_1 = !cir.record<struct "S_1" {!s8i} #cir.record.decl.ast>
!rec_S_2 = !cir.record<struct "S_2" {!s64i, !s64i} #cir.record.decl.ast>
!rec_anon_struct = !cir.record<struct  {!s32i}>
!rec_U = !cir.record<union "U" {!rec_S_1, !rec_S_2} #cir.record.decl.ast>
!rec_anon_struct1 = !cir.record<struct  {!rec_S_2}>
!rec_anon_struct2 = !cir.record<struct  {!rec_S_1, !cir.array<!u8i x 15>}>
!rec_anon_struct3 = !cir.record<struct  {!s32i, !cir.array<!u8i x 12>}>
module @"/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c" attributes {cir.lang = #cir.lang<c>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  cir.global "private" constant cir_private @__const.bar.x = #cir.const_array<[#cir.global_view<@g> : !cir.ptr<!s32i>, #cir.global_view<@g> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> loc(#loc20)
  cir.global "private" constant cir_private @__const.foo.arr = #cir.const_array<[#cir.const_record<{#cir.const_record<{#cir.int<1> : !s64i, #cir.int<2> : !s64i}> : !rec_S_2}> : !rec_anon_struct1, #cir.const_record<{#cir.const_record<{#cir.int<1> : !s8i}> : !rec_S_1, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 15>}> : !rec_anon_struct2]> : !cir.array<!rec_U x 2> loc(#loc21)
  cir.func no_proto dso_local @foo() extra(#fn_attr) {
    %0 = cir.alloca !cir.array<!rec_U x 2>, !cir.ptr<!cir.array<!rec_U x 2>>, ["arr"] {alignment = 16 : i64} loc(#loc21)
    %1 = cir.get_global @__const.foo.arr : !cir.ptr<!cir.array<!rec_U x 2>> loc(#loc21)
    cir.copy %1 to %0 : !cir.ptr<!cir.array<!rec_U x 2>> loc(#loc21)
    cir.return loc(#loc6)
  } loc(#loc22)
  cir.global "private" internal dso_local @g = #cir.const_record<{#cir.int<5> : !s32i}> : !rec_anon_struct {alignment = 4 : i64} loc(#loc23)
  cir.func dso_local @bar() extra(#fn_attr) {
    %0 = cir.alloca !cir.array<!cir.ptr<!s32i> x 2>, !cir.ptr<!cir.array<!cir.ptr<!s32i> x 2>>, ["x"] {alignment = 16 : i64} loc(#loc20)
    %1 = cir.const #cir.const_array<[#cir.global_view<@g> : !cir.ptr<!s32i>, #cir.global_view<@g> : !cir.ptr<!s32i>]> : !cir.array<!cir.ptr<!s32i> x 2> loc(#loc20)
    %2 = cir.get_global @__const.bar.x : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 2>> loc(#loc20)
    cir.copy %2 to %0 : !cir.ptr<!cir.array<!cir.ptr<!s32i> x 2>> loc(#loc20)
    cir.return loc(#loc10)
  } loc(#loc24)
  cir.global "private" internal dso_local @g1 = #cir.const_array<[#cir.const_record<{#cir.int<66> : !s32i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 12>}> : !rec_anon_struct3, #cir.const_record<{#cir.int<66> : !s32i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 12>}> : !rec_anon_struct3, #cir.const_record<{#cir.int<66> : !s32i, #cir.const_array<[#cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i, #cir.zero : !u8i]> : !cir.array<!u8i x 12>}> : !rec_anon_struct3]> : !cir.array<!rec_anon_struct3 x 3> {alignment = 16 : i64} loc(#loc25)
  cir.global external @g2 = #cir.global_view<@g1, [1, 1, 4]> : !cir.ptr<!s32i> {alignment = 8 : i64} loc(#loc26)
  cir.func dso_local @baz() extra(#fn_attr) {
    %0 = cir.const #cir.int<4> : !s32i loc(#loc17)
    %1 = cir.get_global @g2 : !cir.ptr<!cir.ptr<!s32i>> loc(#loc26)
    %2 = cir.load deref align(8) %1 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> loc(#loc18)
    cir.store align(4) %0, %2 : !s32i, !cir.ptr<!s32i> loc(#loc28)
    cir.return loc(#loc16)
  } loc(#loc27)
} loc(#loc)
#loc = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":0:0)
#loc1 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":37:3)
#loc2 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":37:30)
#loc3 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":31:14)
#loc4 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":31:51)
#loc5 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":31:1)
#loc6 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":31:54)
#loc7 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":24:1)
#loc8 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":24:17)
#loc9 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":36:1)
#loc10 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":38:1)
#loc11 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":53:1)
#loc12 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":53:40)
#loc13 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":54:1)
#loc14 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":54:21)
#loc15 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":57:1)
#loc16 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":59:1)
#loc17 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":58:11)
#loc18 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":58:5)
#loc19 = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union-array.c":58:3)
#loc20 = loc(fused[#loc1, #loc2])
#loc21 = loc(fused[#loc3, #loc4])
#loc22 = loc(fused[#loc5, #loc6])
#loc23 = loc(fused[#loc7, #loc8])
#loc24 = loc(fused[#loc9, #loc10])
#loc25 = loc(fused[#loc11, #loc12])
#loc26 = loc(fused[#loc13, #loc14])
#loc27 = loc(fused[#loc15, #loc16])
#loc28 = loc(fused[#loc19, #loc17])
