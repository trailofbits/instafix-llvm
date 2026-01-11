!s16i = !cir.int<s, 16>
!s32i = !cir.int<s, 32>
#fn_attr = #cir<extra({inline = #cir.inline<no>, nothrow = #cir.nothrow, optnone = #cir.optnone})>
#loc15 = loc("clang/test/CIR/CodeGen/union.cpp":57:32)
#loc16 = loc("clang/test/CIR/CodeGen/union.cpp":57:40)
#true = #cir.bool<true> : !cir.bool
!rec_A = !cir.record<union "A" {!s16i, !s32i} #cir.record.decl.ast>
!rec_U = !cir.record<union "U" {!cir.bool, !s16i, !s32i, !cir.float, !cir.double} #cir.record.decl.ast>
!rec_U23A3ADummy = !cir.record<struct "U2::Dummy" {!s16i, !cir.float} #cir.record.decl.ast>
!rec_yolm33A3Aanon = !cir.record<struct "yolm3::anon" {!cir.bool, !s32i} #cir.record.decl.ast>
!rec_yolm3A3Aanon = !cir.record<struct "yolm::anon" {!s32i} #cir.record.decl.ast>
!rec_yolo = !cir.record<struct "yolo" {!s32i} #cir.record.decl.ast>
#loc41 = loc(fused[#loc15, #loc16])
!rec_U2 = !cir.record<union "U2" {!cir.bool, !rec_U23A3ADummy} #cir.record.decl.ast>
!rec_U3 = !cir.record<union "U3" {!s16i, !rec_U} #cir.record.decl.ast>
!rec_yolm = !cir.record<union "yolm" {!rec_yolo, !rec_yolm3A3Aanon} #cir.record.decl.ast>
!rec_yolm23A3Aanon = !cir.record<struct "yolm2::anon" {!cir.ptr<!s32i>, !s32i} #cir.record.decl.ast>
!rec_yolm3 = !cir.record<union "yolm3" {!rec_yolo, !rec_yolm33A3Aanon} #cir.record.decl.ast>
!rec_yolm2 = !cir.record<union "yolm2" {!rec_yolo, !rec_yolm23A3Aanon} #cir.record.decl.ast>
module @"/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union.cpp" attributes {cir.lang = #cir.lang<cxx>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "x86_64-unknown-linux-gnu", cir.type_size_info = #cir.type_size_info<char = 8, int = 32, size_t = 64>, dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>} {
  cir.global external @u2 = #cir.zero : !rec_U2 {alignment = 4 : i64} loc(#loc34)
  cir.global external @u3 = #cir.zero : !rec_U3 {alignment = 8 : i64} loc(#loc35)
  cir.func dso_local @_Z1mv() extra(#fn_attr) {
    %0 = cir.alloca !rec_yolm, !cir.ptr<!rec_yolm>, ["q"] {alignment = 4 : i64} loc(#loc37)
    %1 = cir.alloca !rec_yolm2, !cir.ptr<!rec_yolm2>, ["q2"] {alignment = 8 : i64} loc(#loc38)
    %2 = cir.alloca !rec_yolm3, !cir.ptr<!rec_yolm3>, ["q3"] {alignment = 4 : i64} loc(#loc39)
    cir.return loc(#loc6)
  } loc(#loc36)
  cir.func dso_local @_Z25shouldGenerateUnionAccess1U(%arg0: !rec_U loc(fused[#loc15, #loc16])) extra(#fn_attr) {
    %0 = cir.alloca !rec_U, !cir.ptr<!rec_U>, ["u", init] {alignment = 8 : i64} loc(#loc41)
    cir.store %arg0, %0 : !rec_U, !cir.ptr<!rec_U> loc(#loc17)
    %1 = cir.const #true loc(#loc18)
    %2 = cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.bool> loc(#loc19)
    cir.store align(8) %1, %2 : !cir.bool, !cir.ptr<!cir.bool> loc(#loc42)
    %3 = cir.get_member %0[0] {name = "b"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.bool> loc(#loc19)
    %4 = cir.const #cir.int<1> : !s32i loc(#loc21)
    %5 = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U> -> !cir.ptr<!s32i> loc(#loc22)
    cir.store align(8) %4, %5 : !s32i, !cir.ptr<!s32i> loc(#loc43)
    %6 = cir.get_member %0[2] {name = "i"} : !cir.ptr<!rec_U> -> !cir.ptr<!s32i> loc(#loc22)
    %7 = cir.const #cir.fp<1.000000e-01> : !cir.float loc(#loc24)
    %8 = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.float> loc(#loc25)
    cir.store align(8) %7, %8 : !cir.float, !cir.ptr<!cir.float> loc(#loc44)
    %9 = cir.get_member %0[3] {name = "f"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.float> loc(#loc25)
    %10 = cir.const #cir.fp<1.000000e-01> : !cir.double loc(#loc27)
    %11 = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.double> loc(#loc28)
    cir.store align(8) %10, %11 : !cir.double, !cir.ptr<!cir.double> loc(#loc45)
    %12 = cir.get_member %0[4] {name = "d"} : !cir.ptr<!rec_U> -> !cir.ptr<!cir.double> loc(#loc28)
    cir.return loc(#loc14)
  } loc(#loc40)
  cir.func dso_local @_Z23noCrushOnDifferentSizesv() extra(#fn_attr) {
    %0 = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a"] {alignment = 4 : i64} loc(#loc47)
    %1 = cir.const #cir.zero : !rec_A loc(#loc47)
    cir.store align(4) %1, %0 : !rec_A, !cir.ptr<!rec_A> loc(#loc47)
    cir.return loc(#loc31)
  } loc(#loc46)
} loc(#loc)
#loc = loc("/home/wizardengineer/src/instafix-llvm/clang/test/CIR/CodeGen/union.cpp":0:0)
#loc1 = loc("clang/test/CIR/CodeGen/union.cpp":30:1)
#loc2 = loc("clang/test/CIR/CodeGen/union.cpp":36:3)
#loc3 = loc("clang/test/CIR/CodeGen/union.cpp":40:1)
#loc4 = loc("clang/test/CIR/CodeGen/union.cpp":43:3)
#loc5 = loc("clang/test/CIR/CodeGen/union.cpp":46:1)
#loc6 = loc("clang/test/CIR/CodeGen/union.cpp":50:1)
#loc7 = loc("clang/test/CIR/CodeGen/union.cpp":47:3)
#loc8 = loc("clang/test/CIR/CodeGen/union.cpp":47:8)
#loc9 = loc("clang/test/CIR/CodeGen/union.cpp":48:3)
#loc10 = loc("clang/test/CIR/CodeGen/union.cpp":48:9)
#loc11 = loc("clang/test/CIR/CodeGen/union.cpp":49:3)
#loc12 = loc("clang/test/CIR/CodeGen/union.cpp":49:9)
#loc13 = loc("clang/test/CIR/CodeGen/union.cpp":57:1)
#loc14 = loc("clang/test/CIR/CodeGen/union.cpp":78:1)
#loc17 = loc("clang/test/CIR/CodeGen/union.cpp":57:43)
#loc18 = loc("clang/test/CIR/CodeGen/union.cpp":58:9)
#loc19 = loc("clang/test/CIR/CodeGen/union.cpp":21:8)
#loc20 = loc("clang/test/CIR/CodeGen/union.cpp":58:3)
#loc21 = loc("clang/test/CIR/CodeGen/union.cpp":63:9)
#loc22 = loc("clang/test/CIR/CodeGen/union.cpp":23:7)
#loc23 = loc("clang/test/CIR/CodeGen/union.cpp":63:3)
#loc24 = loc("clang/test/CIR/CodeGen/union.cpp":68:9)
#loc25 = loc("clang/test/CIR/CodeGen/union.cpp":24:9)
#loc26 = loc("clang/test/CIR/CodeGen/union.cpp":68:3)
#loc27 = loc("clang/test/CIR/CodeGen/union.cpp":73:9)
#loc28 = loc("clang/test/CIR/CodeGen/union.cpp":25:10)
#loc29 = loc("clang/test/CIR/CodeGen/union.cpp":73:3)
#loc30 = loc("clang/test/CIR/CodeGen/union.cpp":85:1)
#loc31 = loc("clang/test/CIR/CodeGen/union.cpp":91:1)
#loc32 = loc("clang/test/CIR/CodeGen/union.cpp":86:3)
#loc33 = loc("clang/test/CIR/CodeGen/union.cpp":86:11)
#loc34 = loc(fused[#loc1, #loc2])
#loc35 = loc(fused[#loc3, #loc4])
#loc36 = loc(fused[#loc5, #loc6])
#loc37 = loc(fused[#loc7, #loc8])
#loc38 = loc(fused[#loc9, #loc10])
#loc39 = loc(fused[#loc11, #loc12])
#loc40 = loc(fused[#loc13, #loc14])
#loc42 = loc(fused[#loc20, #loc18])
#loc43 = loc(fused[#loc23, #loc21])
#loc44 = loc(fused[#loc26, #loc24])
#loc45 = loc(fused[#loc29, #loc27])
#loc46 = loc(fused[#loc30, #loc31])
#loc47 = loc(fused[#loc32, #loc33])
