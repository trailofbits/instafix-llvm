// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:    -fclangir -disable-O0-optnone \
// RUN:  -flax-vector-conversions=none -emit-llvm -fno-clangir-call-conv-lowering -o - %s \
// RUN: | opt -S -passes=mem2reg,simplifycfg -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// REQUIRES: aarch64-registered-target || arm-registered-target

// This test mimics clang/test/CodeGen/AArch64/neon-intrinsics.c, which eventually
// CIR shall be able to support fully. Since this is going to take some time to converge,
// the unsupported/NYI code is commented out, so that we can incrementally improve this.
// The NYI filecheck used contains the LLVM output from OG codegen that should guide the
// correct result when implementing this into the CIR pipeline.

#include <arm_neon.h>

// NYI-LABEL: @test_vadd_s8(
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[ADD_I]]
// int8x8_t test_vadd_s8(int8x8_t v1, int8x8_t v2) {
//   return vadd_s8(v1, v2);
// }

// NYI-LABEL: @test_vadd_s16(
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[ADD_I]]
// int16x4_t test_vadd_s16(int16x4_t v1, int16x4_t v2) {
//   return vadd_s16(v1, v2);
// }

// NYI-LABEL: @test_vadd_s32(
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[ADD_I]]
// int32x2_t test_vadd_s32(int32x2_t v1, int32x2_t v2) {
//   return vadd_s32(v1, v2);
// }

// NYI-LABEL: @test_vadd_s64(
// NYI:   [[ADD_I:%.*]] = add <1 x i64> %v1, %v2
// NYI:   ret <1 x i64> [[ADD_I]]
// int64x1_t test_vadd_s64(int64x1_t v1, int64x1_t v2) {
//   return vadd_s64(v1, v2);
// }

// NYI-LABEL: @test_vadd_f32(
// NYI:   [[ADD_I:%.*]] = fadd <2 x float> %v1, %v2
// NYI:   ret <2 x float> [[ADD_I]]
// float32x2_t test_vadd_f32(float32x2_t v1, float32x2_t v2) {
//   return vadd_f32(v1, v2);
// }

// NYI-LABEL: @test_vadd_u8(
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[ADD_I]]
// uint8x8_t test_vadd_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vadd_u8(v1, v2);
// }

// NYI-LABEL: @test_vadd_u16(
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[ADD_I]]
// uint16x4_t test_vadd_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vadd_u16(v1, v2);
// }

// NYI-LABEL: @test_vadd_u32(
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[ADD_I]]
// uint32x2_t test_vadd_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vadd_u32(v1, v2);
// }

// NYI-LABEL: @test_vadd_u64(
// NYI:   [[ADD_I:%.*]] = add <1 x i64> %v1, %v2
// NYI:   ret <1 x i64> [[ADD_I]]
// uint64x1_t test_vadd_u64(uint64x1_t v1, uint64x1_t v2) {
//   return vadd_u64(v1, v2);
// }

// NYI-LABEL: @test_vaddq_s8(
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[ADD_I]]
// int8x16_t test_vaddq_s8(int8x16_t v1, int8x16_t v2) {
//   return vaddq_s8(v1, v2);
// }

// NYI-LABEL: @test_vaddq_s16(
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vaddq_s16(int16x8_t v1, int16x8_t v2) {
//   return vaddq_s16(v1, v2);
// }

// NYI-LABEL: @test_vaddq_s32(
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vaddq_s32(int32x4_t v1, int32x4_t v2) {
//   return vaddq_s32(v1, v2);
// }

// NYI-LABEL: @test_vaddq_s64(
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %v1, %v2
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vaddq_s64(int64x2_t v1, int64x2_t v2) {
//   return vaddq_s64(v1, v2);
// }

// NYI-LABEL: @test_vaddq_f32(
// NYI:   [[ADD_I:%.*]] = fadd <4 x float> %v1, %v2
// NYI:   ret <4 x float> [[ADD_I]]
// float32x4_t test_vaddq_f32(float32x4_t v1, float32x4_t v2) {
//   return vaddq_f32(v1, v2);
// }

// NYI-LABEL: @test_vaddq_f64(
// NYI:   [[ADD_I:%.*]] = fadd <2 x double> %v1, %v2
// NYI:   ret <2 x double> [[ADD_I]]
// float64x2_t test_vaddq_f64(float64x2_t v1, float64x2_t v2) {
//   return vaddq_f64(v1, v2);
// }

// NYI-LABEL: @test_vaddq_u8(
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[ADD_I]]
// uint8x16_t test_vaddq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vaddq_u8(v1, v2);
// }

// NYI-LABEL: @test_vaddq_u16(
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vaddq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vaddq_u16(v1, v2);
// }

// NYI-LABEL: @test_vaddq_u32(
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vaddq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vaddq_u32(v1, v2);
// }

// NYI-LABEL: @test_vaddq_u64(
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %v1, %v2
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vaddq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vaddq_u64(v1, v2);
// }

// NYI-LABEL: @test_vsub_s8(
// NYI:   [[SUB_I:%.*]] = sub <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[SUB_I]]
// int8x8_t test_vsub_s8(int8x8_t v1, int8x8_t v2) {
//   return vsub_s8(v1, v2);
// }

// NYI-LABEL: @test_vsub_s16(
// NYI:   [[SUB_I:%.*]] = sub <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[SUB_I]]
// int16x4_t test_vsub_s16(int16x4_t v1, int16x4_t v2) {
//   return vsub_s16(v1, v2);
// }

// NYI-LABEL: @test_vsub_s32(
// NYI:   [[SUB_I:%.*]] = sub <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[SUB_I]]
// int32x2_t test_vsub_s32(int32x2_t v1, int32x2_t v2) {
//   return vsub_s32(v1, v2);
// }

// NYI-LABEL: @test_vsub_s64(
// NYI:   [[SUB_I:%.*]] = sub <1 x i64> %v1, %v2
// NYI:   ret <1 x i64> [[SUB_I]]
// int64x1_t test_vsub_s64(int64x1_t v1, int64x1_t v2) {
//   return vsub_s64(v1, v2);
// }

// NYI-LABEL: @test_vsub_f32(
// NYI:   [[SUB_I:%.*]] = fsub <2 x float> %v1, %v2
// NYI:   ret <2 x float> [[SUB_I]]
// float32x2_t test_vsub_f32(float32x2_t v1, float32x2_t v2) {
//   return vsub_f32(v1, v2);
// }

// NYI-LABEL: @test_vsub_u8(
// NYI:   [[SUB_I:%.*]] = sub <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[SUB_I]]
// uint8x8_t test_vsub_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vsub_u8(v1, v2);
// }

// NYI-LABEL: @test_vsub_u16(
// NYI:   [[SUB_I:%.*]] = sub <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[SUB_I]]
// uint16x4_t test_vsub_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vsub_u16(v1, v2);
// }

// NYI-LABEL: @test_vsub_u32(
// NYI:   [[SUB_I:%.*]] = sub <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[SUB_I]]
// uint32x2_t test_vsub_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vsub_u32(v1, v2);
// }

// NYI-LABEL: @test_vsub_u64(
// NYI:   [[SUB_I:%.*]] = sub <1 x i64> %v1, %v2
// NYI:   ret <1 x i64> [[SUB_I]]
// uint64x1_t test_vsub_u64(uint64x1_t v1, uint64x1_t v2) {
//   return vsub_u64(v1, v2);
// }

// NYI-LABEL: @test_vsubq_s8(
// NYI:   [[SUB_I:%.*]] = sub <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[SUB_I]]
// int8x16_t test_vsubq_s8(int8x16_t v1, int8x16_t v2) {
//   return vsubq_s8(v1, v2);
// }

// NYI-LABEL: @test_vsubq_s16(
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vsubq_s16(int16x8_t v1, int16x8_t v2) {
//   return vsubq_s16(v1, v2);
// }

// NYI-LABEL: @test_vsubq_s32(
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vsubq_s32(int32x4_t v1, int32x4_t v2) {
//   return vsubq_s32(v1, v2);
// }

// NYI-LABEL: @test_vsubq_s64(
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %v1, %v2
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vsubq_s64(int64x2_t v1, int64x2_t v2) {
//   return vsubq_s64(v1, v2);
// }

// NYI-LABEL: @test_vsubq_f32(
// NYI:   [[SUB_I:%.*]] = fsub <4 x float> %v1, %v2
// NYI:   ret <4 x float> [[SUB_I]]
// float32x4_t test_vsubq_f32(float32x4_t v1, float32x4_t v2) {
//   return vsubq_f32(v1, v2);
// }

// NYI-LABEL: @test_vsubq_f64(
// NYI:   [[SUB_I:%.*]] = fsub <2 x double> %v1, %v2
// NYI:   ret <2 x double> [[SUB_I]]
// float64x2_t test_vsubq_f64(float64x2_t v1, float64x2_t v2) {
//   return vsubq_f64(v1, v2);
// }

// NYI-LABEL: @test_vsubq_u8(
// NYI:   [[SUB_I:%.*]] = sub <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[SUB_I]]
// uint8x16_t test_vsubq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vsubq_u8(v1, v2);
// }

// NYI-LABEL: @test_vsubq_u16(
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vsubq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vsubq_u16(v1, v2);
// }

// NYI-LABEL: @test_vsubq_u32(
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vsubq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vsubq_u32(v1, v2);
// }

// NYI-LABEL: @test_vsubq_u64(
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %v1, %v2
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vsubq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vsubq_u64(v1, v2);
// }

// NYI-LABEL: @test_vmul_s8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[MUL_I]]
// int8x8_t test_vmul_s8(int8x8_t v1, int8x8_t v2) {
//   return vmul_s8(v1, v2);
// }

// NYI-LABEL: @test_vmul_s16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[MUL_I]]
// int16x4_t test_vmul_s16(int16x4_t v1, int16x4_t v2) {
//   return vmul_s16(v1, v2);
// }

// NYI-LABEL: @test_vmul_s32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[MUL_I]]
// int32x2_t test_vmul_s32(int32x2_t v1, int32x2_t v2) {
//   return vmul_s32(v1, v2);
// }

// NYI-LABEL: @test_vmul_f32(
// NYI:   [[MUL_I:%.*]] = fmul <2 x float> %v1, %v2
// NYI:   ret <2 x float> [[MUL_I]]
// float32x2_t test_vmul_f32(float32x2_t v1, float32x2_t v2) {
//   return vmul_f32(v1, v2);
// }

// NYI-LABEL: @test_vmul_u8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v1, %v2
// NYI:   ret <8 x i8> [[MUL_I]]
// uint8x8_t test_vmul_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vmul_u8(v1, v2);
// }

// NYI-LABEL: @test_vmul_u16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v1, %v2
// NYI:   ret <4 x i16> [[MUL_I]]
// uint16x4_t test_vmul_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vmul_u16(v1, v2);
// }

// NYI-LABEL: @test_vmul_u32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v1, %v2
// NYI:   ret <2 x i32> [[MUL_I]]
// uint32x2_t test_vmul_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vmul_u32(v1, v2);
// }

// NYI-LABEL: @test_vmulq_s8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[MUL_I]]
// int8x16_t test_vmulq_s8(int8x16_t v1, int8x16_t v2) {
//   return vmulq_s8(v1, v2);
// }

// NYI-LABEL: @test_vmulq_s16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[MUL_I]]
// int16x8_t test_vmulq_s16(int16x8_t v1, int16x8_t v2) {
//   return vmulq_s16(v1, v2);
// }

// NYI-LABEL: @test_vmulq_s32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[MUL_I]]
// int32x4_t test_vmulq_s32(int32x4_t v1, int32x4_t v2) {
//   return vmulq_s32(v1, v2);
// }

// NYI-LABEL: @test_vmulq_u8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v1, %v2
// NYI:   ret <16 x i8> [[MUL_I]]
// uint8x16_t test_vmulq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vmulq_u8(v1, v2);
// }

// NYI-LABEL: @test_vmulq_u16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v1, %v2
// NYI:   ret <8 x i16> [[MUL_I]]
// uint16x8_t test_vmulq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vmulq_u16(v1, v2);
// }

// NYI-LABEL: @test_vmulq_u32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v1, %v2
// NYI:   ret <4 x i32> [[MUL_I]]
// uint32x4_t test_vmulq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vmulq_u32(v1, v2);
// }

// NYI-LABEL: @test_vmulq_f32(
// NYI:   [[MUL_I:%.*]] = fmul <4 x float> %v1, %v2
// NYI:   ret <4 x float> [[MUL_I]]
// float32x4_t test_vmulq_f32(float32x4_t v1, float32x4_t v2) {
//   return vmulq_f32(v1, v2);
// }

// NYI-LABEL: @test_vmulq_f64(
// NYI:   [[MUL_I:%.*]] = fmul <2 x double> %v1, %v2
// NYI:   ret <2 x double> [[MUL_I]]
// float64x2_t test_vmulq_f64(float64x2_t v1, float64x2_t v2) {
//   return vmulq_f64(v1, v2);
// }

// NYI-LABEL: @test_vmul_p8(
// NYI:   [[VMUL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.pmul.v8i8(<8 x i8> %v1, <8 x i8> %v2)
// NYI:   ret <8 x i8> [[VMUL_V_I]]
// poly8x8_t test_vmul_p8(poly8x8_t v1, poly8x8_t v2) {
//   return vmul_p8(v1, v2);
// }

// NYI-LABEL: @test_vmulq_p8(
// NYI:   [[VMULQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.pmul.v16i8(<16 x i8> %v1, <16 x i8> %v2)
// NYI:   ret <16 x i8> [[VMULQ_V_I]]
// poly8x16_t test_vmulq_p8(poly8x16_t v1, poly8x16_t v2) {
//   return vmulq_p8(v1, v2);
// }

// NYI-LABEL: @test_vmla_s8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[MUL_I]]
// NYI:   ret <8 x i8> [[ADD_I]]
// int8x8_t test_vmla_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
//   return vmla_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_s16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[MUL_I]]
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[ADD_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vmla_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
//   return (int8x8_t)vmla_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_s32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[MUL_I]]
// NYI:   ret <2 x i32> [[ADD_I]]
// int32x2_t test_vmla_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
//   return vmla_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_f32(
// NYI:   [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// NYI:   [[ADD_I:%.*]] = fadd <2 x float> %v1, [[MUL_I]]
// NYI:   ret <2 x float> [[ADD_I]]
// float32x2_t test_vmla_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
//   return vmla_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_u8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[MUL_I]]
// NYI:   ret <8 x i8> [[ADD_I]]
// uint8x8_t test_vmla_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
//   return vmla_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_u16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[MUL_I]]
// NYI:   ret <4 x i16> [[ADD_I]]
// uint16x4_t test_vmla_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
//   return vmla_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmla_u32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[MUL_I]]
// NYI:   ret <2 x i32> [[ADD_I]]
// uint32x2_t test_vmla_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
//   return vmla_u32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_s8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[MUL_I]]
// NYI:   ret <16 x i8> [[ADD_I]]
// int8x16_t test_vmlaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
//   return vmlaq_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_s16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[MUL_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vmlaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
//   return vmlaq_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_s32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[MUL_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vmlaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
//   return vmlaq_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_f32(
// NYI:   [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// NYI:   [[ADD_I:%.*]] = fadd <4 x float> %v1, [[MUL_I]]
// NYI:   ret <4 x float> [[ADD_I]]
// float32x4_t test_vmlaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
//   return vmlaq_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_u8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[MUL_I]]
// NYI:   ret <16 x i8> [[ADD_I]]
// uint8x16_t test_vmlaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
//   return vmlaq_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_u16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[MUL_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vmlaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
//   return vmlaq_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_u32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[MUL_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vmlaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
//   return vmlaq_u32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlaq_f64(
// NYI:   [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// NYI:   [[ADD_I:%.*]] = fadd <2 x double> %v1, [[MUL_I]]
// NYI:   ret <2 x double> [[ADD_I]]
// float64x2_t test_vmlaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
//   return vmlaq_f64(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_s8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <8 x i8> %v1, [[MUL_I]]
// NYI:   ret <8 x i8> [[SUB_I]]
// int8x8_t test_vmls_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
//   return vmls_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_s16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <4 x i16> %v1, [[MUL_I]]
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SUB_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vmls_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
//   return (int8x8_t)vmls_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_s32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <2 x i32> %v1, [[MUL_I]]
// NYI:   ret <2 x i32> [[SUB_I]]
// int32x2_t test_vmls_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
//   return vmls_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_f32(
// NYI:   [[MUL_I:%.*]] = fmul <2 x float> %v2, %v3
// NYI:   [[SUB_I:%.*]] = fsub <2 x float> %v1, [[MUL_I]]
// NYI:   ret <2 x float> [[SUB_I]]
// float32x2_t test_vmls_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
//   return vmls_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_u8(
// NYI:   [[MUL_I:%.*]] = mul <8 x i8> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <8 x i8> %v1, [[MUL_I]]
// NYI:   ret <8 x i8> [[SUB_I]]
// uint8x8_t test_vmls_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
//   return vmls_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_u16(
// NYI:   [[MUL_I:%.*]] = mul <4 x i16> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <4 x i16> %v1, [[MUL_I]]
// NYI:   ret <4 x i16> [[SUB_I]]
// uint16x4_t test_vmls_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
//   return vmls_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmls_u32(
// NYI:   [[MUL_I:%.*]] = mul <2 x i32> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <2 x i32> %v1, [[MUL_I]]
// NYI:   ret <2 x i32> [[SUB_I]]
// uint32x2_t test_vmls_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
//   return vmls_u32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_s8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <16 x i8> %v1, [[MUL_I]]
// NYI:   ret <16 x i8> [[SUB_I]]
// int8x16_t test_vmlsq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
//   return vmlsq_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_s16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %v1, [[MUL_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vmlsq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
//   return vmlsq_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_s32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %v1, [[MUL_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vmlsq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
//   return vmlsq_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_f32(
// NYI:   [[MUL_I:%.*]] = fmul <4 x float> %v2, %v3
// NYI:   [[SUB_I:%.*]] = fsub <4 x float> %v1, [[MUL_I]]
// NYI:   ret <4 x float> [[SUB_I]]
// float32x4_t test_vmlsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
//   return vmlsq_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_u8(
// NYI:   [[MUL_I:%.*]] = mul <16 x i8> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <16 x i8> %v1, [[MUL_I]]
// NYI:   ret <16 x i8> [[SUB_I]]
// uint8x16_t test_vmlsq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
//   return vmlsq_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_u16(
// NYI:   [[MUL_I:%.*]] = mul <8 x i16> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %v1, [[MUL_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vmlsq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
//   return vmlsq_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_u32(
// NYI:   [[MUL_I:%.*]] = mul <4 x i32> %v2, %v3
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %v1, [[MUL_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vmlsq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
//   return vmlsq_u32(v1, v2, v3);
// }

// NYI-LABEL: @test_vmlsq_f64(
// NYI:   [[MUL_I:%.*]] = fmul <2 x double> %v2, %v3
// NYI:   [[SUB_I:%.*]] = fsub <2 x double> %v1, [[MUL_I]]
// NYI:   ret <2 x double> [[SUB_I]]
// float64x2_t test_vmlsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
//   return vmlsq_f64(v1, v2, v3);
// }

// NYI-LABEL: @test_vfma_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// NYI:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %v2, <2 x float> %v3, <2 x float> %v1)
// NYI:   ret <2 x float> [[TMP3]]
// float32x2_t test_vfma_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
//   return vfma_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vfmaq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// NYI:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> %v2, <4 x float> %v3, <4 x float> %v1)
// NYI:   ret <4 x float> [[TMP3]]
// float32x4_t test_vfmaq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
//   return vfmaq_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vfmaq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// NYI:   [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> %v2, <2 x double> %v3, <2 x double> %v1)
// NYI:   ret <2 x double> [[TMP3]]
// float64x2_t test_vfmaq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
//   return vfmaq_f64(v1, v2, v3);
// }

// NYI-LABEL: @test_vfms_f32(
// NYI:   [[SUB_I:%.*]] = fneg <2 x float> %v2
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> [[SUB_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x float> %v3 to <8 x i8>
// NYI:   [[TMP3:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[SUB_I]], <2 x float> %v3, <2 x float> %v1)
// NYI:   ret <2 x float> [[TMP3]]
// float32x2_t test_vfms_f32(float32x2_t v1, float32x2_t v2, float32x2_t v3) {
//   return vfms_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vfmsq_f32(
// NYI:   [[SUB_I:%.*]] = fneg <4 x float> %v2
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> [[SUB_I]] to <16 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <4 x float> %v3 to <16 x i8>
// NYI:   [[TMP3:%.*]] = call <4 x float> @llvm.fma.v4f32(<4 x float> [[SUB_I]], <4 x float> %v3, <4 x float> %v1)
// NYI:   ret <4 x float> [[TMP3]]
// float32x4_t test_vfmsq_f32(float32x4_t v1, float32x4_t v2, float32x4_t v3) {
//   return vfmsq_f32(v1, v2, v3);
// }

// NYI-LABEL: @test_vfmsq_f64(
// NYI:   [[SUB_I:%.*]] = fneg <2 x double> %v2
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> [[SUB_I]] to <16 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x double> %v3 to <16 x i8>
// NYI:   [[TMP3:%.*]] = call <2 x double> @llvm.fma.v2f64(<2 x double> [[SUB_I]], <2 x double> %v3, <2 x double> %v1)
// NYI:   ret <2 x double> [[TMP3]]
// float64x2_t test_vfmsq_f64(float64x2_t v1, float64x2_t v2, float64x2_t v3) {
//   return vfmsq_f64(v1, v2, v3);
// }

// NYI-LABEL: @test_vdivq_f64(
// NYI:   [[DIV_I:%.*]] = fdiv <2 x double> %v1, %v2
// NYI:   ret <2 x double> [[DIV_I]]
// float64x2_t test_vdivq_f64(float64x2_t v1, float64x2_t v2) {
//   return vdivq_f64(v1, v2);
// }

// NYI-LABEL: @test_vdivq_f32(
// NYI:   [[DIV_I:%.*]] = fdiv <4 x float> %v1, %v2
// NYI:   ret <4 x float> [[DIV_I]]
// float32x4_t test_vdivq_f32(float32x4_t v1, float32x4_t v2) {
//   return vdivq_f32(v1, v2);
// }

// NYI-LABEL: @test_vdiv_f32(
// NYI:   [[DIV_I:%.*]] = fdiv <2 x float> %v1, %v2
// NYI:   ret <2 x float> [[DIV_I]]
// float32x2_t test_vdiv_f32(float32x2_t v1, float32x2_t v2) {
//   return vdiv_f32(v1, v2);
// }

// NYI-LABEL: @test_vaba_s8(
// NYI:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %v2, <8 x i8> %v3)
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[VABD_I_I]]
// NYI:   ret <8 x i8> [[ADD_I]]
// int8x8_t test_vaba_s8(int8x8_t v1, int8x8_t v2, int8x8_t v3) {
//   return vaba_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vaba_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %v2, <4 x i16> %v3)
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[VABD2_I_I]]
// NYI:   ret <4 x i16> [[ADD_I]]
// int16x4_t test_vaba_s16(int16x4_t v1, int16x4_t v2, int16x4_t v3) {
//   return vaba_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vaba_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %v2, <2 x i32> %v3)
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[VABD2_I_I]]
// NYI:   ret <2 x i32> [[ADD_I]]
// int32x2_t test_vaba_s32(int32x2_t v1, int32x2_t v2, int32x2_t v3) {
//   return vaba_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vaba_u8(
// NYI:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %v2, <8 x i8> %v3)
// NYI:   [[ADD_I:%.*]] = add <8 x i8> %v1, [[VABD_I_I]]
// NYI:   ret <8 x i8> [[ADD_I]]
// uint8x8_t test_vaba_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
//   return vaba_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vaba_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %v2, <4 x i16> %v3)
// NYI:   [[ADD_I:%.*]] = add <4 x i16> %v1, [[VABD2_I_I]]
// NYI:   ret <4 x i16> [[ADD_I]]
// uint16x4_t test_vaba_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
//   return vaba_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vaba_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %v3 to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %v2, <2 x i32> %v3)
// NYI:   [[ADD_I:%.*]] = add <2 x i32> %v1, [[VABD2_I_I]]
// NYI:   ret <2 x i32> [[ADD_I]]
// uint32x2_t test_vaba_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
//   return vaba_u32(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_s8(
// NYI:   [[VABD_I_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> %v2, <16 x i8> %v3)
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[VABD_I_I]]
// NYI:   ret <16 x i8> [[ADD_I]]
// int8x16_t test_vabaq_s8(int8x16_t v1, int8x16_t v2, int8x16_t v3) {
//   return vabaq_s8(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> %v2, <8 x i16> %v3)
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[VABD2_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vabaq_s16(int16x8_t v1, int16x8_t v2, int16x8_t v3) {
//   return vabaq_s16(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> %v2, <4 x i32> %v3)
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[VABD2_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vabaq_s32(int32x4_t v1, int32x4_t v2, int32x4_t v3) {
//   return vabaq_s32(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_u8(
// NYI:   [[VABD_I_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> %v2, <16 x i8> %v3)
// NYI:   [[ADD_I:%.*]] = add <16 x i8> %v1, [[VABD_I_I]]
// NYI:   ret <16 x i8> [[ADD_I]]
// uint8x16_t test_vabaq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
//   return vabaq_u8(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> %v2, <8 x i16> %v3)
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %v1, [[VABD2_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vabaq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
//   return vabaq_u16(v1, v2, v3);
// }

// NYI-LABEL: @test_vabaq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %v3 to <16 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> %v2, <4 x i32> %v3)
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %v1, [[VABD2_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vabaq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
//   return vabaq_u32(v1, v2, v3);
// }

int8x8_t test_vabd_s8(int8x8_t v1, int8x8_t v2) {
  return vabd_s8(v1, v2);

  // CIR-LABEL: vabd_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>)

  // LLVM: {{.*}}test_vabd_s8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VABD_I]]
}

int16x4_t test_vabd_s16(int16x4_t v1, int16x4_t v2) {
  return vabd_s16(v1, v2);

  // CIR-LABEL: vabd_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>)

  // LLVM: {{.*}}test_vabd_s16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[VABD_I]]
}

int32x2_t test_vabd_s32(int32x2_t v1, int32x2_t v2) {
  return vabd_s32(v1, v2);

  // CIR-LABEL: vabd_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>)

  // LLVM: {{.*}}test_vabd_s32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[VABD_I]]
}

uint8x8_t test_vabd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vabd_u8(v1, v2);

  // CIR-LABEL: vabd_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>)

  // LLVM: {{.*}}test_vabd_u8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VABD_I]]
}

uint16x4_t test_vabd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vabd_u16(v1, v2);

  // CIR-LABEL: vabd_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>)

  // LLVM: {{.*}}test_vabd_u16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: ret <4 x i16> [[VABD_I]]
}

uint32x2_t test_vabd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vabd_u32(v1, v2);

  // CIR-LABEL: vabd_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>)

  // LLVM: {{.*}}test_vabd_u32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: ret <2 x i32> [[VABD_I]]
}

float32x2_t test_vabd_f32(float32x2_t v1, float32x2_t v2) {
  return vabd_f32(v1, v2);

  // CIR-LABEL: vabd_f32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.float x 2>, !cir.vector<!cir.float x 2>)

  // LLVM: {{.*}}test_vabd_f32(<2 x float>{{.*}}[[V1:%.*]], <2 x float>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_F:%.*]] = call <2 x float> @llvm.aarch64.neon.fabd.v2f32(<2 x float> [[V1]], <2 x float> [[V2]])
  // LLVM: ret <2 x float> [[VABD_F]]
}

int8x16_t test_vabdq_s8(int8x16_t v1, int8x16_t v2) {
  return vabdq_s8(v1, v2);

  // CIR-LABEL: vabdq_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>)

  // LLVM: {{.*}}test_vabdq_s8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sabd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VABD_I]]
}

int16x8_t test_vabdq_s16(int16x8_t v1, int16x8_t v2) {
  return vabdq_s16(v1, v2);

  // CIR-LABEL: vabdq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>)

  // LLVM: {{.*}}test_vabdq_s16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sabd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[VABD_I]]
}

int32x4_t test_vabdq_s32(int32x4_t v1, int32x4_t v2) {
  return vabdq_s32(v1, v2);

  // CIR-LABEL: vabdq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>)

  // LLVM: {{.*}}test_vabdq_s32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sabd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[VABD_I]]
}

uint8x16_t test_vabdq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vabdq_u8(v1, v2);

  // CIR-LABEL: vabdq_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>)

  // LLVM: {{.*}}test_vabdq_u8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uabd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VABD_I]]
}

uint16x8_t test_vabdq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vabdq_u16(v1, v2);

  // CIR-LABEL: vabdq_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>)

  // LLVM: {{.*}}test_vabdq_u16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uabd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: ret <8 x i16> [[VABD_I]]
}

uint32x4_t test_vabdq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vabdq_u32(v1, v2);

  // CIR-LABEL: vabdq_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>)

  // LLVM: {{.*}}test_vabdq_u32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uabd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: ret <4 x i32> [[VABD_I]]
}

float32x4_t test_vabdq_f32(float32x4_t v1, float32x4_t v2) {
  return vabdq_f32(v1, v2);

  // CIR-LABEL: vabdq_f32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.float x 4>, !cir.vector<!cir.float x 4>)

  // LLVM: {{.*}}test_vabdq_f32(<4 x float>{{.*}}[[V1:%.*]], <4 x float>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_F:%.*]] = call <4 x float> @llvm.aarch64.neon.fabd.v4f32(<4 x float> [[V1]], <4 x float> [[V2]])
  // LLVM: ret <4 x float> [[VABD_F]]
}

float64x2_t test_vabdq_f64(float64x2_t v1, float64x2_t v2) {
  return vabdq_f64(v1, v2);

  // CIR-LABEL: vabdq_f64
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fabd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>)

  // LLVM: {{.*}}test_vabdq_f64(<2 x double>{{.*}}[[V1:%.*]], <2 x double>{{.*}}[[V2:%.*]])
  // LLVM: [[VABD_F:%.*]] = call <2 x double> @llvm.aarch64.neon.fabd.v2f64(<2 x double> [[V1]], <2 x double> [[V2]])
  // LLVM: ret <2 x double> [[VABD_F]]
}

int8x8_t test_vbsl_s8(uint8x8_t v1, int8x8_t v2, int8x8_t v3) {
  return vbsl_s8(v1, v2, v3);

  // CIR-LABEL: vbsl_s8
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s8i x 8>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s8i x 8>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vbsl_s8(<8 x i8>{{.*}}[[v1:%.*]], <8 x i8>{{.*}}[[v2:%.*]], <8 x i8>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <8 x i8> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <8 x i8> [[v1]], splat (i8 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <8 x i8> [[VBSL2_I]]
}

int8x8_t test_vbsl_s16(uint16x4_t v1, int16x4_t v2, int16x4_t v3) {
  return (int8x8_t)vbsl_s16(v1, v2, v3);

  // CIR-LABEL: vbsl_s16
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s16i x 4>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s16i x 4>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vbsl_s16(<4 x i16>{{.*}}[[v1:%.*]], <4 x i16>{{.*}}[[v2:%.*]], <4 x i16>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <4 x i16> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <4 x i16> [[v1]], splat (i16 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   [[TMP4:%.*]] = bitcast <4 x i16> [[VBSL5_I]] to <8 x i8>
  // LLVM:   ret <8 x i8> [[TMP4]]
}

int32x2_t test_vbsl_s32(uint32x2_t v1, int32x2_t v2, int32x2_t v3) {
  return vbsl_s32(v1, v2, v3);

  // CIR-LABEL: vbsl_s32
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s32i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s32i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vbsl_s32(<2 x i32>{{.*}}[[v1:%.*]], <2 x i32>{{.*}}[[v2:%.*]], <2 x i32>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <2 x i32> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <2 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <2 x i32> [[VBSL5_I]]
}

int64x1_t test_vbsl_s64(uint64x1_t v1, int64x1_t v2, int64x1_t v3) {
  return vbsl_s64(v1, v2, v3);

  // CIR-LABEL: vbsl_s64
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s64i x 1>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s64i x 1>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s64i x 1>

  // LLVM: {{.*}}test_vbsl_s64(<1 x i64>{{.*}}[[v1:%.*]], <1 x i64>{{.*}}[[v2:%.*]], <1 x i64>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <1 x i64> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <1 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <1 x i64> [[VBSL5_I]]
}

uint8x8_t test_vbsl_u8(uint8x8_t v1, uint8x8_t v2, uint8x8_t v3) {
  return vbsl_u8(v1, v2, v3);

  // CIR-LABEL: vbsl_u8
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u8i x 8>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u8i x 8>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vbsl_u8(<8 x i8>{{.*}}[[v1:%.*]], <8 x i8>{{.*}}[[v2:%.*]], <8 x i8>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <8 x i8> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <8 x i8> [[v1]], splat (i8 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <8 x i8> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <8 x i8> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <8 x i8> [[VBSL5_I]]
}

uint16x4_t test_vbsl_u16(uint16x4_t v1, uint16x4_t v2, uint16x4_t v3) {
  return vbsl_u16(v1, v2, v3);

  // CIR-LABEL: vbsl_u16
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u16i x 4>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u16i x 4>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vbsl_u16(<4 x i16>{{.*}}[[v1:%.*]], <4 x i16>{{.*}}[[v2:%.*]], <4 x i16>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <4 x i16> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <4 x i16> [[v1]], splat (i16 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <4 x i16> [[VBSL5_I]]
}


uint32x2_t test_vbsl_u32(uint32x2_t v1, uint32x2_t v2, uint32x2_t v3) {
  return vbsl_u32(v1, v2, v3);

  // CIR-LABEL: vbsl_u32
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u32i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u32i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vbsl_u32(<2 x i32>{{.*}}[[v1:%.*]], <2 x i32>{{.*}}[[v2:%.*]], <2 x i32>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <2 x i32> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <2 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <2 x i32> [[VBSL5_I]]
}

uint64x1_t test_vbsl_u64(uint64x1_t v1, uint64x1_t v2, uint64x1_t v3) {
  return vbsl_u64(v1, v2, v3);

  // CIR-LABEL: vbsl_u64
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u64i x 1>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u64i x 1>, !cir.vector<!u64i x 1>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u64i x 1>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u64i x 1>

  // LLVM: {{.*}}test_vbsl_u64(<1 x i64>{{.*}}[[v1:%.*]], <1 x i64>{{.*}}[[v2:%.*]], <1 x i64>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL3_I:%.*]] = and <1 x i64> [[v1]], [[v2]]
  // LLVM:   [[TMP3:%.*]] = xor <1 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], [[v3]]
  // LLVM:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   ret <1 x i64> [[VBSL5_I]]
}

float32x2_t test_vbsl_f32(uint32x2_t v1, float32x2_t v2, float32x2_t v3) {
  return vbsl_f32(v1, v2, v3);

  // CIR-LABEL: vbsl_f32
  // CIR: [[v1:%.*]]  = cir.cast(bitcast, {{%.*}} : !cir.vector<!u32i x 2>), !cir.vector<!s8i x 8>
  // CIR: [[v2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.float x 2>), !cir.vector<!s8i x 8>
  // CIR: [[v3:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.float x 2>), !cir.vector<!s8i x 8>
  // CIR: [[v1_tmp:%.*]] = cir.cast(bitcast, [[v1]] : !cir.vector<!s8i x 8>), !cir.vector<!s32i x 2>
  // CIR: [[v2_tmp:%.*]] = cir.cast(bitcast, [[v2]] : !cir.vector<!s8i x 8>), !cir.vector<!s32i x 2>
  // CIR: [[v3_tmp:%.*]] = cir.cast(bitcast, [[v3]] : !cir.vector<!s8i x 8>), !cir.vector<!s32i x 2>
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1_tmp]], [[v2_tmp]]) : !cir.vector<!s32i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1_tmp]]) : !cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3_tmp]]) : !cir.vector<!s32i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s32i x 2>
  // CIR: cir.cast(bitcast, [[VBSL2_I]] : !cir.vector<!s32i x 2>), !cir.vector<!cir.float x 2>

  // LLVM: {{.*}}test_vbsl_f32(<2 x i32>{{.*}}[[v1:%.*]], <2 x float>{{.*}}[[v2:%.*]], <2 x float>{{.*}}[[v3:%.*]])
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[v1]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <2 x float> [[v2]] to <8 x i8>
  // LLVM:   [[TMP3:%.*]] = bitcast <2 x float> [[v3]] to <8 x i8>
  // LLVM:   [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <2 x i32>
  // LLVM:   [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <2 x i32>
  // LLVM:   [[VBSL3_I:%.*]] = and <2 x i32> [[v1]], [[VBSL1_I]]
  // LLVM:   [[TMP4:%.*]] = xor <2 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <2 x i32> [[TMP4]], [[VBSL2_I]]
  // LLVM:   [[VBSL5_I:%.*]] = or <2 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   [[TMP5:%.*]] = bitcast <2 x i32> [[VBSL5_I]] to <2 x float>
  // LLVM:   ret <2 x float> [[TMP5]]
}

float64x1_t test_vbsl_f64(uint64x1_t v1, float64x1_t v2, float64x1_t v3) {
  return vbsl_f64(v1, v2, v3);

  // CIR-LABEL: vbsl_f64
  // CIR: [[v1:%.*]]  = cir.cast(bitcast, {{%.*}} : !cir.vector<!u64i x 1>), !cir.vector<!s8i x 8>
  // CIR: [[v2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.double x 1>), !cir.vector<!s8i x 8>
  // CIR: [[v3:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.double x 1>), !cir.vector<!s8i x 8>
  // CIR: [[v1_tmp:%.*]] = cir.cast(bitcast, [[v1]] : !cir.vector<!s8i x 8>), !cir.vector<!s64i x 1>
  // CIR: [[v2_tmp:%.*]] = cir.cast(bitcast, [[v2]] : !cir.vector<!s8i x 8>), !cir.vector<!s64i x 1>
  // CIR: [[v3_tmp:%.*]] = cir.cast(bitcast, [[v3]] : !cir.vector<!s8i x 8>), !cir.vector<!s64i x 1>
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1_tmp]], [[v2_tmp]]) : !cir.vector<!s64i x 1>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1_tmp]]) : !cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3_tmp]]) : !cir.vector<!s64i x 1>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s64i x 1>
  // CIR: cir.cast(bitcast, [[VBSL2_I]] : !cir.vector<!s64i x 1>), !cir.vector<!cir.double x 1>

  // LLVM: {{.*}}test_vbsl_f64(<1 x i64>{{.*}}[[v1:%.*]], <1 x double>{{.*}}[[v2:%.*]], <1 x double>{{.*}}[[v3:%.*]])
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> [[v1]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <1 x double> [[v2]] to <8 x i8>
  // LLVM:   [[TMP3:%.*]] = bitcast <1 x double> [[v3]] to <8 x i8>
  // LLVM:   [[VBSL1_I:%.*]] = bitcast <8 x i8> [[TMP2]] to <1 x i64>
  // LLVM:   [[VBSL2_I:%.*]] = bitcast <8 x i8> [[TMP3]] to <1 x i64>
  // LLVM:   [[VBSL3_I:%.*]] = and <1 x i64> [[v1]], [[VBSL1_I]]
  // LLVM:   [[TMP4:%.*]] = xor <1 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP4]], [[VBSL2_I]]
  // LLVM:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   [[TMP5:%.*]] = bitcast <1 x i64> [[VBSL5_I]] to <1 x double>
  // LLVM:   ret <1 x double> [[TMP5]]
}

// NYI-LABEL: @test_vbsl_p8(
// NYI:   [[VBSL_I:%.*]] = and <8 x i8> %v1, %v2
// NYI:   [[TMP0:%.*]] = xor <8 x i8> %v1, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// NYI:   [[VBSL1_I:%.*]] = and <8 x i8> [[TMP0]], %v3
// NYI:   [[VBSL2_I:%.*]] = or <8 x i8> [[VBSL_I]], [[VBSL1_I]]
// NYI:   ret <8 x i8> [[VBSL2_I]]
// poly8x8_t test_vbsl_p8(uint8x8_t v1, poly8x8_t v2, poly8x8_t v3) {
//   return vbsl_p8(v1, v2, v3);
// }

// NYI-LABEL: @test_vbsl_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> %v3 to <8 x i8>
// NYI:   [[VBSL3_I:%.*]] = and <4 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = xor <4 x i16> %v1, <i16 -1, i16 -1, i16 -1, i16 -1>
// NYI:   [[VBSL4_I:%.*]] = and <4 x i16> [[TMP3]], %v3
// NYI:   [[VBSL5_I:%.*]] = or <4 x i16> [[VBSL3_I]], [[VBSL4_I]]
// NYI:   ret <4 x i16> [[VBSL5_I]]
// poly16x4_t test_vbsl_p16(uint16x4_t v1, poly16x4_t v2, poly16x4_t v3) {
//   return vbsl_p16(v1, v2, v3);
// }

int8x16_t test_vbslq_s8(uint8x16_t v1, int8x16_t v2, int8x16_t v3) {
  return vbslq_s8(v1, v2, v3);

  // CIR-LABEL: vbslq_s8
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s8i x 16>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s8i x 16>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vbslq_s8(<16 x i8>{{.*}}[[v1:%.*]], <16 x i8>{{.*}}[[v2:%.*]], <16 x i8>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <16 x i8> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <16 x i8> [[v1]], splat (i8 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <16 x i8> [[VBSL2_I]]
}

int16x8_t test_vbslq_s16(uint16x8_t v1, int16x8_t v2, int16x8_t v3) {
  return vbslq_s16(v1, v2, v3);

  // CIR-LABEL: vbslq_s16
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s16i x 8>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s16i x 8>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vbslq_s16(<8 x i16>{{.*}}[[v1:%.*]], <8 x i16>{{.*}}[[v2:%.*]], <8 x i16>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <8 x i16> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <8 x i16> [[v1]], splat (i16 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <8 x i16> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <8 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <8 x i16> [[VBSL2_I]]
}

int32x4_t test_vbslq_s32(uint32x4_t v1, int32x4_t v2, int32x4_t v3) {
  return vbslq_s32(v1, v2, v3);

  // CIR-LABEL: vbslq_s32
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s32i x 4>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s32i x 4>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vbslq_s32(<4 x i32>{{.*}}[[v1:%.*]], <4 x i32>{{.*}}[[v2:%.*]], <4 x i32>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <4 x i32> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <4 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <4 x i32> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <4 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <4 x i32> [[VBSL2_I]]
}

int64x2_t test_vbslq_s64(uint64x2_t v1, int64x2_t v2, int64x2_t v3) {
  return vbslq_s64(v1, v2, v3);

  // CIR-LABEL: vbslq_s64
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!s64i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!s64i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vbslq_s64(<2 x i64>{{.*}}[[v1:%.*]], <2 x i64>{{.*}}[[v2:%.*]], <2 x i64>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <2 x i64> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <2 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <2 x i64> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <2 x i64> [[VBSL2_I]]
}

uint8x16_t test_vbslq_u8(uint8x16_t v1, uint8x16_t v2, uint8x16_t v3) {
  return vbslq_u8(v1, v2, v3);

  // CIR-LABEL: vbslq_u8
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u8i x 16>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u8i x 16>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u8i x 16>

  // LLVM: {{.*}}test_vbslq_u8(<16 x i8>{{.*}}[[v1:%.*]], <16 x i8>{{.*}}[[v2:%.*]], <16 x i8>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <16 x i8> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <16 x i8> [[v1]], splat (i8 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <16 x i8> [[VBSL2_I]]
}

uint16x8_t test_vbslq_u16(uint16x8_t v1, uint16x8_t v2, uint16x8_t v3) {
  return vbslq_u16(v1, v2, v3);

  // CIR-LABEL: vbslq_u16
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u16i x 8>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u16i x 8>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vbslq_u16(<8 x i16>{{.*}}[[v1:%.*]], <8 x i16>{{.*}}[[v2:%.*]], <8 x i16>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <8 x i16> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <8 x i16> [[v1]], splat (i16 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <8 x i16> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <8 x i16> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <8 x i16> [[VBSL2_I]]
}

uint32x4_t test_vbslq_u32(uint32x4_t v1, uint32x4_t v2, uint32x4_t v3) {
  return vbslq_u32(v1, v2, v3);

  // CIR-LABEL: vbslq_u32
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u32i x 4>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u32i x 4>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vbslq_u32(<4 x i32>{{.*}}[[v1:%.*]], <4 x i32>{{.*}}[[v2:%.*]], <4 x i32>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <4 x i32> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <4 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <4 x i32> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <4 x i32> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <4 x i32> [[VBSL2_I]]
}

uint64x2_t test_vbslq_u64(uint64x2_t v1, uint64x2_t v2, uint64x2_t v3) {
  return vbslq_u64(v1, v2, v3);

  // CIR-LABEL: vbslq_u64
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1:%.*]], [[v2:%.*]]) : !cir.vector<!u64i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1]]) : !cir.vector<!u64i x 2>, !cir.vector<!u64i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3:%.*]]) : !cir.vector<!u64i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vbslq_u64(<2 x i64>{{.*}}[[v1:%.*]], <2 x i64>{{.*}}[[v2:%.*]], <2 x i64>{{.*}}[[v3:%.*]])
  // LLVM:   [[VBSL_I:%.*]] = and <2 x i64> [[v1]], [[v2]]
  // LLVM:   [[TMP0:%.*]] = xor <2 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL1_I:%.*]] = and <2 x i64> [[TMP0]], [[v3]]
  // LLVM:   [[VBSL2_I:%.*]] = or <2 x i64> [[VBSL_I]], [[VBSL1_I]]
  // LLVM:   ret <2 x i64> [[VBSL2_I]]
}

float32x4_t test_vbslq_f32(uint32x4_t v1, float32x4_t v2, float32x4_t v3) {
  return vbslq_f32(v1, v2, v3);

  // CIR-LABEL: vbslq_f32
  // CIR: [[v1:%.*]]  = cir.cast(bitcast, {{%.*}} : !cir.vector<!u32i x 4>), !cir.vector<!s8i x 16>
  // CIR: [[v2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.float x 4>), !cir.vector<!s8i x 16>
  // CIR: [[v3:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.float x 4>), !cir.vector<!s8i x 16>
  // CIR: [[v1_tmp:%.*]] = cir.cast(bitcast, [[v1]] : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[v2_tmp:%.*]] = cir.cast(bitcast, [[v2]] : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[v3_tmp:%.*]] = cir.cast(bitcast, [[v3]] : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1_tmp]], [[v2_tmp]]) : !cir.vector<!s32i x 4>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1_tmp]]) : !cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3_tmp]]) : !cir.vector<!s32i x 4>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s32i x 4>
  // CIR: cir.cast(bitcast, [[VBSL2_I]] : !cir.vector<!s32i x 4>), !cir.vector<!cir.float x 4>

  // LLVM: {{.*}}test_vbslq_f32(<4 x i32>{{.*}}[[v1:%.*]], <4 x float>{{.*}}[[v2:%.*]], <4 x float>{{.*}}[[v3:%.*]])
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[v1]] to <16 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <4 x float> [[v2]] to <16 x i8>
  // LLVM:   [[TMP3:%.*]] = bitcast <4 x float> [[v3]] to <16 x i8>
  // LLVM:   [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <4 x i32>
  // LLVM:   [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <4 x i32>
  // LLVM:   [[VBSL3_I:%.*]] = and <4 x i32> [[v1]], [[VBSL1_I]]
  // LLVM:   [[TMP4:%.*]] = xor <4 x i32> [[v1]], splat (i32 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <4 x i32> [[TMP4]], [[VBSL2_I]]
  // LLVM:   [[VBSL5_I:%.*]] = or <4 x i32> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   [[TMP5:%.*]] = bitcast <4 x i32> [[VBSL5_I]] to <4 x float>
  // LLVM:   ret <4 x float> [[TMP5]]
}

float64x2_t test_vbslq_f64(uint64x2_t v1, float64x2_t v2, float64x2_t v3) {
  return vbslq_f64(v1, v2, v3);

  // CIR-LABEL: vbslq_f64
  // CIR: [[v1:%.*]]  = cir.cast(bitcast, {{%.*}} : !cir.vector<!u64i x 2>), !cir.vector<!s8i x 16>
  // CIR: [[v2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.double x 2>), !cir.vector<!s8i x 16>
  // CIR: [[v3:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!cir.double x 2>), !cir.vector<!s8i x 16>
  // CIR: [[v1_tmp:%.*]] = cir.cast(bitcast, [[v1]] : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[v2_tmp:%.*]] = cir.cast(bitcast, [[v2]] : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[v3_tmp:%.*]] = cir.cast(bitcast, [[v3]] : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[VBSL_I:%.*]] = cir.binop(and, [[v1_tmp]], [[v2_tmp]]) : !cir.vector<!s64i x 2>
  // CIR: [[TMP0:%.*]] = cir.unary(not, [[v1_tmp]]) : !cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>
  // CIR: [[VBSL1_I:%.*]] = cir.binop(and, [[TMP0]], [[v3_tmp]]) : !cir.vector<!s64i x 2>
  // CIR: [[VBSL2_I:%.*]] = cir.binop(or, [[VBSL_I]], [[VBSL1_I]]) : !cir.vector<!s64i x 2>
  // CIR: cir.cast(bitcast, [[VBSL2_I]] : !cir.vector<!s64i x 2>), !cir.vector<!cir.double x 2>

  // LLVM: {{.*}}test_vbslq_f64(<2 x i64>{{.*}}[[v1:%.*]], <2 x double>{{.*}}[[v2:%.*]], <2 x double>{{.*}}[[v3:%.*]])
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> [[v1]] to <16 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <2 x double> [[v2]] to <16 x i8>
  // LLVM:   [[TMP3:%.*]] = bitcast <2 x double> [[v3]] to <16 x i8>
  // LLVM:   [[VBSL1_I:%.*]] = bitcast <16 x i8> [[TMP2]] to <2 x i64>
  // LLVM:   [[VBSL2_I:%.*]] = bitcast <16 x i8> [[TMP3]] to <2 x i64>
  // LLVM:   [[VBSL3_I:%.*]] = and <2 x i64> [[v1]], [[VBSL1_I]]
  // LLVM:   [[TMP4:%.*]] = xor <2 x i64> [[v1]], splat (i64 -1)
  // LLVM:   [[VBSL4_I:%.*]] = and <2 x i64> [[TMP4]], [[VBSL2_I]]
  // LLVM:   [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
  // LLVM:   [[TMP5:%.*]] = bitcast <2 x i64> [[VBSL5_I]] to <2 x double>
  // LLVM:   ret <2 x double> [[TMP5]]
}

// NYI-LABEL: @test_vbslq_p8(
// NYI:   [[VBSL_I:%.*]] = and <16 x i8> %v1, %v2
// NYI:   [[TMP0:%.*]] = xor <16 x i8> %v1, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// NYI:   [[VBSL1_I:%.*]] = and <16 x i8> [[TMP0]], %v3
// NYI:   [[VBSL2_I:%.*]] = or <16 x i8> [[VBSL_I]], [[VBSL1_I]]
// NYI:   ret <16 x i8> [[VBSL2_I]]
// poly8x16_t test_vbslq_p8(uint8x16_t v1, poly8x16_t v2, poly8x16_t v3) {
//   return vbslq_p8(v1, v2, v3);
// }

// NYI-LABEL: @test_vbslq_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <8 x i16> %v3 to <16 x i8>
// NYI:   [[VBSL3_I:%.*]] = and <8 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = xor <8 x i16> %v1, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
// NYI:   [[VBSL4_I:%.*]] = and <8 x i16> [[TMP3]], %v3
// NYI:   [[VBSL5_I:%.*]] = or <8 x i16> [[VBSL3_I]], [[VBSL4_I]]
// NYI:   ret <8 x i16> [[VBSL5_I]]
// poly16x8_t test_vbslq_p16(uint16x8_t v1, poly16x8_t v2, poly16x8_t v3) {
//   return vbslq_p16(v1, v2, v3);
// }

// NYI-LABEL: @test_vrecps_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VRECPS_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frecps.v2f32(<2 x float> %v1, <2 x float> %v2)
// NYI:   ret <2 x float> [[VRECPS_V2_I]]
// float32x2_t test_vrecps_f32(float32x2_t v1, float32x2_t v2) {
//   return vrecps_f32(v1, v2);
// }

// NYI-LABEL: @test_vrecpsq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VRECPSQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frecps.v4f32(<4 x float> %v1, <4 x float> %v2)
// NYI:   [[VRECPSQ_V3_I:%.*]] = bitcast <4 x float> [[VRECPSQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x float> [[VRECPSQ_V2_I]]
// float32x4_t test_vrecpsq_f32(float32x4_t v1, float32x4_t v2) {
//   return vrecpsq_f32(v1, v2);
// }

// NYI-LABEL: @test_vrecpsq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VRECPSQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frecps.v2f64(<2 x double> %v1, <2 x double> %v2)
// NYI:   [[VRECPSQ_V3_I:%.*]] = bitcast <2 x double> [[VRECPSQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x double> [[VRECPSQ_V2_I]]
// float64x2_t test_vrecpsq_f64(float64x2_t v1, float64x2_t v2) {
//   return vrecpsq_f64(v1, v2);
// }

// NYI-LABEL: @test_vrsqrts_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VRSQRTS_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.frsqrts.v2f32(<2 x float> %v1, <2 x float> %v2)
// NYI:   [[VRSQRTS_V3_I:%.*]] = bitcast <2 x float> [[VRSQRTS_V2_I]] to <8 x i8>
// NYI:   ret <2 x float> [[VRSQRTS_V2_I]]
// float32x2_t test_vrsqrts_f32(float32x2_t v1, float32x2_t v2) {
//   return vrsqrts_f32(v1, v2);
// }

// NYI-LABEL: @test_vrsqrtsq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VRSQRTSQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.frsqrts.v4f32(<4 x float> %v1, <4 x float> %v2)
// NYI:   [[VRSQRTSQ_V3_I:%.*]] = bitcast <4 x float> [[VRSQRTSQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x float> [[VRSQRTSQ_V2_I]]
// float32x4_t test_vrsqrtsq_f32(float32x4_t v1, float32x4_t v2) {
//   return vrsqrtsq_f32(v1, v2);
// }

// NYI-LABEL: @test_vrsqrtsq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VRSQRTSQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.frsqrts.v2f64(<2 x double> %v1, <2 x double> %v2)
// NYI:   [[VRSQRTSQ_V3_I:%.*]] = bitcast <2 x double> [[VRSQRTSQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x double> [[VRSQRTSQ_V2_I]]
// float64x2_t test_vrsqrtsq_f64(float64x2_t v1, float64x2_t v2) {
//   return vrsqrtsq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcage_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VCAGE_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float> %v1, <2 x float> %v2)
// NYI:   ret <2 x i32> [[VCAGE_V2_I]]
// uint32x2_t test_vcage_f32(float32x2_t v1, float32x2_t v2) {
//   return vcage_f32(v1, v2);
// }

// NYI-LABEL: @test_vcage_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VCAGE_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facge.v1i64.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x i64> [[VCAGE_V2_I]]
// uint64x1_t test_vcage_f64(float64x1_t a, float64x1_t b) {
//   return vcage_f64(a, b);
// }

// NYI-LABEL: @test_vcageq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VCAGEQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float> %v1, <4 x float> %v2)
// NYI:   ret <4 x i32> [[VCAGEQ_V2_I]]
// uint32x4_t test_vcageq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcageq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcageq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VCAGEQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double> %v1, <2 x double> %v2)
// NYI:   ret <2 x i64> [[VCAGEQ_V2_I]]
// uint64x2_t test_vcageq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcageq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcagt_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VCAGT_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float> %v1, <2 x float> %v2)
// NYI:   ret <2 x i32> [[VCAGT_V2_I]]
// uint32x2_t test_vcagt_f32(float32x2_t v1, float32x2_t v2) {
//   return vcagt_f32(v1, v2);
// }

// NYI-LABEL: @test_vcagt_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VCAGT_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facgt.v1i64.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x i64> [[VCAGT_V2_I]]
// uint64x1_t test_vcagt_f64(float64x1_t a, float64x1_t b) {
//   return vcagt_f64(a, b);
// }

// NYI-LABEL: @test_vcagtq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VCAGTQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float> %v1, <4 x float> %v2)
// NYI:   ret <4 x i32> [[VCAGTQ_V2_I]]
// uint32x4_t test_vcagtq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcagtq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcagtq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VCAGTQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double> %v1, <2 x double> %v2)
// NYI:   ret <2 x i64> [[VCAGTQ_V2_I]]
// uint64x2_t test_vcagtq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcagtq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcale_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VCALE_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facge.v2i32.v2f32(<2 x float> %v2, <2 x float> %v1)
// NYI:   ret <2 x i32> [[VCALE_V2_I]]
// uint32x2_t test_vcale_f32(float32x2_t v1, float32x2_t v2) {
//   return vcale_f32(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vcale_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VCALE_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facge.v1i64.v1f64(<1 x double> %b, <1 x double> %a)
// NYI:   ret <1 x i64> [[VCALE_V2_I]]
// uint64x1_t test_vcale_f64(float64x1_t a, float64x1_t b) {
//   return vcale_f64(a, b);
// }

// NYI-LABEL: @test_vcaleq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VCALEQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facge.v4i32.v4f32(<4 x float> %v2, <4 x float> %v1)
// NYI:   ret <4 x i32> [[VCALEQ_V2_I]]
// uint32x4_t test_vcaleq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcaleq_f32(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vcaleq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VCALEQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facge.v2i64.v2f64(<2 x double> %v2, <2 x double> %v1)
// NYI:   ret <2 x i64> [[VCALEQ_V2_I]]
// uint64x2_t test_vcaleq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcaleq_f64(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vcalt_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %v2 to <8 x i8>
// NYI:   [[VCALT_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.facgt.v2i32.v2f32(<2 x float> %v2, <2 x float> %v1)
// NYI:   ret <2 x i32> [[VCALT_V2_I]]
// uint32x2_t test_vcalt_f32(float32x2_t v1, float32x2_t v2) {
//   return vcalt_f32(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vcalt_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VCALT_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.facgt.v1i64.v1f64(<1 x double> %b, <1 x double> %a)
// NYI:   ret <1 x i64> [[VCALT_V2_I]]
// uint64x1_t test_vcalt_f64(float64x1_t a, float64x1_t b) {
//   return vcalt_f64(a, b);
// }

// NYI-LABEL: @test_vcaltq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %v2 to <16 x i8>
// NYI:   [[VCALTQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.facgt.v4i32.v4f32(<4 x float> %v2, <4 x float> %v1)
// NYI:   ret <4 x i32> [[VCALTQ_V2_I]]
// uint32x4_t test_vcaltq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcaltq_f32(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vcaltq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %v2 to <16 x i8>
// NYI:   [[VCALTQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.facgt.v2i64.v2f64(<2 x double> %v2, <2 x double> %v1)
// NYI:   ret <2 x i64> [[VCALTQ_V2_I]]
// uint64x2_t test_vcaltq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcaltq_f64(v1, v2);
//   // Using registers other than v0, v1 are possible, but would be odd.
// }

// NYI-LABEL: @test_vtst_s8(
// NYI:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// NYI:   ret <8 x i8> [[VTST_I]]
// uint8x8_t test_vtst_s8(int8x8_t v1, int8x8_t v2) {
//   return vtst_s8(v1, v2);
// }

// NYI-LABEL: @test_vtst_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// NYI:   ret <4 x i16> [[VTST_I]]
// uint16x4_t test_vtst_s16(int16x4_t v1, int16x4_t v2) {
//   return vtst_s16(v1, v2);
// }

// NYI-LABEL: @test_vtst_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <2 x i32> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <2 x i32> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i32>
// NYI:   ret <2 x i32> [[VTST_I]]
// uint32x2_t test_vtst_s32(int32x2_t v1, int32x2_t v2) {
//   return vtst_s32(v1, v2);
// }

// NYI-LABEL: @test_vtst_u8(
// NYI:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// NYI:   ret <8 x i8> [[VTST_I]]
// uint8x8_t test_vtst_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vtst_u8(v1, v2);
// }

// NYI-LABEL: @test_vtst_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// NYI:   ret <4 x i16> [[VTST_I]]
// uint16x4_t test_vtst_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vtst_u16(v1, v2);
// }

// NYI-LABEL: @test_vtst_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <2 x i32> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <2 x i32> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i32>
// NYI:   ret <2 x i32> [[VTST_I]]
// uint32x2_t test_vtst_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vtst_u32(v1, v2);
// }

// NYI-LABEL: @test_vtstq_s8(
// NYI:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// NYI:   ret <16 x i8> [[VTST_I]]
// uint8x16_t test_vtstq_s8(int8x16_t v1, int8x16_t v2) {
//   return vtstq_s8(v1, v2);
// }

// NYI-LABEL: @test_vtstq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// NYI:   ret <8 x i16> [[VTST_I]]
// uint16x8_t test_vtstq_s16(int16x8_t v1, int16x8_t v2) {
//   return vtstq_s16(v1, v2);
// }

// NYI-LABEL: @test_vtstq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <4 x i32> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <4 x i32> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i32>
// NYI:   ret <4 x i32> [[VTST_I]]
// uint32x4_t test_vtstq_s32(int32x4_t v1, int32x4_t v2) {
//   return vtstq_s32(v1, v2);
// }

// NYI-LABEL: @test_vtstq_u8(
// NYI:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// NYI:   ret <16 x i8> [[VTST_I]]
// uint8x16_t test_vtstq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vtstq_u8(v1, v2);
// }

// NYI-LABEL: @test_vtstq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// NYI:   ret <8 x i16> [[VTST_I]]
// uint16x8_t test_vtstq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vtstq_u16(v1, v2);
// }

// NYI-LABEL: @test_vtstq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <4 x i32> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <4 x i32> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i32>
// NYI:   ret <4 x i32> [[VTST_I]]
// uint32x4_t test_vtstq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vtstq_u32(v1, v2);
// }

// NYI-LABEL: @test_vtstq_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <2 x i64> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <2 x i64> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i64>
// NYI:   ret <2 x i64> [[VTST_I]]
// uint64x2_t test_vtstq_s64(int64x2_t v1, int64x2_t v2) {
//   return vtstq_s64(v1, v2);
// }

// NYI-LABEL: @test_vtstq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <2 x i64> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <2 x i64> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP3]] to <2 x i64>
// NYI:   ret <2 x i64> [[VTST_I]]
// uint64x2_t test_vtstq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vtstq_u64(v1, v2);
// }

// NYI-LABEL: @test_vtst_p8(
// NYI:   [[TMP0:%.*]] = and <8 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <8 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP1]] to <8 x i8>
// NYI:   ret <8 x i8> [[VTST_I]]
// uint8x8_t test_vtst_p8(poly8x8_t v1, poly8x8_t v2) {
//   return vtst_p8(v1, v2);
// }

// NYI-LABEL: @test_vtst_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %v1 to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %v2 to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <4 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <4 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <4 x i1> [[TMP3]] to <4 x i16>
// NYI:   ret <4 x i16> [[VTST_I]]
// uint16x4_t test_vtst_p16(poly16x4_t v1, poly16x4_t v2) {
//   return vtst_p16(v1, v2);
// }

// NYI-LABEL: @test_vtstq_p8(
// NYI:   [[TMP0:%.*]] = and <16 x i8> %v1, %v2
// NYI:   [[TMP1:%.*]] = icmp ne <16 x i8> [[TMP0]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <16 x i1> [[TMP1]] to <16 x i8>
// NYI:   ret <16 x i8> [[VTST_I]]
// uint8x16_t test_vtstq_p8(poly8x16_t v1, poly8x16_t v2) {
//   return vtstq_p8(v1, v2);
// }

// NYI-LABEL: @test_vtstq_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %v1 to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %v2 to <16 x i8>
// NYI:   [[TMP2:%.*]] = and <8 x i16> %v1, %v2
// NYI:   [[TMP3:%.*]] = icmp ne <8 x i16> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <8 x i1> [[TMP3]] to <8 x i16>
// NYI:   ret <8 x i16> [[VTST_I]]
// uint16x8_t test_vtstq_p16(poly16x8_t v1, poly16x8_t v2) {
//   return vtstq_p16(v1, v2);
// }

// NYI-LABEL: @test_vtst_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <1 x i64> %a, %b
// NYI:   [[TMP3:%.*]] = icmp ne <1 x i64> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <1 x i1> [[TMP3]] to <1 x i64>
// NYI:   ret <1 x i64> [[VTST_I]]
// uint64x1_t test_vtst_s64(int64x1_t a, int64x1_t b) {
//   return vtst_s64(a, b);
// }

// NYI-LABEL: @test_vtst_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[TMP2:%.*]] = and <1 x i64> %a, %b
// NYI:   [[TMP3:%.*]] = icmp ne <1 x i64> [[TMP2]], zeroinitializer
// NYI:   [[VTST_I:%.*]] = sext <1 x i1> [[TMP3]] to <1 x i64>
// NYI:   ret <1 x i64> [[VTST_I]]
// uint64x1_t test_vtst_u64(uint64x1_t a, uint64x1_t b) {
//   return vtst_u64(a, b);
// }

// NYI-LABEL: @test_vceq_s8(
// NYI:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vceq_s8(int8x8_t v1, int8x8_t v2) {
//   return vceq_s8(v1, v2);
// }

// NYI-LABEL: @test_vceq_s16(
// NYI:   [[CMP_I:%.*]] = icmp eq <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vceq_s16(int16x4_t v1, int16x4_t v2) {
//   return vceq_s16(v1, v2);
// }

// NYI-LABEL: @test_vceq_s32(
// NYI:   [[CMP_I:%.*]] = icmp eq <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vceq_s32(int32x2_t v1, int32x2_t v2) {
//   return vceq_s32(v1, v2);
// }

// NYI-LABEL: @test_vceq_s64(
// NYI:   [[CMP_I:%.*]] = icmp eq <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vceq_s64(int64x1_t a, int64x1_t b) {
//   return vceq_s64(a, b);
// }

// NYI-LABEL: @test_vceq_u64(
// NYI:   [[CMP_I:%.*]] = icmp eq <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vceq_u64(uint64x1_t a, uint64x1_t b) {
//   return vceq_u64(a, b);
// }

// NYI-LABEL: @test_vceq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp oeq <2 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vceq_f32(float32x2_t v1, float32x2_t v2) {
//   return vceq_f32(v1, v2);
// }

// NYI-LABEL: @test_vceq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp oeq <1 x double> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vceq_f64(float64x1_t a, float64x1_t b) {
//   return vceq_f64(a, b);
// }

// NYI-LABEL: @test_vceq_u8(
// NYI:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vceq_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vceq_u8(v1, v2);
// }

// NYI-LABEL: @test_vceq_u16(
// NYI:   [[CMP_I:%.*]] = icmp eq <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vceq_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vceq_u16(v1, v2);
// }

// NYI-LABEL: @test_vceq_u32(
// NYI:   [[CMP_I:%.*]] = icmp eq <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vceq_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vceq_u32(v1, v2);
// }

// NYI-LABEL: @test_vceq_p8(
// NYI:   [[CMP_I:%.*]] = icmp eq <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vceq_p8(poly8x8_t v1, poly8x8_t v2) {
//   return vceq_p8(v1, v2);
// }

// NYI-LABEL: @test_vceqq_s8(
// NYI:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vceqq_s8(int8x16_t v1, int8x16_t v2) {
//   return vceqq_s8(v1, v2);
// }

// NYI-LABEL: @test_vceqq_s16(
// NYI:   [[CMP_I:%.*]] = icmp eq <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vceqq_s16(int16x8_t v1, int16x8_t v2) {
//   return vceqq_s16(v1, v2);
// }

// NYI-LABEL: @test_vceqq_s32(
// NYI:   [[CMP_I:%.*]] = icmp eq <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vceqq_s32(int32x4_t v1, int32x4_t v2) {
//   return vceqq_s32(v1, v2);
// }

// NYI-LABEL: @test_vceqq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp oeq <4 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vceqq_f32(float32x4_t v1, float32x4_t v2) {
//   return vceqq_f32(v1, v2);
// }

// NYI-LABEL: @test_vceqq_u8(
// NYI:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vceqq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vceqq_u8(v1, v2);
// }

// NYI-LABEL: @test_vceqq_u16(
// NYI:   [[CMP_I:%.*]] = icmp eq <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vceqq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vceqq_u16(v1, v2);
// }

// NYI-LABEL: @test_vceqq_u32(
// NYI:   [[CMP_I:%.*]] = icmp eq <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vceqq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vceqq_u32(v1, v2);
// }

// NYI-LABEL: @test_vceqq_p8(
// NYI:   [[CMP_I:%.*]] = icmp eq <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vceqq_p8(poly8x16_t v1, poly8x16_t v2) {
//   return vceqq_p8(v1, v2);
// }

// NYI-LABEL: @test_vceqq_s64(
// NYI:   [[CMP_I:%.*]] = icmp eq <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vceqq_s64(int64x2_t v1, int64x2_t v2) {
//   return vceqq_s64(v1, v2);
// }

// NYI-LABEL: @test_vceqq_u64(
// NYI:   [[CMP_I:%.*]] = icmp eq <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vceqq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vceqq_u64(v1, v2);
// }

// NYI-LABEL: @test_vceqq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp oeq <2 x double> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vceqq_f64(float64x2_t v1, float64x2_t v2) {
//   return vceqq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcge_s8(
// NYI:   [[CMP_I:%.*]] = icmp sge <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vcge_s8(int8x8_t v1, int8x8_t v2) {
//   return vcge_s8(v1, v2);
// }

// NYI-LABEL: @test_vcge_s16(
// NYI:   [[CMP_I:%.*]] = icmp sge <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcge_s16(int16x4_t v1, int16x4_t v2) {
//   return vcge_s16(v1, v2);
// }

// NYI-LABEL: @test_vcge_s32(
// NYI:   [[CMP_I:%.*]] = icmp sge <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcge_s32(int32x2_t v1, int32x2_t v2) {
//   return vcge_s32(v1, v2);
// }

// NYI-LABEL: @test_vcge_s64(
// NYI:   [[CMP_I:%.*]] = icmp sge <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcge_s64(int64x1_t a, int64x1_t b) {
//   return vcge_s64(a, b);
// }

// NYI-LABEL: @test_vcge_u64(
// NYI:   [[CMP_I:%.*]] = icmp uge <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcge_u64(uint64x1_t a, uint64x1_t b) {
//   return vcge_u64(a, b);
// }

// NYI-LABEL: @test_vcge_f32(
// NYI:   [[CMP_I:%.*]] = fcmp oge <2 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcge_f32(float32x2_t v1, float32x2_t v2) {
//   return vcge_f32(v1, v2);
// }

// NYI-LABEL: @test_vcge_f64(
// NYI:   [[CMP_I:%.*]] = fcmp oge <1 x double> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcge_f64(float64x1_t a, float64x1_t b) {
//   return vcge_f64(a, b);
// }

// NYI-LABEL: @test_vcge_u8(
// NYI:   [[CMP_I:%.*]] = icmp uge <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vcge_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vcge_u8(v1, v2);
// }

// NYI-LABEL: @test_vcge_u16(
// NYI:   [[CMP_I:%.*]] = icmp uge <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcge_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vcge_u16(v1, v2);
// }

// NYI-LABEL: @test_vcge_u32(
// NYI:   [[CMP_I:%.*]] = icmp uge <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcge_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vcge_u32(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_s8(
// NYI:   [[CMP_I:%.*]] = icmp sge <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcgeq_s8(int8x16_t v1, int8x16_t v2) {
//   return vcgeq_s8(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_s16(
// NYI:   [[CMP_I:%.*]] = icmp sge <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcgeq_s16(int16x8_t v1, int16x8_t v2) {
//   return vcgeq_s16(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_s32(
// NYI:   [[CMP_I:%.*]] = icmp sge <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgeq_s32(int32x4_t v1, int32x4_t v2) {
//   return vcgeq_s32(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp oge <4 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgeq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcgeq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_u8(
// NYI:   [[CMP_I:%.*]] = icmp uge <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcgeq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vcgeq_u8(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_u16(
// NYI:   [[CMP_I:%.*]] = icmp uge <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcgeq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vcgeq_u16(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_u32(
// NYI:   [[CMP_I:%.*]] = icmp uge <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgeq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vcgeq_u32(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_s64(
// NYI:   [[CMP_I:%.*]] = icmp sge <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgeq_s64(int64x2_t v1, int64x2_t v2) {
//   return vcgeq_s64(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_u64(
// NYI:   [[CMP_I:%.*]] = icmp uge <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgeq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vcgeq_u64(v1, v2);
// }

// NYI-LABEL: @test_vcgeq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp oge <2 x double> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgeq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcgeq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcle_s8(
// NYI:   [[CMP_I:%.*]] = icmp sle <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// Notes about vcle:
// LE condition predicate implemented as GE, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
// uint8x8_t test_vcle_s8(int8x8_t v1, int8x8_t v2) {
//   return vcle_s8(v1, v2);
// }

// NYI-LABEL: @test_vcle_s16(
// NYI:   [[CMP_I:%.*]] = icmp sle <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcle_s16(int16x4_t v1, int16x4_t v2) {
//   return vcle_s16(v1, v2);
// }

// NYI-LABEL: @test_vcle_s32(
// NYI:   [[CMP_I:%.*]] = icmp sle <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcle_s32(int32x2_t v1, int32x2_t v2) {
//   return vcle_s32(v1, v2);
// }

// NYI-LABEL: @test_vcle_s64(
// NYI:   [[CMP_I:%.*]] = icmp sle <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcle_s64(int64x1_t a, int64x1_t b) {
//   return vcle_s64(a, b);
// }

// NYI-LABEL: @test_vcle_u64(
// NYI:   [[CMP_I:%.*]] = icmp ule <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcle_u64(uint64x1_t a, uint64x1_t b) {
//   return vcle_u64(a, b);
// }

// NYI-LABEL: @test_vcle_f32(
// NYI:   [[CMP_I:%.*]] = fcmp ole <2 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcle_f32(float32x2_t v1, float32x2_t v2) {
//   return vcle_f32(v1, v2);
// }

// NYI-LABEL: @test_vcle_f64(
// NYI:   [[CMP_I:%.*]] = fcmp ole <1 x double> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcle_f64(float64x1_t a, float64x1_t b) {
//   return vcle_f64(a, b);
// }

// NYI-LABEL: @test_vcle_u8(
// NYI:   [[CMP_I:%.*]] = icmp ule <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vcle_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vcle_u8(v1, v2);
// }

// NYI-LABEL: @test_vcle_u16(
// NYI:   [[CMP_I:%.*]] = icmp ule <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcle_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vcle_u16(v1, v2);
// }

// NYI-LABEL: @test_vcle_u32(
// NYI:   [[CMP_I:%.*]] = icmp ule <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcle_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vcle_u32(v1, v2);
// }

// NYI-LABEL: @test_vcleq_s8(
// NYI:   [[CMP_I:%.*]] = icmp sle <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcleq_s8(int8x16_t v1, int8x16_t v2) {
//   return vcleq_s8(v1, v2);
// }

// NYI-LABEL: @test_vcleq_s16(
// NYI:   [[CMP_I:%.*]] = icmp sle <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcleq_s16(int16x8_t v1, int16x8_t v2) {
//   return vcleq_s16(v1, v2);
// }

// NYI-LABEL: @test_vcleq_s32(
// NYI:   [[CMP_I:%.*]] = icmp sle <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcleq_s32(int32x4_t v1, int32x4_t v2) {
//   return vcleq_s32(v1, v2);
// }

// NYI-LABEL: @test_vcleq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp ole <4 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcleq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcleq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcleq_u8(
// NYI:   [[CMP_I:%.*]] = icmp ule <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcleq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vcleq_u8(v1, v2);
// }

// NYI-LABEL: @test_vcleq_u16(
// NYI:   [[CMP_I:%.*]] = icmp ule <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcleq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vcleq_u16(v1, v2);
// }

// NYI-LABEL: @test_vcleq_u32(
// NYI:   [[CMP_I:%.*]] = icmp ule <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcleq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vcleq_u32(v1, v2);
// }

// NYI-LABEL: @test_vcleq_s64(
// NYI:   [[CMP_I:%.*]] = icmp sle <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcleq_s64(int64x2_t v1, int64x2_t v2) {
//   return vcleq_s64(v1, v2);
// }

// NYI-LABEL: @test_vcleq_u64(
// NYI:   [[CMP_I:%.*]] = icmp ule <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcleq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vcleq_u64(v1, v2);
// }

// NYI-LABEL: @test_vcleq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp ole <2 x double> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcleq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcleq_f64(v1, v2);
// }

// NYI-LABEL: @test_vcgt_s8(
// NYI:   [[CMP_I:%.*]] = icmp sgt <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vcgt_s8(int8x8_t v1, int8x8_t v2) {
//   return vcgt_s8(v1, v2);
// }

// NYI-LABEL: @test_vcgt_s16(
// NYI:   [[CMP_I:%.*]] = icmp sgt <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcgt_s16(int16x4_t v1, int16x4_t v2) {
//   return vcgt_s16(v1, v2);
// }

// NYI-LABEL: @test_vcgt_s32(
// NYI:   [[CMP_I:%.*]] = icmp sgt <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcgt_s32(int32x2_t v1, int32x2_t v2) {
//   return vcgt_s32(v1, v2);
// }

// NYI-LABEL: @test_vcgt_s64(
// NYI:   [[CMP_I:%.*]] = icmp sgt <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcgt_s64(int64x1_t a, int64x1_t b) {
//   return vcgt_s64(a, b);
// }

// NYI-LABEL: @test_vcgt_u64(
// NYI:   [[CMP_I:%.*]] = icmp ugt <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcgt_u64(uint64x1_t a, uint64x1_t b) {
//   return vcgt_u64(a, b);
// }

// NYI-LABEL: @test_vcgt_f32(
// NYI:   [[CMP_I:%.*]] = fcmp ogt <2 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcgt_f32(float32x2_t v1, float32x2_t v2) {
//   return vcgt_f32(v1, v2);
// }

// NYI-LABEL: @test_vcgt_f64(
// NYI:   [[CMP_I:%.*]] = fcmp ogt <1 x double> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vcgt_f64(float64x1_t a, float64x1_t b) {
//   return vcgt_f64(a, b);
// }

// NYI-LABEL: @test_vcgt_u8(
// NYI:   [[CMP_I:%.*]] = icmp ugt <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vcgt_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vcgt_u8(v1, v2);
// }

// NYI-LABEL: @test_vcgt_u16(
// NYI:   [[CMP_I:%.*]] = icmp ugt <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vcgt_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vcgt_u16(v1, v2);
// }

// NYI-LABEL: @test_vcgt_u32(
// NYI:   [[CMP_I:%.*]] = icmp ugt <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vcgt_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vcgt_u32(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_s8(
// NYI:   [[CMP_I:%.*]] = icmp sgt <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcgtq_s8(int8x16_t v1, int8x16_t v2) {
//   return vcgtq_s8(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_s16(
// NYI:   [[CMP_I:%.*]] = icmp sgt <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcgtq_s16(int16x8_t v1, int16x8_t v2) {
//   return vcgtq_s16(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_s32(
// NYI:   [[CMP_I:%.*]] = icmp sgt <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgtq_s32(int32x4_t v1, int32x4_t v2) {
//   return vcgtq_s32(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp ogt <4 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgtq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcgtq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_u8(
// NYI:   [[CMP_I:%.*]] = icmp ugt <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcgtq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vcgtq_u8(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_u16(
// NYI:   [[CMP_I:%.*]] = icmp ugt <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcgtq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vcgtq_u16(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_u32(
// NYI:   [[CMP_I:%.*]] = icmp ugt <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcgtq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vcgtq_u32(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_s64(
// NYI:   [[CMP_I:%.*]] = icmp sgt <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgtq_s64(int64x2_t v1, int64x2_t v2) {
//   return vcgtq_s64(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_u64(
// NYI:   [[CMP_I:%.*]] = icmp ugt <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgtq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vcgtq_u64(v1, v2);
// }

// NYI-LABEL: @test_vcgtq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp ogt <2 x double> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcgtq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcgtq_f64(v1, v2);
// }

// NYI-LABEL: @test_vclt_s8(
// NYI:   [[CMP_I:%.*]] = icmp slt <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// Notes about vclt:
// LT condition predicate implemented as GT, so check reversed operands.
// Using registers other than v0, v1 are possible, but would be odd.
// uint8x8_t test_vclt_s8(int8x8_t v1, int8x8_t v2) {
//   return vclt_s8(v1, v2);
// }

// NYI-LABEL: @test_vclt_s16(
// NYI:   [[CMP_I:%.*]] = icmp slt <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vclt_s16(int16x4_t v1, int16x4_t v2) {
//   return vclt_s16(v1, v2);
// }

// NYI-LABEL: @test_vclt_s32(
// NYI:   [[CMP_I:%.*]] = icmp slt <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vclt_s32(int32x2_t v1, int32x2_t v2) {
//   return vclt_s32(v1, v2);
// }

// NYI-LABEL: @test_vclt_s64(
// NYI:   [[CMP_I:%.*]] = icmp slt <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vclt_s64(int64x1_t a, int64x1_t b) {
//   return vclt_s64(a, b);
// }

// NYI-LABEL: @test_vclt_u64(
// NYI:   [[CMP_I:%.*]] = icmp ult <1 x i64> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vclt_u64(uint64x1_t a, uint64x1_t b) {
//   return vclt_u64(a, b);
// }

// NYI-LABEL: @test_vclt_f32(
// NYI:   [[CMP_I:%.*]] = fcmp olt <2 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vclt_f32(float32x2_t v1, float32x2_t v2) {
//   return vclt_f32(v1, v2);
// }

// NYI-LABEL: @test_vclt_f64(
// NYI:   [[CMP_I:%.*]] = fcmp olt <1 x double> %a, %b
// NYI:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// NYI:   ret <1 x i64> [[SEXT_I]]
// uint64x1_t test_vclt_f64(float64x1_t a, float64x1_t b) {
//   return vclt_f64(a, b);
// }

// NYI-LABEL: @test_vclt_u8(
// NYI:   [[CMP_I:%.*]] = icmp ult <8 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[SEXT_I]]
// uint8x8_t test_vclt_u8(uint8x8_t v1, uint8x8_t v2) {
//   return vclt_u8(v1, v2);
// }

// NYI-LABEL: @test_vclt_u16(
// NYI:   [[CMP_I:%.*]] = icmp ult <4 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[SEXT_I]]
// uint16x4_t test_vclt_u16(uint16x4_t v1, uint16x4_t v2) {
//   return vclt_u16(v1, v2);
// }

// NYI-LABEL: @test_vclt_u32(
// NYI:   [[CMP_I:%.*]] = icmp ult <2 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[SEXT_I]]
// uint32x2_t test_vclt_u32(uint32x2_t v1, uint32x2_t v2) {
//   return vclt_u32(v1, v2);
// }

// NYI-LABEL: @test_vcltq_s8(
// NYI:   [[CMP_I:%.*]] = icmp slt <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcltq_s8(int8x16_t v1, int8x16_t v2) {
//   return vcltq_s8(v1, v2);
// }

// NYI-LABEL: @test_vcltq_s16(
// NYI:   [[CMP_I:%.*]] = icmp slt <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcltq_s16(int16x8_t v1, int16x8_t v2) {
//   return vcltq_s16(v1, v2);
// }

// NYI-LABEL: @test_vcltq_s32(
// NYI:   [[CMP_I:%.*]] = icmp slt <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcltq_s32(int32x4_t v1, int32x4_t v2) {
//   return vcltq_s32(v1, v2);
// }

// NYI-LABEL: @test_vcltq_f32(
// NYI:   [[CMP_I:%.*]] = fcmp olt <4 x float> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcltq_f32(float32x4_t v1, float32x4_t v2) {
//   return vcltq_f32(v1, v2);
// }

// NYI-LABEL: @test_vcltq_u8(
// NYI:   [[CMP_I:%.*]] = icmp ult <16 x i8> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <16 x i1> [[CMP_I]] to <16 x i8>
// NYI:   ret <16 x i8> [[SEXT_I]]
// uint8x16_t test_vcltq_u8(uint8x16_t v1, uint8x16_t v2) {
//   return vcltq_u8(v1, v2);
// }

// NYI-LABEL: @test_vcltq_u16(
// NYI:   [[CMP_I:%.*]] = icmp ult <8 x i16> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <8 x i1> [[CMP_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[SEXT_I]]
// uint16x8_t test_vcltq_u16(uint16x8_t v1, uint16x8_t v2) {
//   return vcltq_u16(v1, v2);
// }

// NYI-LABEL: @test_vcltq_u32(
// NYI:   [[CMP_I:%.*]] = icmp ult <4 x i32> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <4 x i1> [[CMP_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[SEXT_I]]
// uint32x4_t test_vcltq_u32(uint32x4_t v1, uint32x4_t v2) {
//   return vcltq_u32(v1, v2);
// }

// NYI-LABEL: @test_vcltq_s64(
// NYI:   [[CMP_I:%.*]] = icmp slt <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcltq_s64(int64x2_t v1, int64x2_t v2) {
//   return vcltq_s64(v1, v2);
// }

// NYI-LABEL: @test_vcltq_u64(
// NYI:   [[CMP_I:%.*]] = icmp ult <2 x i64> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcltq_u64(uint64x2_t v1, uint64x2_t v2) {
//   return vcltq_u64(v1, v2);
// }

// NYI-LABEL: @test_vcltq_f64(
// NYI:   [[CMP_I:%.*]] = fcmp olt <2 x double> %v1, %v2
// NYI:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[SEXT_I]]
// uint64x2_t test_vcltq_f64(float64x2_t v1, float64x2_t v2) {
//   return vcltq_f64(v1, v2);
// }

int8x8_t test_vhadd_s8(int8x8_t v1, int8x8_t v2) {
  return vhadd_s8(v1, v2);

  // CIR-LABEL: vhadd_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vhadd_s8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.shadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VHADD_V_I]]
}

int16x4_t test_vhadd_s16(int16x4_t v1, int16x4_t v2) {
  return vhadd_s16(v1, v2);

  // CIR-LABEL: vhadd_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vhadd_s16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.shadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <4 x i16> [[VHADD_V2_I]] to <8 x i8>
  // LLVM:  ret <4 x i16> [[VHADD_V2_I]]
}

int32x2_t test_vhadd_s32(int32x2_t v1, int32x2_t v2) {
  return vhadd_s32(v1, v2);

  // CIR-LABEL: vhadd_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vhadd_s32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.shadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <2 x i32> [[VHADD_V2_I]] to <8 x i8>
  // LLVM:  ret <2 x i32> [[VHADD_V2_I]]
}

uint8x8_t test_vhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vhadd_u8(v1, v2);

  // CIR-LABEL: vhadd_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vhadd_u8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VHADD_V_I]]
}

uint16x4_t test_vhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vhadd_u16(v1, v2);

  // CIR-LABEL: vhadd_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vhadd_u16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <4 x i16> [[VHADD_V2_I]] to <8 x i8>
  // LLVM:  ret <4 x i16> [[VHADD_V2_I]]
}

uint32x2_t test_vhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vhadd_u32(v1, v2);

  // CIR-LABEL: vhadd_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vhadd_u32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <2 x i32> [[VHADD_V2_I]] to <8 x i8>
  // LLVM:  ret <2 x i32> [[VHADD_V2_I]]
}

int8x16_t test_vhaddq_s8(int8x16_t v1, int8x16_t v2) {
  return vhaddq_s8(v1, v2);

  // CIR-LABEL: vhaddq_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vhaddq_s8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VHADD_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.shadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VHADD_V_I]]
}

int16x8_t test_vhaddq_s16(int16x8_t v1, int16x8_t v2) {
  return vhaddq_s16(v1, v2);

  // CIR-LABEL: vhaddq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vhaddq_s16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.shadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <8 x i16> [[VHADD_V2_I]] to <16 x i8>
  // LLVM:  ret <8 x i16> [[VHADD_V2_I]]
}

int32x4_t test_vhaddq_s32(int32x4_t v1, int32x4_t v2) {
  return vhaddq_s32(v1, v2);

  // CIR-LABEL: vhaddq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vhaddq_s32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.shadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <4 x i32> [[VHADD_V2_I]] to <16 x i8>
  // LLVM:  ret <4 x i32> [[VHADD_V2_I]]
}

uint8x16_t test_vhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vhaddq_u8(v1, v2);

  // CIR-LABEL: vhaddq_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}test_vhaddq_u8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VHADD_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VHADD_V_I]]
}

uint16x8_t test_vhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vhaddq_u16(v1, v2);

  // CIR-LABEL: vhaddq_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vhaddq_u16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <8 x i16> [[VHADD_V2_I]] to <16 x i8>
  // LLVM:  ret <8 x i16> [[VHADD_V2_I]]
}

uint32x4_t test_vhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vhaddq_u32(v1, v2);

  // CIR-LABEL: vhaddq_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vhaddq_u32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM:  [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM:  [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM:  [[VHADD_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM:  [[VHADD_V3_I:%.*]] = bitcast <4 x i32> [[VHADD_V2_I]] to <16 x i8>
  // LLVM:  ret <4 x i32> [[VHADD_V2_I]]
}

int8x8_t test_vhsub_s8(int8x8_t v1, int8x8_t v2) {
  return vhsub_s8(v1, v2);

  // CIR-LABEL: vhsub_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vhsub_s8(<8 x i8>{{.*}}[[v1:%.*]], <8 x i8>{{.*}}[[v2:%.*]])
  // LLVM:   [[VHSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.shsub.v8i8(<8 x i8> [[v1]], <8 x i8> [[v2]])
  // LLVM:   ret <8 x i8> [[VHSUB_V_I]]
}

int16x4_t test_vhsub_s16(int16x4_t v1, int16x4_t v2) {
  return vhsub_s16(v1, v2);

  // CIR-LABEL: vhsub_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vhsub_s16(<4 x i16>{{.*}}[[v1:%.*]], <4 x i16>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[v1]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[v2]] to <8 x i8>
  // LLVM:   [[VHSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.shsub.v4i16(<4 x i16> [[v1]], <4 x i16> [[v2]])
  // LLVM:   [[VHSUB_V3_I:%.*]] = bitcast <4 x i16> [[VHSUB_V2_I]] to <8 x i8>
  // LLVM:   ret <4 x i16> [[VHSUB_V2_I]]
}

int32x2_t test_vhsub_s32(int32x2_t v1, int32x2_t v2) {
  return vhsub_s32(v1, v2);

  // CIR-LABEL: vhsub_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vhsub_s32(<2 x i32>{{.*}}[[v1:%.*]], <2 x i32>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[v1]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[v2]] to <8 x i8>
  // LLVM:   [[VHSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.shsub.v2i32(<2 x i32> [[v1]], <2 x i32> [[v2]])
  // LLVM:   [[VHSUB_V3_I:%.*]] = bitcast <2 x i32> [[VHSUB_V2_I]] to <8 x i8>
  // LLVM:   ret <2 x i32> [[VHSUB_V2_I]]
}

uint8x8_t test_vhsub_u8(uint8x8_t v1, uint8x8_t v2) {
  return vhsub_u8(v1, v2);

  // CIR-LABEL: vhsub_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vhsub_u8(<8 x i8>{{.*}}[[v1:%.*]], <8 x i8>{{.*}}[[v2:%.*]])
  // LLVM:   [[VHSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uhsub.v8i8(<8 x i8> [[v1]], <8 x i8> [[v2]])
  // LLVM:   ret <8 x i8> [[VHSUB_V_I]]
}

uint16x4_t test_vhsub_u16(uint16x4_t v1, uint16x4_t v2) {
  return vhsub_u16(v1, v2);

  // CIR-LABEL: vhsub_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vhsub_u16(<4 x i16>{{.*}}[[v1:%.*]], <4 x i16>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[v1]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[v2]] to <8 x i8>
  // LLVM:   [[VHSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uhsub.v4i16(<4 x i16> [[v1]], <4 x i16> [[v2]])
  // LLVM:   [[VHSUB_V3_I:%.*]] = bitcast <4 x i16> [[VHSUB_V2_I]] to <8 x i8>
  // LLVM:   ret <4 x i16> [[VHSUB_V2_I]]
}

uint32x2_t test_vhsub_u32(uint32x2_t v1, uint32x2_t v2) {
  return vhsub_u32(v1, v2);

  // CIR-LABEL: vhsub_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vhsub_u32(<2 x i32>{{.*}}[[v1:%.*]], <2 x i32>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[v1]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[v2]] to <8 x i8>
  // LLVM:   [[VHSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uhsub.v2i32(<2 x i32> [[v1]], <2 x i32> [[v2]])
  // LLVM:   [[VHSUB_V3_I:%.*]] = bitcast <2 x i32> [[VHSUB_V2_I]] to <8 x i8>
  // LLVM:   ret <2 x i32> [[VHSUB_V2_I]]
}

int8x16_t test_vhsubq_s8(int8x16_t v1, int8x16_t v2) {
  return vhsubq_s8(v1, v2);

  // CIR-LABEL: vhsubq_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}@test_vhsubq_s8(<16 x i8>{{.*}}[[v1:%.*]], <16 x i8>{{.*}}[[v2:%.*]])
  // LLVM:   [[VHSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.shsub.v16i8(<16 x i8> [[v1]], <16 x i8> [[v2]])
  // LLVM:   ret <16 x i8> [[VHSUBQ_V_I]]
}

int16x8_t test_vhsubq_s16(int16x8_t v1, int16x8_t v2) {
  return vhsubq_s16(v1, v2);

  // CIR-LABEL: vhsubq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}@test_vhsubq_s16(<8 x i16>{{.*}}[[v1:%.*]], <8 x i16>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[v1]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[v2]] to <16 x i8>
  // LLVM:   [[VHSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.shsub.v8i16(<8 x i16> [[v1]], <8 x i16> [[v2]])
  // LLVM:   [[VHSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VHSUBQ_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VHSUBQ_V2_I]]
}

int32x4_t test_vhsubq_s32(int32x4_t v1, int32x4_t v2) {
  return vhsubq_s32(v1, v2);

  // CIR-LABEL: vhsubq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.shsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}@test_vhsubq_s32(<4 x i32>{{.*}}[[v1:%.*]], <4 x i32>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[v1]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[v2]] to <16 x i8>
  // LLVM:   [[VHSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.shsub.v4i32(<4 x i32> [[v1]], <4 x i32> [[v2]])
  // LLVM:   [[VHSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VHSUBQ_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VHSUBQ_V2_I]]
}

uint8x16_t test_vhsubq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vhsubq_u8(v1, v2);

  // CIR-LABEL: vhsubq_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}@test_vhsubq_u8(<16 x i8>{{.*}}[[v1:%.*]], <16 x i8>{{.*}}[[v2:%.*]])
  // LLVM:   [[VHSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uhsub.v16i8(<16 x i8> [[v1]], <16 x i8> [[v2]])
  // LLVM:   ret <16 x i8> [[VHSUBQ_V_I]]
}

uint16x8_t test_vhsubq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vhsubq_u16(v1, v2);

  // CIR-LABEL: vhsubq_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}@test_vhsubq_u16(<8 x i16>{{.*}}[[v1:%.*]], <8 x i16>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[v1]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[v2]] to <16 x i8>
  // LLVM:   [[VHSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uhsub.v8i16(<8 x i16> [[v1]], <8 x i16> [[v2]])
  // LLVM:   [[VHSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VHSUBQ_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VHSUBQ_V2_I]]
}

uint32x4_t test_vhsubq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vhsubq_u32(v1, v2);

  // CIR-LABEL: vhsubq_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uhsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vhsubq_u32(<4 x i32>{{.*}}[[v1:%.*]], <4 x i32>{{.*}}[[v2:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[v1]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[v2]] to <16 x i8>
  // LLVM:   [[VHSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uhsub.v4i32(<4 x i32> [[v1]], <4 x i32> [[v2]])
  // LLVM:   [[VHSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VHSUBQ_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VHSUBQ_V2_I]]
}

int8x8_t test_vrhadd_s8(int8x8_t v1, int8x8_t v2) {
  return vrhadd_s8(v1, v2);

  // CIR-LABEL: vrhadd_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vrhadd_s8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VRHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.srhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VRHADD_V_I]]
}

int16x4_t test_vrhadd_s16(int16x4_t v1, int16x4_t v2) {
  return vrhadd_s16(v1, v2);

  // CIR-LABEL: vrhadd_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vrhadd_s16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM: [[VRHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.srhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: [[VRHADD_V3_I:%.*]] = bitcast <4 x i16> [[VRHADD_V2_I]] to <8 x i8>
  // LLVM: ret <4 x i16> [[VRHADD_V2_I]]
}

int32x2_t test_vrhadd_s32(int32x2_t v1, int32x2_t v2) {
  return vrhadd_s32(v1, v2);

  // CIR-LABEL: vrhadd_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vrhadd_s32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM: [[VRHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.srhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: [[VRHADD_V3_I:%.*]] = bitcast <2 x i32> [[VRHADD_V2_I]] to <8 x i8>
  // LLVM: ret <2 x i32> [[VRHADD_V2_I]]
}

uint8x8_t test_vrhadd_u8(uint8x8_t v1, uint8x8_t v2) {
  return vrhadd_u8(v1, v2);

  // CIR-LABEL: vrhadd_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vrhadd_u8(<8 x i8>{{.*}}[[V1:%.*]], <8 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VRHADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.urhadd.v8i8(<8 x i8> [[V1]], <8 x i8> [[V2]])
  // LLVM: ret <8 x i8> [[VRHADD_V_I]]
}

uint16x4_t test_vrhadd_u16(uint16x4_t v1, uint16x4_t v2) {
  return vrhadd_u16(v1, v2);

  // CIR-LABEL: vrhadd_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vrhadd_u16(<4 x i16>{{.*}}[[V1:%.*]], <4 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[V2]] to <8 x i8>
  // LLVM: [[VRHADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.urhadd.v4i16(<4 x i16> [[V1]], <4 x i16> [[V2]])
  // LLVM: [[VRHADD_V3_I:%.*]] = bitcast <4 x i16> [[VRHADD_V2_I]] to <8 x i8>
  // LLVM: ret <4 x i16> [[VRHADD_V2_I]]
}

uint32x2_t test_vrhadd_u32(uint32x2_t v1, uint32x2_t v2) {
  return vrhadd_u32(v1, v2);

  // CIR-LABEL: vrhadd_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vrhadd_u32(<2 x i32>{{.*}}[[V1:%.*]], <2 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[V1]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[V2]] to <8 x i8>
  // LLVM: [[VRHADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.urhadd.v2i32(<2 x i32> [[V1]], <2 x i32> [[V2]])
  // LLVM: [[VRHADD_V3_I:%.*]] = bitcast <2 x i32> [[VRHADD_V2_I]] to <8 x i8>
  // LLVM: ret <2 x i32> [[VRHADD_V2_I]]
}

int8x16_t test_vrhaddq_s8(int8x16_t v1, int8x16_t v2) {
  return vrhaddq_s8(v1, v2);

  // CIR-LABEL: vrhaddq_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}@test_vrhaddq_s8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VRHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.srhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VRHADDQ_V_I]]
}

int16x8_t test_vrhaddq_s16(int16x8_t v1, int16x8_t v2) {
  return vrhaddq_s16(v1, v2);

  // CIR-LABEL: vrhaddq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}@test_vrhaddq_s16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM: [[VRHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.srhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: [[VRHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VRHADDQ_V2_I]] to <16 x i8>
  // LLVM: ret <8 x i16> [[VRHADDQ_V2_I]]
}

int32x4_t test_vrhaddq_s32(int32x4_t v1, int32x4_t v2) {
  return vrhaddq_s32(v1, v2);

  // CIR-LABEL: vrhaddq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}@test_vrhaddq_s32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM: [[VRHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.srhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: [[VRHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VRHADDQ_V2_I]] to <16 x i8>
  // LLVM: ret <4 x i32> [[VRHADDQ_V2_I]]
}

uint8x16_t test_vrhaddq_u8(uint8x16_t v1, uint8x16_t v2) {
  return vrhaddq_u8(v1, v2);

  // CIR-LABEL: vrhaddq_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}@test_vrhaddq_u8(<16 x i8>{{.*}}[[V1:%.*]], <16 x i8>{{.*}}[[V2:%.*]])
  // LLVM: [[VRHADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.urhadd.v16i8(<16 x i8> [[V1]], <16 x i8> [[V2]])
  // LLVM: ret <16 x i8> [[VRHADDQ_V_I]]
}

uint16x8_t test_vrhaddq_u16(uint16x8_t v1, uint16x8_t v2) {
  return vrhaddq_u16(v1, v2);

  // CIR-LABEL: vrhaddq_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}@test_vrhaddq_u16(<8 x i16>{{.*}}[[V1:%.*]], <8 x i16>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[V2]] to <16 x i8>
  // LLVM: [[VRHADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.urhadd.v8i16(<8 x i16> [[V1]], <8 x i16> [[V2]])
  // LLVM: [[VRHADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VRHADDQ_V2_I]] to <16 x i8>
  // LLVM: ret <8 x i16> [[VRHADDQ_V2_I]]
}

uint32x4_t test_vrhaddq_u32(uint32x4_t v1, uint32x4_t v2) {
  return vrhaddq_u32(v1, v2);

  // CIR-LABEL: vrhaddq_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urhadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vrhaddq_u32(<4 x i32>{{.*}}[[V1:%.*]], <4 x i32>{{.*}}[[V2:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[V1]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[V2]] to <16 x i8>
  // LLVM: [[VRHADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.urhadd.v4i32(<4 x i32> [[V1]], <4 x i32> [[V2]])
  // LLVM: [[VRHADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VRHADDQ_V2_I]] to <16 x i8>
  // LLVM: ret <4 x i32> [[VRHADDQ_V2_I]]
}

int8x8_t test_vqadd_s8(int8x8_t a, int8x8_t b) {
  return vqadd_s8(a, b);
  // CIR-LABEL: vqadd_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM-LABEL: @test_vqadd_s8(
  // LLVM:   [[VQADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> %0, <8 x i8> %1)
  // LLVM:   ret <8 x i8> [[VQADD_V_I]]
}

  int16x4_t test_vqadd_s16(int16x4_t a, int16x4_t b) {
    return vqadd_s16(a, b);
    // CIR-LABEL: vqadd_s16
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

    // LLVM-LABEL: @test_vqadd_s16(
    // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> %0 to <8 x i8>
    // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> %1 to <8 x i8>
    // LLVM:   [[VQADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> %0, <4 x i16> %1)
    // LLVM:   [[VQADD_V3_I:%.*]] = bitcast <4 x i16> [[VQADD_V2_I]] to <8 x i8>
    // LLVM:   ret <4 x i16> [[VQADD_V2_I]]
  }

  int32x2_t test_vqadd_s32(int32x2_t a, int32x2_t b) {
    return vqadd_s32(a, b);
    // CIR-LABEL: vqadd_s32
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

    // LLVM-LABEL: @test_vqadd_s32(
    // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> %0 to <8 x i8>
    // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> %1 to <8 x i8>
    // LLVM:   [[VQADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqadd.v2i32(<2 x i32> %0, <2 x i32> %1)
    // LLVM:   [[VQADD_V3_I:%.*]] = bitcast <2 x i32> [[VQADD_V2_I]] to <8 x i8>
    // LLVM:   ret <2 x i32> [[VQADD_V2_I]]
  }

  int64x1_t test_vqadd_s64(int64x1_t a, int64x1_t b) {
    return vqadd_s64(a, b);
    // CIR-LABEL: vqadd_s64
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

    // LLVM-LABEL: @test_vqadd_s64(
    // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> %0 to <8 x i8>
    // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> %1 to <8 x i8>
    // LLVM:   [[VQADD_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqadd.v1i64(<1 x i64> %0, <1 x i64> %1)
    // LLVM:   [[VQADD_V3_I:%.*]] = bitcast <1 x i64> [[VQADD_V2_I]] to <8 x i8>
    // LLVM:   ret <1 x i64> [[VQADD_V2_I]]
  }

  uint8x8_t test_vqadd_u8(uint8x8_t a, uint8x8_t b) {
    return vqadd_u8(a, b);
    // CIR-LABEL: vqadd_u8
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

    // LLVM-LABEL: @test_vqadd_u8(
    // LLVM:   [[VQADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> %0, <8 x i8> %1)
    // LLVM:   ret <8 x i8> [[VQADD_V_I]]
  }

  uint16x4_t test_vqadd_u16(uint16x4_t a, uint16x4_t b) {
    return vqadd_u16(a, b);
    // CIR-LABEL: vqadd_u16
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

    // LLVM-LABEL: @test_vqadd_u16(
    // LLVM:   [[VQADD_V_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> %0, <4 x i16> %1)
    // LLVM:   ret <4 x i16> [[VQADD_V_I]]
  }

  uint32x2_t test_vqadd_u32(uint32x2_t a, uint32x2_t b) {
    return vqadd_u32(a, b);
    // CIR-LABEL: vqadd_u32
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

    // LLVM-LABEL: @test_vqadd_u32(
    // LLVM:   [[VQADD_V_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqadd.v2i32(<2 x i32> %0, <2 x i32> %1)
    // LLVM:   ret <2 x i32> [[VQADD_V_I]]
  }

  uint64x1_t test_vqadd_u64(uint64x1_t a, uint64x1_t b) {
    return vqadd_u64(a, b);
    // CIR-LABEL: vqadd_u64
    // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} :
    // CIR-SAME: (!cir.vector<!u64i x 1>, !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>

    // LLVM-LABEL: @test_vqadd_u64(
    // LLVM:   [[VQADD_V_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqadd.v1i64(<1 x i64> %0, <1 x i64> %1)
    // LLVM:   ret <1 x i64> [[VQADD_V_I]]
  }

// NYI-LABEL: @test_vqaddq_s8(
// NYI:   [[VQADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQADDQ_V_I]]
// int8x16_t test_vqaddq_s8(int8x16_t a, int8x16_t b) {
//   return vqaddq_s8(a, b);
// }

// NYI-LABEL: @test_vqaddq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQADDQ_V2_I]]
// int16x8_t test_vqaddq_s16(int16x8_t a, int16x8_t b) {
//   return vqaddq_s16(a, b);
// }

// NYI-LABEL: @test_vqaddq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQADDQ_V2_I]]
// int32x4_t test_vqaddq_s32(int32x4_t a, int32x4_t b) {
//   return vqaddq_s32(a, b);
// }

// NYI-LABEL: @test_vqaddq_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQADDQ_V2_I]]
// int64x2_t test_vqaddq_s64(int64x2_t a, int64x2_t b) {
//   return vqaddq_s64(a, b);
// }

// NYI-LABEL: @test_vqaddq_u8(
// NYI:   [[VQADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQADDQ_V_I]]
// uint8x16_t test_vqaddq_u8(uint8x16_t a, uint8x16_t b) {
//   return vqaddq_u8(a, b);
// }

// NYI-LABEL: @test_vqaddq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQADDQ_V2_I]]
// uint16x8_t test_vqaddq_u16(uint16x8_t a, uint16x8_t b) {
//   return vqaddq_u16(a, b);
// }

// NYI-LABEL: @test_vqaddq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQADDQ_V2_I]]
// uint32x4_t test_vqaddq_u32(uint32x4_t a, uint32x4_t b) {
//   return vqaddq_u32(a, b);
// }

// NYI-LABEL: @test_vqaddq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VQADDQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQADDQ_V2_I]]
// uint64x2_t test_vqaddq_u64(uint64x2_t a, uint64x2_t b) {
//   return vqaddq_u64(a, b);
// }

// NYI-LABEL: @test_vqsub_s8(
// NYI:   [[VQSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQSUB_V_I]]
// int8x8_t test_vqsub_s8(int8x8_t a, int8x8_t b) {
//   return vqsub_s8(a, b);
// }

// NYI-LABEL: @test_vqsub_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <4 x i16> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQSUB_V2_I]]
// int16x4_t test_vqsub_s16(int16x4_t a, int16x4_t b) {
//   return vqsub_s16(a, b);
// }

// NYI-LABEL: @test_vqsub_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqsub.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <2 x i32> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQSUB_V2_I]]
// int32x2_t test_vqsub_s32(int32x2_t a, int32x2_t b) {
//   return vqsub_s32(a, b);
// }

// NYI-LABEL: @test_vqsub_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqsub.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <1 x i64> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQSUB_V2_I]]
// int64x1_t test_vqsub_s64(int64x1_t a, int64x1_t b) {
//   return vqsub_s64(a, b);
// }

// NYI-LABEL: @test_vqsub_u8(
// NYI:   [[VQSUB_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQSUB_V_I]]
// uint8x8_t test_vqsub_u8(uint8x8_t a, uint8x8_t b) {
//   return vqsub_u8(a, b);
// }

// NYI-LABEL: @test_vqsub_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <4 x i16> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQSUB_V2_I]]
// uint16x4_t test_vqsub_u16(uint16x4_t a, uint16x4_t b) {
//   return vqsub_u16(a, b);
// }

// NYI-LABEL: @test_vqsub_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqsub.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <2 x i32> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQSUB_V2_I]]
// uint32x2_t test_vqsub_u32(uint32x2_t a, uint32x2_t b) {
//   return vqsub_u32(a, b);
// }

// NYI-LABEL: @test_vqsub_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQSUB_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqsub.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQSUB_V3_I:%.*]] = bitcast <1 x i64> [[VQSUB_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQSUB_V2_I]]
// uint64x1_t test_vqsub_u64(uint64x1_t a, uint64x1_t b) {
//   return vqsub_u64(a, b);
// }

// NYI-LABEL: @test_vqsubq_s8(
// NYI:   [[VQSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqsub.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQSUBQ_V_I]]
// int8x16_t test_vqsubq_s8(int8x16_t a, int8x16_t b) {
//   return vqsubq_s8(a, b);
// }

// NYI-LABEL: @test_vqsubq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqsub.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQSUBQ_V2_I]]
// int16x8_t test_vqsubq_s16(int16x8_t a, int16x8_t b) {
//   return vqsubq_s16(a, b);
// }

// NYI-LABEL: @test_vqsubq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQSUBQ_V2_I]]
// int32x4_t test_vqsubq_s32(int32x4_t a, int32x4_t b) {
//   return vqsubq_s32(a, b);
// }

// NYI-LABEL: @test_vqsubq_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQSUBQ_V2_I]]
// int64x2_t test_vqsubq_s64(int64x2_t a, int64x2_t b) {
//   return vqsubq_s64(a, b);
// }

// NYI-LABEL: @test_vqsubq_u8(
// NYI:   [[VQSUBQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqsub.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQSUBQ_V_I]]
// uint8x16_t test_vqsubq_u8(uint8x16_t a, uint8x16_t b) {
//   return vqsubq_u8(a, b);
// }

// NYI-LABEL: @test_vqsubq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqsub.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQSUBQ_V2_I]]
// uint16x8_t test_vqsubq_u16(uint16x8_t a, uint16x8_t b) {
//   return vqsubq_u16(a, b);
// }

// NYI-LABEL: @test_vqsubq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqsub.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQSUBQ_V2_I]]
// uint32x4_t test_vqsubq_u32(uint32x4_t a, uint32x4_t b) {
//   return vqsubq_u32(a, b);
// }

// NYI-LABEL: @test_vqsubq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSUBQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqsub.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQSUBQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSUBQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQSUBQ_V2_I]]
// uint64x2_t test_vqsubq_u64(uint64x2_t a, uint64x2_t b) {
//   return vqsubq_u64(a, b);
// }

int8x8_t test_vshl_s8(int8x8_t a, int8x8_t b) {
  return vshl_s8(a, b);

  // CIR-LABEL: vshl_s8
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!s8i x 8>, {{%.*}} : !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vshl_s8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM:   [[VSHL_V_I:%.*]] = shl <8 x i8> [[A]], [[B]]
  // LLVM:   ret <8 x i8> [[VSHL_V_I]]
}

int16x4_t test_vshl_s16(int16x4_t a, int16x4_t b) {
  return vshl_s16(a, b);

  // CIR-LABEL: vshl_s16
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!s16i x 4>, {{%.*}} : !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vshl_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <4 x i16> [[A]], [[B]]
  // LLVM:   ret <4 x i16> [[VSHL_V2_I]]
}

int32x2_t test_vshl_s32(int32x2_t a, int32x2_t b) {
  return vshl_s32(a, b);

  // CIR-LABEL: vshl_s32
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!s32i x 2>, {{%.*}} : !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vshl_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <2 x i32> [[A]], [[B]]
  // LLVM:   ret <2 x i32> [[VSHL_V2_I]]
}

int64x1_t test_vshl_s64(int64x1_t a, int64x1_t b) {
  return vshl_s64(a, b);

  // CIR-LABEL: vshl_s64
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!s64i x 1>, {{%.*}} : !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

  // LLVM: {{.*}}test_vshl_s64(<1 x i64>{{.*}}[[A:%.*]], <1 x i64>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> [[B]] to <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <1 x i64> [[A]], [[B]]
  // LLVM:   ret <1 x i64> [[VSHL_V2_I]]
}

uint8x8_t test_vshl_u8(uint8x8_t a, int8x8_t b) {
  return vshl_u8(a, b);

  // CIR-LABEL: vshl_u8
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!u8i x 8>, {{%.*}} : !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vshl_u8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM:   [[VSHL_V_I:%.*]] = shl <8 x i8> [[A]], [[B]]
  // LLVM:   ret <8 x i8> [[VSHL_V_I]]
}

uint16x4_t test_vshl_u16(uint16x4_t a, int16x4_t b) {
  return vshl_u16(a, b);

  // CIR-LABEL: vshl_u16
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !cir.vector<!u16i x 4>, {{%.*}} : !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vshl_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <4 x i16> [[A]], [[B]]
  // LLVM:   ret <4 x i16> [[VSHL_V2_I]]
}

uint32x2_t test_vshl_u32(uint32x2_t a, int32x2_t b) {
  return vshl_u32(a, b);

  // CIR-LABEL: vshl_u32
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u32i x 2>, {{%.*}} : !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vshl_u32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to  <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to  <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <2 x i32> [[A]], [[B]]
  // LLVM:   ret <2 x i32> [[VSHL_V2_I]]
}

uint64x1_t test_vshl_u64(uint64x1_t a, int64x1_t b) {
  return vshl_u64(a, b);

  // CIR-LABEL: vshl_u64
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u64i x 1>, {{%.*}} : !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>

  // LLVM: {{.*}}test_vshl_u64(<1 x i64>{{.*}}[[A:%.*]], <1 x i64>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> [[B]] to <8 x i8>
  // LLVM:   [[VSHL_V2_I:%.*]] = shl <1 x i64> [[A]], [[B]]
  // LLVM:   ret <1 x i64> [[VSHL_V2_I]]
}

int8x16_t test_vshlq_s8(int8x16_t a, int8x16_t b) {
  return vshlq_s8(a, b);

  // CIR-LABEL: vshlq_s8
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!s8i x 16>, {{%.*}} : !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vshlq_s8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM:   [[VSHLQ_V_I:%.*]] = shl <16 x i8> [[A]], [[B]]
  // LLVM:   ret <16 x i8> [[VSHLQ_V_I]]
}

int16x8_t test_vshlq_s16(int16x8_t a, int16x8_t b) {
  return vshlq_s16(a, b);

  // CIR-LABEL: vshlq_s16
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!s16i x 8>, {{%.*}} : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vshlq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <8 x i16> [[A]], [[B]]
  // LLVM:   ret <8 x i16> [[VSHLQ_V2_I]]
}

int32x4_t test_vshlq_s32(int32x4_t a, int32x4_t b) {
  return vshlq_s32(a, b);

  // CIR-LABEL: vshlq_s32
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!s32i x 4>, {{%.*}} : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vshlq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <4 x i32> [[A]], [[B]]
  // LLVM:   ret <4 x i32> [[VSHLQ_V2_I]]
}

int64x2_t test_vshlq_s64(int64x2_t a, int64x2_t b) {
  return vshlq_s64(a, b);

  // CIR-LABEL: vshlq_s64
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!s64i x 2>, {{%.*}} : !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vshlq_s64(<2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <2 x i64> [[A]], [[B]]
  // LLVM:   ret <2 x i64> [[VSHLQ_V2_I]]
}

uint8x16_t test_vshlq_u8(uint8x16_t a, int8x16_t b) {
  return vshlq_u8(a, b);

  // CIR-LABEL: vshlq_u8
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u8i x 16>, {{%.*}} : !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}test_vshlq_u8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM:   [[VSHLQ_V_I:%.*]] = shl <16 x i8> [[A]], [[B]]
  // LLVM:   ret <16 x i8> [[VSHLQ_V_I]]
}

uint16x8_t test_vshlq_u16(uint16x8_t a, int16x8_t b) {
  return vshlq_u16(a, b);

  // CIR-LABEL: vshlq_u16
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u16i x 8>, {{%.*}} : !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vshlq_u16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <8 x i16> [[A]], [[B]]
  // LLVM:   ret <8 x i16> [[VSHLQ_V2_I]]
}

uint32x4_t test_vshlq_u32(uint32x4_t a, int32x4_t b) {
  return vshlq_u32(a, b);

  // CIR-LABEL: vshlq_u32
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u32i x 4>, {{%.*}} : !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vshlq_u32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <4 x i32> [[A]], [[B]]
  // LLVM:   ret <4 x i32> [[VSHLQ_V2_I]]
}

uint64x2_t test_vshlq_u64(uint64x2_t a, int64x2_t b) {
  return vshlq_u64(a, b);

  // CIR-LABEL: vshlq_u64
  // CIR: cir.shift(left, {{%.*}} : !cir.vector<!u64i x 2>, {{%.*}} : !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vshlq_u64(<2 x i64>{{.*}}[[A:%.*]], <2 x i64>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> [[B]] to <16 x i8>
  // LLVM:   [[VSHLQ_V2_I:%.*]] = shl <2 x i64> [[A]], [[B]]
  // LLVM:   ret <2 x i64> [[VSHLQ_V2_I]]
}

// NYI-LABEL: @test_vqshl_s8(
// NYI:   [[VQSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQSHL_V_I]]
// int8x8_t test_vqshl_s8(int8x8_t a, int8x8_t b) {
//   return vqshl_s8(a, b);
// }

// NYI-LABEL: @test_vqshl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQSHL_V2_I]]
// int16x4_t test_vqshl_s16(int16x4_t a, int16x4_t b) {
//   return vqshl_s16(a, b);
// }

// NYI-LABEL: @test_vqshl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQSHL_V2_I]]
// int32x2_t test_vqshl_s32(int32x2_t a, int32x2_t b) {
//   return vqshl_s32(a, b);
// }

// NYI-LABEL: @test_vqshl_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQSHL_V2_I]]
// int64x1_t test_vqshl_s64(int64x1_t a, int64x1_t b) {
//   return vqshl_s64(a, b);
// }

// NYI-LABEL: @test_vqshl_u8(
// NYI:   [[VQSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQSHL_V_I]]
// uint8x8_t test_vqshl_u8(uint8x8_t a, int8x8_t b) {
//   return vqshl_u8(a, b);
// }

// NYI-LABEL: @test_vqshl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQSHL_V2_I]]
// uint16x4_t test_vqshl_u16(uint16x4_t a, int16x4_t b) {
//   return vqshl_u16(a, b);
// }

// NYI-LABEL: @test_vqshl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQSHL_V2_I]]
// uint32x2_t test_vqshl_u32(uint32x2_t a, int32x2_t b) {
//   return vqshl_u32(a, b);
// }

// NYI-LABEL: @test_vqshl_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQSHL_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQSHL_V2_I]]
// uint64x1_t test_vqshl_u64(uint64x1_t a, int64x1_t b) {
//   return vqshl_u64(a, b);
// }

// NYI-LABEL: @test_vqshlq_s8(
// NYI:   [[VQSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQSHLQ_V_I]]
// int8x16_t test_vqshlq_s8(int8x16_t a, int8x16_t b) {
//   return vqshlq_s8(a, b);
// }

// NYI-LABEL: @test_vqshlq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQSHLQ_V2_I]]
// int16x8_t test_vqshlq_s16(int16x8_t a, int16x8_t b) {
//   return vqshlq_s16(a, b);
// }

// NYI-LABEL: @test_vqshlq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQSHLQ_V2_I]]
// int32x4_t test_vqshlq_s32(int32x4_t a, int32x4_t b) {
//   return vqshlq_s32(a, b);
// }

// NYI-LABEL: @test_vqshlq_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQSHLQ_V2_I]]
// int64x2_t test_vqshlq_s64(int64x2_t a, int64x2_t b) {
//   return vqshlq_s64(a, b);
// }

// NYI-LABEL: @test_vqshlq_u8(
// NYI:   [[VQSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQSHLQ_V_I]]
// uint8x16_t test_vqshlq_u8(uint8x16_t a, int8x16_t b) {
//   return vqshlq_u8(a, b);
// }

// NYI-LABEL: @test_vqshlq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQSHLQ_V2_I]]
// uint16x8_t test_vqshlq_u16(uint16x8_t a, int16x8_t b) {
//   return vqshlq_u16(a, b);
// }

// NYI-LABEL: @test_vqshlq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQSHLQ_V2_I]]
// uint32x4_t test_vqshlq_u32(uint32x4_t a, int32x4_t b) {
//   return vqshlq_u32(a, b);
// }

// NYI-LABEL: @test_vqshlq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQSHLQ_V2_I]]
// uint64x2_t test_vqshlq_u64(uint64x2_t a, int64x2_t b) {
//   return vqshlq_u64(a, b);
// }

int8x8_t test_vrshl_s8(int8x8_t a, int8x8_t b) {
  return vrshl_s8(a, b);

  // CIR-LABEL: vrshl_s8
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vrshl_s8(<8 x i8>{{.*}}[[a:%.*]], <8 x i8>{{.*}}[[b:%.*]])
  // LLVM:   [[VRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> [[a]], <8 x i8> [[b]])
  // LLVM:   ret <8 x i8> [[VRSHL_V_I]]
}

int16x4_t test_vrshl_s16(int16x4_t a, int16x4_t b) {
  return vrshl_s16(a, b);

  // CIR-LABEL: vrshl_s16
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vrshl_s16(<4 x i16>{{.*}}[[a:%.*]], <4 x i16>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> [[a]], <4 x i16> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VRSHL_V2_I]] to <8 x i8>
  // LLVM:   ret <4 x i16> [[VRSHL_V2_I]]
}

int32x2_t test_vrshl_s32(int32x2_t a, int32x2_t b) {
  return vrshl_s32(a, b);

  // CIR-LABEL: vrshl_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vrshl_s32(<2 x i32>{{.*}}[[a:%.*]], <2 x i32>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> [[a]], <2 x i32> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VRSHL_V2_I]] to <8 x i8>
  // LLVM:   ret <2 x i32> [[VRSHL_V2_I]]
}

int64x1_t test_vrshl_s64(int64x1_t a, int64x1_t b) {
  return vrshl_s64(a, b);

  // CIR-LABEL: vrshl_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

  // LLVM: {{.*}}test_vrshl_s64(<1 x i64>{{.*}}[[a:%.*]], <1 x i64>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[a]], <1 x i64> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VRSHL_V2_I]] to <8 x i8>
  // LLVM:   ret <1 x i64> [[VRSHL_V2_I]]
}

uint8x8_t test_vrshl_u8(uint8x8_t a, int8x8_t b) {
  return vrshl_u8(a, b);

  // CIR-LABEL: vrshl_u8
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vrshl_u8(<8 x i8>{{.*}}[[a:%.*]], <8 x i8>{{.*}}[[b:%.*]])
  // LLVM: [[VRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> [[a]], <8 x i8> [[b]])
  // LLVM: ret <8 x i8> [[VRSHL_V_I]]
}

uint16x4_t test_vrshl_u16(uint16x4_t a, int16x4_t b) {
  return vrshl_u16(a, b);

  // CIR-LABEL: vrshl_u16
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vrshl_u16(<4 x i16>{{.*}}[[a:%.*]], <4 x i16>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> [[a]], <4 x i16> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <4 x i16>
  // LLVM:   ret <4 x i16> [[VRSHL_V2_I]]
}

uint32x2_t test_vrshl_u32(uint32x2_t a, int32x2_t b) {
  return vrshl_u32(a, b);

  // CIR-LABEL: vrshl_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vrshl_u32(<2 x i32>{{.*}}[[a:%.*]], <2 x i32>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> [[a]], <2 x i32> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <2 x i32>
  // LLVM:   ret <2 x i32> [[VRSHL_V2_I]]
}

uint64x1_t test_vrshl_u64(uint64x1_t a, int64x1_t b) {
  return vrshl_u64(a, b);

  // CIR-LABEL: vrshl_u64
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u64i x 1>, !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>

  // LLVM: {{.*}}test_vrshl_u64(<1 x i64>{{.*}}[[a:%.*]], <1 x i64>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> [[b]] to <8 x i8>
  // LLVM:   [[VRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[a]], <1 x i64> [[b]])
  // LLVM:   [[VRSHL_V3_I:%.*]] = bitcast <1 x i64>
  // LLVM:   ret <1 x i64> [[VRSHL_V2_I]]
}

int8x16_t test_vrshlq_s8(int8x16_t a, int8x16_t b) {
  return vrshlq_s8(a, b);

  // CIR-LABEL: vrshlq_s8
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vrshlq_s8(<16 x i8>{{.*}}[[a:%.*]], <16 x i8>{{.*}}[[b:%.*]])
  // LLVM:   [[VRSHL_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> [[a]], <16 x i8> [[b]])
  // LLVM:   ret <16 x i8> [[VRSHL_V_I]]
}

int16x8_t test_vrshlq_s16(int16x8_t a, int16x8_t b) {
  return vrshlq_s16(a, b);

  // CIR-LABEL: vrshlq_s16
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vrshlq_s16(<8 x i16>{{.*}}[[a:%.*]], <8 x i16>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> [[a]], <8 x i16> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VRSHLQ_V2_I]]
}

int32x4_t test_vrshlq_s32(int32x4_t a, int32x4_t b) {
  return vrshlq_s32(a, b);

  // CIR-LABEL: vrshlq_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vrshlq_s32(<4 x i32>{{.*}}[[a:%.*]], <4 x i32>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> [[a]], <4 x i32> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VRSHLQ_V2_I]]
}

int64x2_t test_vrshlq_s64(int64x2_t a, int64x2_t b) {
  return vrshlq_s64(a, b);

  // CIR-LABEL: vrshlq_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vrshlq_s64(<2 x i64>{{.*}}[[a:%.*]], <2 x i64>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> [[a]], <2 x i64> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <2 x i64> [[VRSHLQ_V2_I]]
}

uint8x16_t test_vrshlq_u8(uint8x16_t a, int8x16_t b) {
  return vrshlq_u8(a, b);

  // CIR-LABEL: vrshlq_u8
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}test_vrshlq_u8(<16 x i8>{{.*}}[[a:%.*]], <16 x i8>{{.*}}[[b:%.*]])
  // LLVM:   [[VRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> [[a]], <16 x i8> [[b]])
  // LLVM:   ret <16 x i8> [[VRSHLQ_V_I]]
}

uint16x8_t test_vrshlq_u16(uint16x8_t a, int16x8_t b) {
  return vrshlq_u16(a, b);

  // CIR-LABEL: vrshlq_u16
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vrshlq_u16(<8 x i16>{{.*}}[[a:%.*]], <8 x i16>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> [[a]], <8 x i16> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VRSHLQ_V2_I]]
}

uint32x4_t test_vrshlq_u32(uint32x4_t a, int32x4_t b) {
  return vrshlq_u32(a, b);

  // CIR-LABEL: vrshlq_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vrshlq_u32(<4 x i32>{{.*}}[[a:%.*]], <4 x i32>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> [[a]], <4 x i32> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VRSHLQ_V2_I]]
}

uint64x2_t test_vrshlq_u64(uint64x2_t a, int64x2_t b) {
  return vrshlq_u64(a, b);

  // CIR-LABEL: vrshlq_u64
  // CIR: cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u64i x 2>, !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vrshlq_u64(<2 x i64>{{.*}}[[a:%.*]], <2 x i64>{{.*}}[[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> [[b]] to <16 x i8>
  // LLVM:   [[VRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> [[a]], <2 x i64> [[b]])
  // LLVM:   [[VRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VRSHLQ_V2_I]] to <16 x i8>
  // LLVM:   ret <2 x i64> [[VRSHLQ_V2_I]]
}

// NYI-LABEL: @test_vqrshl_s8(
// NYI:   [[VQRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQRSHL_V_I]]
// int8x8_t test_vqrshl_s8(int8x8_t a, int8x8_t b) {
//   return vqrshl_s8(a, b);
// }

// NYI-LABEL: @test_vqrshl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQRSHL_V2_I]]
// int16x4_t test_vqrshl_s16(int16x4_t a, int16x4_t b) {
//   return vqrshl_s16(a, b);
// }

// NYI-LABEL: @test_vqrshl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQRSHL_V2_I]]
// int32x2_t test_vqrshl_s32(int32x2_t a, int32x2_t b) {
//   return vqrshl_s32(a, b);
// }

// NYI-LABEL: @test_vqrshl_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqrshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQRSHL_V2_I]]
// int64x1_t test_vqrshl_s64(int64x1_t a, int64x1_t b) {
//   return vqrshl_s64(a, b);
// }

// NYI-LABEL: @test_vqrshl_u8(
// NYI:   [[VQRSHL_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VQRSHL_V_I]]
// uint8x8_t test_vqrshl_u8(uint8x8_t a, int8x8_t b) {
//   return vqrshl_u8(a, b);
// }

// NYI-LABEL: @test_vqrshl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <4 x i16> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VQRSHL_V2_I]]
// uint16x4_t test_vqrshl_u16(uint16x4_t a, int16x4_t b) {
//   return vqrshl_u16(a, b);
// }

// NYI-LABEL: @test_vqrshl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshl.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <2 x i32> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VQRSHL_V2_I]]
// uint32x2_t test_vqrshl_u32(uint32x2_t a, int32x2_t b) {
//   return vqrshl_u32(a, b);
// }

// NYI-LABEL: @test_vqrshl_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VQRSHL_V2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqrshl.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   [[VQRSHL_V3_I:%.*]] = bitcast <1 x i64> [[VQRSHL_V2_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQRSHL_V2_I]]
// uint64x1_t test_vqrshl_u64(uint64x1_t a, int64x1_t b) {
//   return vqrshl_u64(a, b);
// }

// NYI-LABEL: @test_vqrshlq_s8(
// NYI:   [[VQRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqrshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQRSHLQ_V_I]]
// int8x16_t test_vqrshlq_s8(int8x16_t a, int8x16_t b) {
//   return vqrshlq_s8(a, b);
// }

// NYI-LABEL: @test_vqrshlq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQRSHLQ_V2_I]]
// int16x8_t test_vqrshlq_s16(int16x8_t a, int16x8_t b) {
//   return vqrshlq_s16(a, b);
// }

// NYI-LABEL: @test_vqrshlq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQRSHLQ_V2_I]]
// int32x4_t test_vqrshlq_s32(int32x4_t a, int32x4_t b) {
//   return vqrshlq_s32(a, b);
// }

// NYI-LABEL: @test_vqrshlq_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqrshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQRSHLQ_V2_I]]
// int64x2_t test_vqrshlq_s64(int64x2_t a, int64x2_t b) {
//   return vqrshlq_s64(a, b);
// }

// NYI-LABEL: @test_vqrshlq_u8(
// NYI:   [[VQRSHLQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqrshl.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VQRSHLQ_V_I]]
// uint8x16_t test_vqrshlq_u8(uint8x16_t a, int8x16_t b) {
//   return vqrshlq_u8(a, b);
// }

// NYI-LABEL: @test_vqrshlq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqrshl.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VQRSHLQ_V2_I]]
// uint16x8_t test_vqrshlq_u16(uint16x8_t a, int16x8_t b) {
//   return vqrshlq_u16(a, b);
// }

// NYI-LABEL: @test_vqrshlq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqrshl.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQRSHLQ_V2_I]]
// uint32x4_t test_vqrshlq_u32(uint32x4_t a, int32x4_t b) {
//   return vqrshlq_u32(a, b);
// }

// NYI-LABEL: @test_vqrshlq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQRSHLQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqrshl.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VQRSHLQ_V3_I:%.*]] = bitcast <2 x i64> [[VQRSHLQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQRSHLQ_V2_I]]
// uint64x2_t test_vqrshlq_u64(uint64x2_t a, int64x2_t b) {
//   return vqrshlq_u64(a, b);
// }

// NYI-LABEL: @test_vsli_n_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 0)
// NYI:   ret <1 x i64> [[VSLI_N2]]
// poly64x1_t test_vsli_n_p64(poly64x1_t a, poly64x1_t b) {
//   return vsli_n_p64(a, b, 0);
// }

// NYI-LABEL: @test_vsliq_n_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 0)
// NYI:   ret <2 x i64> [[VSLI_N2]]
// poly64x2_t test_vsliq_n_p64(poly64x2_t a, poly64x2_t b) {
//   return vsliq_n_p64(a, b, 0);
// }

int8x8_t test_vmax_s8(int8x8_t a, int8x8_t b) {
  return vmax_s8(a, b);

  // CIR-LABEL: vmax_s8
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s8i x 8>

  // LLVM-LABEL: test_vmax_s8
  // LLVM-SAME: (<8 x i8> [[a:%.*]], <8 x i8> [[b:%.*]])
  // LLVM:    [[VMAX_I:%.*]] = call <8 x i8> @llvm.smax.v8i8(<8 x i8> [[a]], <8 x i8> [[b]])
  // LLVM:    ret <8 x i8> [[VMAX_I]]
}

int16x4_t test_vmax_s16(int16x4_t a, int16x4_t b) {
  return vmax_s16(a, b);

  // CIR-LABEL: vmax_s16
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s16i x 4>

  // LLVM-LABEL: test_vmax_s16
  // LLVM-SAME: (<4 x i16> [[a:%.*]], <4 x i16> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <4 x i16> @llvm.smax.v4i16(<4 x i16> [[a]], <4 x i16> [[b]])
  // LLVM:   ret <4 x i16> [[VMAX2_I]]
}

int32x2_t test_vmax_s32(int32x2_t a, int32x2_t b) {
  return vmax_s32(a, b);

  // CIR-LABEL: vmax_s32
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s32i x 2>

  // LLVM-LABEL: test_vmax_s32
  // LLVM-SAME: (<2 x i32> [[a:%.*]], <2 x i32> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <2 x i32> @llvm.smax.v2i32(<2 x i32> [[a]], <2 x i32> [[b]])
  // LLVM:   ret <2 x i32> [[VMAX2_I]]
}

uint8x8_t test_vmax_u8(uint8x8_t a, uint8x8_t b) {
  return vmax_u8(a, b);

  // CIR-LABEL: vmax_u8
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u8i x 8>

  // LLVM-LABEL: test_vmax_u8
  // LLVM-SAME: (<8 x i8> [[a:%.*]], <8 x i8> [[b:%.*]])
  // LLVM:    [[VMAX_I:%.*]] = call <8 x i8> @llvm.umax.v8i8(<8 x i8> [[a]], <8 x i8> [[b]])
  // LLVM:    ret <8 x i8> [[VMAX_I]]
}

uint16x4_t test_vmax_u16(uint16x4_t a, uint16x4_t b) {
  return vmax_u16(a, b);

  // CIR-LABEL: vmax_u16
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u16i x 4>

  // LLVM-LABEL: test_vmax_u16
  // LLVM-SAME: (<4 x i16> [[a:%.*]], <4 x i16> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <4 x i16> @llvm.umax.v4i16(<4 x i16> [[a]], <4 x i16> [[b]])
  // LLVM:   ret <4 x i16> [[VMAX2_I]]
}

uint32x2_t test_vmax_u32(uint32x2_t a, uint32x2_t b) {
  return vmax_u32(a, b);

  // CIR-LABEL: vmax_u32
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u32i x 2>

  // LLVM-LABEL: test_vmax_u32
  // LLVM-SAME: (<2 x i32> [[a:%.*]], <2 x i32> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <2 x i32> @llvm.umax.v2i32(<2 x i32> [[a]], <2 x i32> [[b]])
  // LLVM:   ret <2 x i32> [[VMAX2_I]]
}

float32x2_t test_vmax_f32(float32x2_t a, float32x2_t b) {
  return vmax_f32(a, b);

  // CIR-LABEL: vmax_f32
  // CIR: cir.fmaximum {{%.*}}, {{%.*}} : !cir.vector<!cir.float x 2>

  // LLVM-LABEL: test_vmax_f32
  // LLVM-SAME: (<2 x float> [[a:%.*]], <2 x float> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x float> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x float> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <2 x float> @llvm.maximum.v2f32(<2 x float> [[a]], <2 x float> [[b]])
  // LLVM:   ret <2 x float> [[VMAX2_I]]
}

int8x16_t test_vmaxq_s8(int8x16_t a, int8x16_t b) {
  return vmaxq_s8(a, b);

  // CIR-LABEL: vmaxq_s8
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s8i x 16>

  // LLVM-LABEL: test_vmaxq_s8
  // LLVM-SAME: (<16 x i8> [[a:%.*]], <16 x i8> [[b:%.*]])
  // LLVM:    [[VMAX_I:%.*]] = call <16 x i8> @llvm.smax.v16i8(<16 x i8> [[a]], <16 x i8> [[b]])
  // LLVM:    ret <16 x i8> [[VMAX_I]]
}

int16x8_t test_vmaxq_s16(int16x8_t a, int16x8_t b) {
  return vmaxq_s16(a, b);

  // CIR-LABEL: vmaxq_s16
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_vmaxq_s16
  // LLVM-SAME: (<8 x i16> [[a:%.*]], <8 x i16> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <8 x i16> @llvm.smax.v8i16(<8 x i16> [[a]], <8 x i16> [[b]])
  // LLVM:   ret <8 x i16> [[VMAX2_I]]
}

int32x4_t test_vmaxq_s32(int32x4_t a, int32x4_t b) {
  return vmaxq_s32(a, b);

  // CIR-LABEL: vmaxq_s32
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!s32i x 4>

  // LLVM-LABEL: test_vmaxq_s32
  // LLVM-SAME: (<4 x i32> [[a:%.*]], <4 x i32> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <4 x i32> @llvm.smax.v4i32(<4 x i32> [[a]], <4 x i32> [[b]])
  // LLVM:   ret <4 x i32> [[VMAX2_I]]
}

uint8x16_t test_vmaxq_u8(uint8x16_t a, uint8x16_t b) {
  return vmaxq_u8(a, b);

  // CIR-LABEL: vmaxq_u8
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u8i x 16>

  // LLVM-LABEL: test_vmaxq_u8
  // LLVM-SAME: (<16 x i8> [[a:%.*]], <16 x i8> [[b:%.*]])
  // LLVM:    [[VMAX_I:%.*]] = call <16 x i8> @llvm.umax.v16i8(<16 x i8> [[a]], <16 x i8> [[b]])
  // LLVM:    ret <16 x i8> [[VMAX_I]]
}

uint16x8_t test_vmaxq_u16(uint16x8_t a, uint16x8_t b) {
  return vmaxq_u16(a, b);

  // CIR-LABEL: vmaxq_u16
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u16i x 8>

  // LLVM-LABEL: test_vmaxq_u16
  // LLVM-SAME: (<8 x i16> [[a:%.*]], <8 x i16> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <8 x i16> @llvm.umax.v8i16(<8 x i16> [[a]], <8 x i16> [[b]])
  // LLVM:   ret <8 x i16> [[VMAX2_I]]
}

uint32x4_t test_vmaxq_u32(uint32x4_t a, uint32x4_t b) {
  return vmaxq_u32(a, b);

  // CIR-LABEL: vmaxq_u32
  // CIR: cir.binop(max, {{%.*}}, {{%.*}}) : !cir.vector<!u32i x 4>

  // LLVM-LABEL: test_vmaxq_u32
  // LLVM-SAME: (<4 x i32> [[a:%.*]], <4 x i32> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <4 x i32> @llvm.umax.v4i32(<4 x i32> [[a]], <4 x i32> [[b]])
  // LLVM:   ret <4 x i32> [[VMAX2_I]]
}

float32x4_t test_vmaxq_f32(float32x4_t a, float32x4_t b) {
  return vmaxq_f32(a, b);

  // CIR-LABEL: vmaxq_f32
  // CIR: cir.fmaximum {{%.*}}, {{%.*}} : !cir.vector<!cir.float x 4>

  // LLVM-LABEL: test_vmaxq_f32
  // LLVM-SAME: (<4 x float> [[a:%.*]], <4 x float> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x float> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x float> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <4 x float> @llvm.maximum.v4f32(<4 x float> [[a]], <4 x float> [[b]])
  // LLVM:   ret <4 x float> [[VMAX2_I]]
}

float64x2_t test_vmaxq_f64(float64x2_t a, float64x2_t b) {
  return vmaxq_f64(a, b);

  // CIR-LABEL: vmaxq_f64
  // CIR: cir.fmaximum {{%.*}}, {{%.*}} : !cir.vector<!cir.double x 2>

  // LLVM-LABEL: test_vmaxq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]], <2 x double> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x double> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x double> [[b]] to <16 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <2 x double> @llvm.maximum.v2f64(<2 x double> [[a]], <2 x double> [[b]])
  // LLVM:   ret <2 x double> [[VMAX2_I]]
}

int8x8_t test_vmin_s8(int8x8_t a, int8x8_t b) {
  return vmin_s8(a, b);

  // CIR-LABEL: vmin_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vmin_s8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smin.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
  // LLVM: ret <8 x i8> [[VMIN_I]]
}

int16x4_t test_vmin_s16(int16x4_t a, int16x4_t b) {
  return vmin_s16(a, b);

  // CIR-LABEL: vmin_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vmin_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smin.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM: ret <4 x i16> [[VMIN2_I]]
}

int32x2_t test_vmin_s32(int32x2_t a, int32x2_t b) {
  return vmin_s32(a, b);

  // CIR-LABEL: vmin_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vmin_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smin.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM: ret <2 x i32> [[VMIN2_I]]
}

uint8x8_t test_vmin_u8(uint8x8_t a, uint8x8_t b) {
  return vmin_u8(a, b);

  // CIR-LABEL: vmin_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vmin_u8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umin.v8i8(<8 x i8> [[A]], <8 x i8> [[B]])
  // LLVM: ret <8 x i8> [[VMIN_I]]
}

uint16x4_t test_vmin_u16(uint16x4_t a, uint16x4_t b) {
  return vmin_u16(a, b);

  // CIR-LABEL: vmin_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vmin_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umin.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM: ret <4 x i16> [[VMIN2_I]]
}

uint32x2_t test_vmin_u32(uint32x2_t a, uint32x2_t b) {
  return vmin_u32(a, b);

  // CIR-LABEL: vmin_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vmin_u32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umin.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM: ret <2 x i32> [[VMIN2_I]]
}

float32x2_t test_vmin_f32(float32x2_t a, float32x2_t b) {
  return vmin_f32(a, b);

  // CIR-LABEL: vmin_f32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fmin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.float x 2>, !cir.vector<!cir.float x 2>) -> !cir.vector<!cir.float x 2>

  // LLVM: {{.*}}@test_vmin_f32(<2 x float>{{.*}}[[A:%.*]], <2 x float>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x float> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x float> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmin.v2f32(<2 x float> [[A]], <2 x float> [[B]])
  // LLVM: ret <2 x float> [[VMIN2_I]]
}

float64x1_t test_vmin_f64(float64x1_t a, float64x1_t b) {
  return vmin_f64(a, b);

  // CIR-LABEL: vmin_f64
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fmin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.double x 1>, !cir.vector<!cir.double x 1>) -> !cir.vector<!cir.double x 1>

  // LLVM: {{.*}}@test_vmin_f64(<1 x double>{{.*}}[[A:%.*]], <1 x double>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <1 x double> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <1 x double> [[B]] to <8 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmin.v1f64(<1 x double> [[A]], <1 x double> [[B]])
  // LLVM: ret <1 x double> [[VMIN2_I]]
}

int8x16_t test_vminq_s8(int8x16_t a, int8x16_t b) {
  return vminq_s8(a, b);

  // CIR-LABEL: vminq_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}@test_vminq_s8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smin.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
  // LLVM: ret <16 x i8> [[VMIN_I]]
}

int16x8_t test_vminq_s16(int16x8_t a, int16x8_t b) {
  return vminq_s16(a, b);

  // CIR-LABEL: vminq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}@test_vminq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smin.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM: ret <8 x i16> [[VMIN2_I]]
}

int32x4_t test_vminq_s32(int32x4_t a, int32x4_t b) {
  return vminq_s32(a, b);

  // CIR-LABEL: vminq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}@test_vminq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smin.v4i32(<4 x i32> [[A]], <4 x i32>
  // LLVM: ret <4 x i32> [[VMIN2_I]]
}

uint8x16_t test_vminq_u8(uint8x16_t a, uint8x16_t b) {
  return vminq_u8(a, b);

  // CIR-LABEL: vminq_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}@test_vminq_u8(<16 x i8>{{.*}}[[A:%.*]], <16 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umin.v16i8(<16 x i8> [[A]], <16 x i8> [[B]])
  // LLVM: ret <16 x i8> [[VMIN_I]]
}

uint16x8_t test_vminq_u16(uint16x8_t a, uint16x8_t b) {
  return vminq_u16(a, b);

  // CIR-LABEL: vminq_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}@test_vminq_u16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umin.v8i16(<8 x i16> [[A]], <8 x i16>
  // LLVM: ret <8 x i16> [[VMIN2_I]]
}

uint32x4_t test_vminq_u32(uint32x4_t a, uint32x4_t b) {
  return vminq_u32(a, b);

  // CIR-LABEL: vminq_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vminq_u32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umin.v4i32(<4 x i32> [[A]], <4 x i32>
  // LLVM: ret <4 x i32> [[VMIN2_I]]
}

float64x2_t test_vminq_f64(float64x2_t a, float64x2_t b) {
  return vminq_f64(a, b);

  // CIR-LABEL: vminq_f64
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.fmin" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!cir.double x 2>, !cir.vector<!cir.double x 2>) -> !cir.vector<!cir.double x 2>

  // LLVM: {{.*}}@test_vminq_f64(<2 x double>{{.*}}[[A:%.*]], <2 x double>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x double> [[A]] to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x double> [[B]] to <16 x i8>
  // LLVM: [[VMIN2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmin.v2f64(<2 x double> [[A]], <2 x double>
  // LLVM: ret <2 x double> [[VMIN2_I]]
}

// NYI-LABEL: @test_vmaxnm_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VMAXNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxnm.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VMAXNM2_I]]
// float32x2_t test_vmaxnm_f32(float32x2_t a, float32x2_t b) {
//   return vmaxnm_f32(a, b);
// }

// NYI-LABEL: @test_vmaxnmq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VMAXNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxnm.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VMAXNM2_I]]
// float32x4_t test_vmaxnmq_f32(float32x4_t a, float32x4_t b) {
//   return vmaxnmq_f32(a, b);
// }

// NYI-LABEL: @test_vmaxnmq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VMAXNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxnm.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VMAXNM2_I]]
// float64x2_t test_vmaxnmq_f64(float64x2_t a, float64x2_t b) {
//   return vmaxnmq_f64(a, b);
// }

// NYI-LABEL: @test_vminnm_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VMINNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminnm.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VMINNM2_I]]
// float32x2_t test_vminnm_f32(float32x2_t a, float32x2_t b) {
//   return vminnm_f32(a, b);
// }

// NYI-LABEL: @test_vminnmq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VMINNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminnm.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VMINNM2_I]]
// float32x4_t test_vminnmq_f32(float32x4_t a, float32x4_t b) {
//   return vminnmq_f32(a, b);
// }

// NYI-LABEL: @test_vminnmq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VMINNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminnm.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VMINNM2_I]]
// float64x2_t test_vminnmq_f64(float64x2_t a, float64x2_t b) {
//   return vminnmq_f64(a, b);
// }

// NYI-LABEL: @test_vpmax_s8(
// NYI:   [[VPMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.smaxp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPMAX_I]]
// int8x8_t test_vpmax_s8(int8x8_t a, int8x8_t b) {
//   return vpmax_s8(a, b);
// }

// NYI-LABEL: @test_vpmax_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.smaxp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   ret <4 x i16> [[VPMAX2_I]]
// int16x4_t test_vpmax_s16(int16x4_t a, int16x4_t b) {
//   return vpmax_s16(a, b);
// }

// NYI-LABEL: @test_vpmax_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.smaxp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   ret <2 x i32> [[VPMAX2_I]]
// int32x2_t test_vpmax_s32(int32x2_t a, int32x2_t b) {
//   return vpmax_s32(a, b);
// }

// NYI-LABEL: @test_vpmax_u8(
// NYI:   [[VPMAX_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.umaxp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPMAX_I]]
// uint8x8_t test_vpmax_u8(uint8x8_t a, uint8x8_t b) {
//   return vpmax_u8(a, b);
// }

// NYI-LABEL: @test_vpmax_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.umaxp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   ret <4 x i16> [[VPMAX2_I]]
// uint16x4_t test_vpmax_u16(uint16x4_t a, uint16x4_t b) {
//   return vpmax_u16(a, b);
// }

// NYI-LABEL: @test_vpmax_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.umaxp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   ret <2 x i32> [[VPMAX2_I]]
// uint32x2_t test_vpmax_u32(uint32x2_t a, uint32x2_t b) {
//   return vpmax_u32(a, b);
// }

// NYI-LABEL: @test_vpmax_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxp.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VPMAX2_I]]
// float32x2_t test_vpmax_f32(float32x2_t a, float32x2_t b) {
//   return vpmax_f32(a, b);
// }

// NYI-LABEL: @test_vpmaxq_s8(
// NYI:   [[VPMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.smaxp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPMAX_I]]
// int8x16_t test_vpmaxq_s8(int8x16_t a, int8x16_t b) {
//   return vpmaxq_s8(a, b);
// }

// NYI-LABEL: @test_vpmaxq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smaxp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i16> [[VPMAX2_I]]
// int16x8_t test_vpmaxq_s16(int16x8_t a, int16x8_t b) {
//   return vpmaxq_s16(a, b);
// }

// NYI-LABEL: @test_vpmaxq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   ret <4 x i32> [[VPMAX2_I]]
// int32x4_t test_vpmaxq_s32(int32x4_t a, int32x4_t b) {
//   return vpmaxq_s32(a, b);
// }

// NYI-LABEL: @test_vpmaxq_u8(
// NYI:   [[VPMAX_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.umaxp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPMAX_I]]
// uint8x16_t test_vpmaxq_u8(uint8x16_t a, uint8x16_t b) {
//   return vpmaxq_u8(a, b);
// }

// NYI-LABEL: @test_vpmaxq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umaxp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i16> [[VPMAX2_I]]
// uint16x8_t test_vpmaxq_u16(uint16x8_t a, uint16x8_t b) {
//   return vpmaxq_u16(a, b);
// }

// NYI-LABEL: @test_vpmaxq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umaxp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   ret <4 x i32> [[VPMAX2_I]]
// uint32x4_t test_vpmaxq_u32(uint32x4_t a, uint32x4_t b) {
//   return vpmaxq_u32(a, b);
// }

// NYI-LABEL: @test_vpmaxq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxp.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VPMAX2_I]]
// float32x4_t test_vpmaxq_f32(float32x4_t a, float32x4_t b) {
//   return vpmaxq_f32(a, b);
// }

// NYI-LABEL: @test_vpmaxq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VPMAX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxp.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VPMAX2_I]]
// float64x2_t test_vpmaxq_f64(float64x2_t a, float64x2_t b) {
//   return vpmaxq_f64(a, b);
// }

// NYI-LABEL: @test_vpmin_s8(
// NYI:   [[VPMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sminp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPMIN_I]]
// int8x8_t test_vpmin_s8(int8x8_t a, int8x8_t b) {
//   return vpmin_s8(a, b);
// }

// NYI-LABEL: @test_vpmin_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sminp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   ret <4 x i16> [[VPMIN2_I]]
// int16x4_t test_vpmin_s16(int16x4_t a, int16x4_t b) {
//   return vpmin_s16(a, b);
// }

// NYI-LABEL: @test_vpmin_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sminp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   ret <2 x i32> [[VPMIN2_I]]
// int32x2_t test_vpmin_s32(int32x2_t a, int32x2_t b) {
//   return vpmin_s32(a, b);
// }

// NYI-LABEL: @test_vpmin_u8(
// NYI:   [[VPMIN_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uminp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPMIN_I]]
// uint8x8_t test_vpmin_u8(uint8x8_t a, uint8x8_t b) {
//   return vpmin_u8(a, b);
// }

// NYI-LABEL: @test_vpmin_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uminp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   ret <4 x i16> [[VPMIN2_I]]
// uint16x4_t test_vpmin_u16(uint16x4_t a, uint16x4_t b) {
//   return vpmin_u16(a, b);
// }

// NYI-LABEL: @test_vpmin_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uminp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   ret <2 x i32> [[VPMIN2_I]]
// uint32x2_t test_vpmin_u32(uint32x2_t a, uint32x2_t b) {
//   return vpmin_u32(a, b);
// }

// NYI-LABEL: @test_vpmin_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminp.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VPMIN2_I]]
// float32x2_t test_vpmin_f32(float32x2_t a, float32x2_t b) {
//   return vpmin_f32(a, b);
// }

// NYI-LABEL: @test_vpminq_s8(
// NYI:   [[VPMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.sminp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPMIN_I]]
// int8x16_t test_vpminq_s8(int8x16_t a, int8x16_t b) {
//   return vpminq_s8(a, b);
// }

// NYI-LABEL: @test_vpminq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sminp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i16> [[VPMIN2_I]]
// int16x8_t test_vpminq_s16(int16x8_t a, int16x8_t b) {
//   return vpminq_s16(a, b);
// }

// NYI-LABEL: @test_vpminq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sminp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   ret <4 x i32> [[VPMIN2_I]]
// int32x4_t test_vpminq_s32(int32x4_t a, int32x4_t b) {
//   return vpminq_s32(a, b);
// }

// NYI-LABEL: @test_vpminq_u8(
// NYI:   [[VPMIN_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.uminp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPMIN_I]]
// uint8x16_t test_vpminq_u8(uint8x16_t a, uint8x16_t b) {
//   return vpminq_u8(a, b);
// }

// NYI-LABEL: @test_vpminq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.uminp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i16> [[VPMIN2_I]]
// uint16x8_t test_vpminq_u16(uint16x8_t a, uint16x8_t b) {
//   return vpminq_u16(a, b);
// }

// NYI-LABEL: @test_vpminq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.uminp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   ret <4 x i32> [[VPMIN2_I]]
// uint32x4_t test_vpminq_u32(uint32x4_t a, uint32x4_t b) {
//   return vpminq_u32(a, b);
// }

// NYI-LABEL: @test_vpminq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminp.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VPMIN2_I]]
// float32x4_t test_vpminq_f32(float32x4_t a, float32x4_t b) {
//   return vpminq_f32(a, b);
// }

// NYI-LABEL: @test_vpminq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VPMIN2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminp.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VPMIN2_I]]
// float64x2_t test_vpminq_f64(float64x2_t a, float64x2_t b) {
//   return vpminq_f64(a, b);
// }

// NYI-LABEL: @test_vpmaxnm_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VPMAXNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmaxnmp.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VPMAXNM2_I]]
// float32x2_t test_vpmaxnm_f32(float32x2_t a, float32x2_t b) {
//   return vpmaxnm_f32(a, b);
// }

// NYI-LABEL: @test_vpmaxnmq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VPMAXNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmaxnmp.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VPMAXNM2_I]]
// float32x4_t test_vpmaxnmq_f32(float32x4_t a, float32x4_t b) {
//   return vpmaxnmq_f32(a, b);
// }

// NYI-LABEL: @test_vpmaxnmq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VPMAXNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmaxnmp.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VPMAXNM2_I]]
// float64x2_t test_vpmaxnmq_f64(float64x2_t a, float64x2_t b) {
//   return vpmaxnmq_f64(a, b);
// }

// NYI-LABEL: @test_vpminnm_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VPMINNM2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fminnmp.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VPMINNM2_I]]
// float32x2_t test_vpminnm_f32(float32x2_t a, float32x2_t b) {
//   return vpminnm_f32(a, b);
// }

// NYI-LABEL: @test_vpminnmq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VPMINNM2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fminnmp.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VPMINNM2_I]]
// float32x4_t test_vpminnmq_f32(float32x4_t a, float32x4_t b) {
//   return vpminnmq_f32(a, b);
// }

// NYI-LABEL: @test_vpminnmq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VPMINNM2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fminnmp.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VPMINNM2_I]]
// float64x2_t test_vpminnmq_f64(float64x2_t a, float64x2_t b) {
//   return vpminnmq_f64(a, b);
// }

// NYI-LABEL: @test_vpadd_s8(
// NYI:   [[VPADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPADD_V_I]]
// int8x8_t test_vpadd_s8(int8x8_t a, int8x8_t b) {
//   return vpadd_s8(a, b);
// }

// NYI-LABEL: @test_vpadd_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VPADD_V3_I:%.*]] = bitcast <4 x i16> [[VPADD_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VPADD_V2_I]]
// int16x4_t test_vpadd_s16(int16x4_t a, int16x4_t b) {
//   return vpadd_s16(a, b);
// }

// NYI-LABEL: @test_vpadd_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VPADD_V3_I:%.*]] = bitcast <2 x i32> [[VPADD_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VPADD_V2_I]]
// int32x2_t test_vpadd_s32(int32x2_t a, int32x2_t b) {
//   return vpadd_s32(a, b);
// }

// NYI-LABEL: @test_vpadd_u8(
// NYI:   [[VPADD_V_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.addp.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VPADD_V_I]]
// uint8x8_t test_vpadd_u8(uint8x8_t a, uint8x8_t b) {
//   return vpadd_u8(a, b);
// }

// NYI-LABEL: @test_vpadd_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VPADD_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.addp.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VPADD_V3_I:%.*]] = bitcast <4 x i16> [[VPADD_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VPADD_V2_I]]
// uint16x4_t test_vpadd_u16(uint16x4_t a, uint16x4_t b) {
//   return vpadd_u16(a, b);
// }

// NYI-LABEL: @test_vpadd_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VPADD_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.addp.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VPADD_V3_I:%.*]] = bitcast <2 x i32> [[VPADD_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VPADD_V2_I]]
// uint32x2_t test_vpadd_u32(uint32x2_t a, uint32x2_t b) {
//   return vpadd_u32(a, b);
// }

// NYI-LABEL: @test_vpadd_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VPADD_V2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.faddp.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   [[VPADD_V3_I:%.*]] = bitcast <2 x float> [[VPADD_V2_I]] to <8 x i8>
// NYI:   ret <2 x float> [[VPADD_V2_I]]
// float32x2_t test_vpadd_f32(float32x2_t a, float32x2_t b) {
//   return vpadd_f32(a, b);
// }

// NYI-LABEL: @test_vpaddq_s8(
// NYI:   [[VPADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPADDQ_V_I]]
// int8x16_t test_vpaddq_s8(int8x16_t a, int8x16_t b) {
//   return vpaddq_s8(a, b);
// }

// NYI-LABEL: @test_vpaddq_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VPADDQ_V2_I]]
// int16x8_t test_vpaddq_s16(int16x8_t a, int16x8_t b) {
//   return vpaddq_s16(a, b);
// }

// NYI-LABEL: @test_vpaddq_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VPADDQ_V2_I]]
// int32x4_t test_vpaddq_s32(int32x4_t a, int32x4_t b) {
//   return vpaddq_s32(a, b);
// }

// NYI-LABEL: @test_vpaddq_u8(
// NYI:   [[VPADDQ_V_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.addp.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VPADDQ_V_I]]
// uint8x16_t test_vpaddq_u8(uint8x16_t a, uint8x16_t b) {
//   return vpaddq_u8(a, b);
// }

// NYI-LABEL: @test_vpaddq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.addp.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <8 x i16> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <8 x i16> [[VPADDQ_V2_I]]
// uint16x8_t test_vpaddq_u16(uint16x8_t a, uint16x8_t b) {
//   return vpaddq_u16(a, b);
// }

// NYI-LABEL: @test_vpaddq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.addp.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x i32> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VPADDQ_V2_I]]
// uint32x4_t test_vpaddq_u32(uint32x4_t a, uint32x4_t b) {
//   return vpaddq_u32(a, b);
// }

// NYI-LABEL: @test_vpaddq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.faddp.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <4 x float> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <4 x float> [[VPADDQ_V2_I]]
// float32x4_t test_vpaddq_f32(float32x4_t a, float32x4_t b) {
//   return vpaddq_f32(a, b);
// }

// NYI-LABEL: @test_vpaddq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VPADDQ_V2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.faddp.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x double> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x double> [[VPADDQ_V2_I]]
// float64x2_t test_vpaddq_f64(float64x2_t a, float64x2_t b) {
//   return vpaddq_f64(a, b);
// }

int16x4_t test_vqdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqdmulh_s16(a, b);

  // CIR-LABEL: vqdmulh_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vqdmulh_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM:   [[VQDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQDMULH_V2_I]] to <8 x i8>
  // LLVM:   ret <4 x i16> [[VQDMULH_V2_I]]
}

int32x2_t test_vqdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqdmulh_s32(a, b);

  // CIR-LABEL: vqdmulh_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vqdmulh_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM:   [[VQDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqdmulh.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM:   [[VQDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQDMULH_V2_I]] to <8 x i8>
  // LLVM:   ret <2 x i32> [[VQDMULH_V2_I]]
}

int16x8_t test_vqdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqdmulhq_s16(a, b);

  // CIR-LABEL: vqdmulhq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vqdmulhq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM:   [[VQDMULH_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqdmulh.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM:   [[VQDMULH_V3_I:%.*]] = bitcast <8 x i16> [[VQDMULH_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VQDMULH_V2_I]]
}

int32x4_t test_vqdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqdmulhq_s32(a, b);

  // CIR-LABEL: vqdmulhq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vqdmulhq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM:   [[VQDMULH_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmulh.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM:   [[VQDMULH_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULH_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VQDMULH_V2_I]]
}

int16x4_t test_vqrdmulh_s16(int16x4_t a, int16x4_t b) {
  return vqrdmulh_s16(a, b);

  // CIR-LABEL: vqrdmulh_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vqrdmulh_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM:   [[VQRDMULH_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM:   [[VQRDMULH_V3_I:%.*]] = bitcast <4 x i16> [[VQRDMULH_V2_I]] to <8 x i8>
  // LLVM:   ret <4 x i16> [[VQRDMULH_V2_I]]
}

int32x2_t test_vqrdmulh_s32(int32x2_t a, int32x2_t b) {
  return vqrdmulh_s32(a, b);

  // CIR-LABEL: vqrdmulh_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vqrdmulh_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM:   [[VQRDMULH_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrdmulh.v2i32(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM:   [[VQRDMULH_V3_I:%.*]] = bitcast <2 x i32> [[VQRDMULH_V2_I]] to <8 x i8>
  // LLVM:   ret <2 x i32> [[VQRDMULH_V2_I]]
}

int16x8_t test_vqrdmulhq_s16(int16x8_t a, int16x8_t b) {
  return vqrdmulhq_s16(a, b);

  // CIR-LABEL: vqrdmulhq_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vqrdmulhq_s16(<8 x i16>{{.*}}[[A:%.*]], <8 x i16>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> [[B]] to <16 x i8>
  // LLVM:   [[VQRDMULHQ_V2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqrdmulh.v8i16(<8 x i16> [[A]], <8 x i16> [[B]])
  // LLVM:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <8 x i16> [[VQRDMULHQ_V2_I]] to <16 x i8>
  // LLVM:   ret <8 x i16> [[VQRDMULHQ_V2_I]]
}

int32x4_t test_vqrdmulhq_s32(int32x4_t a, int32x4_t b) {
  return vqrdmulhq_s32(a, b);

  // CIR-LABEL: vqrdmulhq_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrdmulh" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vqrdmulhq_s32(<4 x i32>{{.*}}[[A:%.*]], <4 x i32>{{.*}}[[B:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> [[B]] to <16 x i8>
  // LLVM:   [[VQRDMULHQ_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqrdmulh.v4i32(<4 x i32> [[A]], <4 x i32> [[B]])
  // LLVM:   [[VQRDMULHQ_V3_I:%.*]] = bitcast <4 x i32> [[VQRDMULHQ_V2_I]] to <16 x i8>
  // LLVM:   ret <4 x i32> [[VQRDMULHQ_V2_I]]
}

// NYI-LABEL: @test_vmulx_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[VMULX2_I:%.*]] = call <2 x float> @llvm.aarch64.neon.fmulx.v2f32(<2 x float> %a, <2 x float> %b)
// NYI:   ret <2 x float> [[VMULX2_I]]
// float32x2_t test_vmulx_f32(float32x2_t a, float32x2_t b) {
//   return vmulx_f32(a, b);
// }

// NYI-LABEL: @test_vmulxq_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[VMULX2_I:%.*]] = call <4 x float> @llvm.aarch64.neon.fmulx.v4f32(<4 x float> %a, <4 x float> %b)
// NYI:   ret <4 x float> [[VMULX2_I]]
// float32x4_t test_vmulxq_f32(float32x4_t a, float32x4_t b) {
//   return vmulxq_f32(a, b);
// }

// NYI-LABEL: @test_vmulxq_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[VMULX2_I:%.*]] = call <2 x double> @llvm.aarch64.neon.fmulx.v2f64(<2 x double> %a, <2 x double> %b)
// NYI:   ret <2 x double> [[VMULX2_I]]
// float64x2_t test_vmulxq_f64(float64x2_t a, float64x2_t b) {
//   return vmulxq_f64(a, b);
// }


int8x8_t test_vshl_n_s8(int8x8_t a) {
  return vshl_n_s8(a, 3);

 // CIR-LABEL: @test_vshl_n_s8
 // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
 // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i]>
 // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s8i x 8>, [[AMT]] : !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

 // LLVM: {{.*}}@test_vshl_n_s8(<8 x i8>{{.*}}[[A:%.*]])
 // LLVM: [[VSHL_N:%.*]] = shl <8 x i8> [[A]], splat (i8 3)
 // LLVM: ret <8 x i8> [[VSHL_N]]
}


int16x4_t test_vshl_n_s16(int16x4_t a) {
  return vshl_n_s16(a, 3);

 // CIR-LABEL: @test_vshl_n_s16
 // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
 // CIR-SAME: #cir.int<3> : !s16i]>
 // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s16i x 4>, [[AMT]] : !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

 // LLVM: {{.*}}@test_vshl_n_s16(<4 x i16>{{.*}}[[A:%.*]])
 // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
 // LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
 // LLVM: [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
 // LLVM: ret <4 x i16> [[VSHL_N]]
}

int32x2_t test_vshl_n_s32(int32x2_t a) {
  return vshl_n_s32(a, 3);

  // CIR-LABEL: @test_vshl_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i]>
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s32i x 2>, [[AMT]] : !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vshl_n_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
  // LLVM: ret <2 x i32> [[VSHL_N]]
}

// NYI-LABEL: @test_vshlq_n_s8(
// NYI:   [[VSHL_N:%.*]] = shl <16 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
// NYI:   ret <16 x i8> [[VSHL_N]]
int8x16_t test_vshlq_n_s8(int8x16_t a) {
  return vshlq_n_s8(a, 3);

  // CIR-LABEL: @test_vshlq_n_s8
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s8i x 16>, {{.*}} : !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}@test_vshlq_n_s8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHL_N:%.*]] = shl <16 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <16 x i8> [[VSHL_N]]
}

int16x8_t test_vshlq_n_s16(int16x8_t a) {
  return vshlq_n_s16(a, 3);

  // CIR-LABEL: @test_vshlq_n_s16
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s16i x 8>, {{.*}} : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM:   {{.*}}@test_vshlq_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <8 x i16> [[VSHL_N]]
}


int32x4_t test_vshlq_n_s32(int32x4_t a) {
  return vshlq_n_s32(a, 3);

  // CIR-LABEL: @test_vshlq_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> :
  // CIR-SAME: !s32i, #cir.int<3> : !s32i, #cir.int<3> : !s32i]>
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s32i x 4>, [[AMT]] :
  // CIR-SAME: !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM:   {{.*}}@test_vshlq_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <4 x i32> [[VSHL_N]]
}

int64x2_t test_vshlq_n_s64(int64x2_t a) {
  return vshlq_n_s64(a, 3);

  // CIR-LABEL: @test_vshlq_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s64i, #cir.int<3> : !s64i]>
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!s64i x 2>, [[AMT]] :
  // CIR-SAME: !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM:   {{.*}}@test_vshlq_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
  // LLVM:   ret <2 x i64> [[VSHL_N]]
}

uint8x8_t test_vshl_n_u8(uint8x8_t a) {
  return vshl_n_u8(a, 3);

  // CIR-LABEL: @test_vshl_n_u8
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u8i x 8>, {{.*}} :
  // CIR-SAME: !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM:   {{.*}}@test_vshl_n_u8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHL_N:%.*]] = shl <8 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <8 x i8> [[VSHL_N]]
}

uint16x4_t test_vshl_n_u16(uint16x4_t a) {
  return vshl_n_u16(a, 3);

  // CIR-LABEL: @test_vshl_n_u16
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u16i x 4>, {{.*}} :
  // CIR-SAME: !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM:   {{.*}}@test_vshl_n_u16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[VSHL_N:%.*]] = shl <4 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <4 x i16> [[VSHL_N]]
}

uint32x2_t test_vshl_n_u32(uint32x2_t a) {
  return vshl_n_u32(a, 3);

  // CIR-LABEL: @test_vshl_n_u32
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u32i x 2>, {{.*}} :
  // CIR-SAME: !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM:   {{.*}}@test_vshl_n_u32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]]
  // LLVM:   [[VSHL_N:%.*]] = shl <2 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <2 x i32> [[VSHL_N]]
}

uint8x16_t test_vshlq_n_u8(uint8x16_t a) {
  return vshlq_n_u8(a, 3);

  // CIR-LABEL: @test_vshlq_n_u8
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u8i x 16>, {{.*}} :
  // CIR-SAME: !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM:   {{.*}}@test_vshlq_n_u8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHL_N:%.*]] = shl <16 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <16 x i8> [[VSHL_N]]
}

uint16x8_t test_vshlq_n_u16(uint16x8_t a) {
  return vshlq_n_u16(a, 3);

  // CIR-LABEL: @test_vshlq_n_u16
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u16i x 8>, {{.*}} :
  // CIR-SAME: !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM:   {{.*}}@test_vshlq_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VSHL_N:%.*]] = shl <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <8 x i16> [[VSHL_N]]
}

uint32x4_t test_vshlq_n_u32(uint32x4_t a) {
  return vshlq_n_u32(a, 3);

  // CIR-LABEL: @test_vshlq_n_u32
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u32i x 4>, {{.*}} :
  // CIR-SAME: !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM:   {{.*}}@test_vshlq_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]]
  // LLVM:   [[VSHL_N:%.*]] = shl <4 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <4 x i32> [[VSHL_N]]
}

uint64x2_t test_vshlq_n_u64(uint64x2_t a) {
  return vshlq_n_u64(a, 3);

  // CIR-LABEL: @test_vshlq_n_u64
  // CIR: {{.*}} = cir.shift(left, {{.*}} : !cir.vector<!u64i x 2>, {{.*}} :
  // CIR-SAME: !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM:   {{.*}}@test_vshlq_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]]
  // LLVM:   [[VSHL_N:%.*]] = shl <2 x i64> [[TMP1]], splat (i64 3)
  // LLVM:   ret <2 x i64> [[VSHL_N]]
}

int8x8_t test_vshr_n_s8(int8x8_t a) {
  return vshr_n_s8(a, 3);

  // CIR-LABEL: vshr_n_s8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i]> : !cir.vector<!s8i x 8>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s8i x 8>, [[AMT]] : !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vshr_n_s8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHR_N:%.*]] = ashr <8 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <8 x i8> [[VSHR_N]]
}

int16x4_t test_vshr_n_s16(int16x4_t a) {
  return vshr_n_s16(a, 3);

  // CIR-LABEL: vshr_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i]> : !cir.vector<!s16i x 4>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s16i x 4>, [[AMT]] : !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vshr_n_s16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[VSHR_N:%.*]] = ashr <4 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <4 x i16> [[VSHR_N]]
}

int32x2_t test_vshr_n_s32(int32x2_t a) {
  return vshr_n_s32(a, 3);

  // CIR-LABEL: vshr_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i]> : !cir.vector<!s32i x 2>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s32i x 2>, [[AMT]] : !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vshr_n_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[VSHR_N:%.*]] = ashr <2 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <2 x i32> [[VSHR_N]]
}

int64x1_t test_vshr_n_s64(int64x1_t a) {
  return vshr_n_s64(a, 3);

  // CIR-LABEL: vshr_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s64i]> : !cir.vector<!s64i x 1>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s64i x 1>, [[AMT]] : !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

  // LLVM: {{.*}}test_vshr_n_s64(<1 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VSHR_N:%.*]] = ashr <1 x i64> [[TMP1]], splat (i64 3)
  // LLVM: ret <1 x i64> [[VSHR_N]]
}

int8x16_t test_vshrq_n_s8(int8x16_t a) {
  return vshrq_n_s8(a, 3);

  // CIR-LABEL: vshrq_n_s8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i]> : !cir.vector<!s8i x 16>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s8i x 16>, [[AMT]] : !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}test_vshrq_n_s8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VSHR_N:%.*]] = ashr <16 x i8> [[A]], splat (i8 3)
  // LLVM: ret <16 x i8> [[VSHR_N]]
}

int16x8_t test_vshrq_n_s16(int16x8_t a) {
  return vshrq_n_s16(a, 3);

  // CIR-LABEL: vshrq_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s16i x 8>, [[AMT]] : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vshrq_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VSHR_N:%.*]] = ashr <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <8 x i16> [[VSHR_N]]
}

int32x4_t test_vshrq_n_s32(int32x4_t a) {
  return vshrq_n_s32(a, 3);

  // CIR-LABEL: vshrq_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i,
  // CIR-SAME: #cir.int<3> : !s32i, #cir.int<3> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s32i x 4>, [[AMT]] : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vshrq_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VSHR_N:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <4 x i32> [[VSHR_N]]
}

// Vector lashr/ashr are undefined when the shift amount is equal to the vector
// element size. Thus in code gen, for singed input, we make the shift amount
// one less than the vector element size.
int32x4_t test_vshrq_n_s32_32(int32x4_t a) {
  return vshrq_n_s32(a, 32);

  // CIR-LABEL: vshrq_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<31> : !s32i, #cir.int<31> : !s32i,
  // CIR-SAME: #cir.int<31> : !s32i, #cir.int<31> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s32i x 4>, [[AMT]] : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vshrq_n_s32_32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VSHR_N:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 31)
  // LLVM:   ret <4 x i32> [[VSHR_N]]
}

int64x2_t test_vshrq_n_s64(int64x2_t a) {
  return vshrq_n_s64(a, 3);

  // CIR-LABEL: vshrq_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s64i, #cir.int<3> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!s64i x 2>, [[AMT]] : !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vshrq_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VSHR_N:%.*]] = ashr <2 x i64> [[TMP1]], splat (i64 3)
  // LLVM:   ret <2 x i64> [[VSHR_N]]
}

uint8x8_t test_vshr_n_u8(uint8x8_t a) {
  return vshr_n_u8(a, 3);

  // CIR-LABEL: vshr_n_u8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i,
  // CIR-SAME: #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i]> : !cir.vector<!u8i x 8>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u8i x 8>, [[AMT]] : !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vshr_n_u8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHR_N:%.*]] = lshr <8 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <8 x i8> [[VSHR_N]]
}

uint16x4_t test_vshr_n_u16(uint16x4_t a) {
  return vshr_n_u16(a, 3);

  // CIR-LABEL: vshr_n_u16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u16i, #cir.int<3> : !u16i,
  // CIR-SAME: #cir.int<3> : !u16i, #cir.int<3> : !u16i]> : !cir.vector<!u16i x 4>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u16i x 4>, [[AMT]] : !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vshr_n_u16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[VSHR_N:%.*]] = lshr <4 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <4 x i16> [[VSHR_N]]
}

// Vector lashr/ashr are undefined when the shift amount is equal to the vector
// element size. Thus in code gen, for unsinged input, return zero vector.
uint16x4_t test_vshr_n_u16_16(uint16x4_t a) {
  return vshr_n_u16(a, 16);

  // CIR-LABEL: vshr_n_u16
  // CIR: {{%.*}} = cir.const #cir.int<16> : !s32i
  // CIR: {{%.*}} = cir.const #cir.zero : !cir.vector<!u16i x 4>
  // CIR-NOT: cir.shift

  // LLVM: {{.*}}test_vshr_n_u16_16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: ret <4 x i16> zeroinitializer
}

uint32x2_t test_vshr_n_u32(uint32x2_t a) {
  return vshr_n_u32(a, 3);

  // CIR-LABEL: vshr_n_u32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u32i, #cir.int<3> : !u32i]> : !cir.vector<!u32i x 2>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u32i x 2>, [[AMT]] : !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vshr_n_u32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[VSHR_N:%.*]] = lshr <2 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <2 x i32> [[VSHR_N]]
}

uint64x1_t test_vshr_n_u64(uint64x1_t a) {
  return vshr_n_u64(a, 1);

  // CIR-LABEL: vshr_n_u64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<1> : !u64i]> : !cir.vector<!u64i x 1>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u64i x 1>, [[AMT]] : !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>

  // LLVM: {{.*}}test_vshr_n_u64(<1 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VSHR_N:%.*]] = lshr <1 x i64> [[TMP1]], splat (i64 1)
  // LLVM: ret <1 x i64> [[VSHR_N]]
}

uint8x16_t test_vshrq_n_u8(uint8x16_t a) {
  return vshrq_n_u8(a, 3);

  // CIR-LABEL: vshrq_n_u8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i,
  // CIR-SAME: #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i,
  // CIR-SAME: #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i,
  // CIR-SAME: #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i, #cir.int<3> : !u8i]> : !cir.vector<!u8i x 16>

  // LLVM: {{.*}}test_vshrq_n_u8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[VSHR_N:%.*]] = lshr <16 x i8> [[A]], splat (i8 3)
  // LLVM:   ret <16 x i8> [[VSHR_N]]
}

uint16x8_t test_vshrq_n_u16(uint16x8_t a) {
  return vshrq_n_u16(a, 3);

  // CIR-LABEL: vshrq_n_u16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i,
  // CIR-SAME: #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i,
  // CIR-SAME: #cir.int<3> : !u16i]> : !cir.vector<!u16i x 8>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u16i x 8>, [[AMT]] : !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vshrq_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VSHR_N:%.*]] = lshr <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   ret <8 x i16> [[VSHR_N]]
}

uint32x4_t test_vshrq_n_u32(uint32x4_t a) {
  return vshrq_n_u32(a, 3);

  // CIR-LABEL: vshrq_n_u32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u32i, #cir.int<3> : !u32i,
  // CIR-SAME: #cir.int<3> : !u32i, #cir.int<3> : !u32i]> : !cir.vector<!u32i x 4>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u32i x 4>, [[AMT]] : !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vshrq_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VSHR_N:%.*]] = lshr <4 x i32> [[TMP1]], splat (i32 3)
  // LLVM:   ret <4 x i32> [[VSHR_N]]
}

uint64x2_t test_vshrq_n_u64(uint64x2_t a) {
  return vshrq_n_u64(a, 3);

  // CIR-LABEL: vshrq_n_u64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u64i, #cir.int<3> : !u64i]> : !cir.vector<!u64i x 2>
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !cir.vector<!u64i x 2>, [[AMT]] : !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vshrq_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VSHR_N:%.*]] = lshr <2 x i64> [[TMP1]], splat (i64 3)
  // LLVM:   ret <2 x i64> [[VSHR_N]]
}

int8x8_t test_vsra_n_s8(int8x8_t a, int8x8_t b) {
  return vsra_n_s8(a, b, 3);

  // CIR-LABEL: vsra_n_s8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s8i x 8>

  // LLVM-LABEL: @test_vsra_n_s8(
  // LLVM: [[VSRA_N:%.*]] = ashr <8 x i8> %1, splat (i8 3)
  // LLVM: [[TMP0:%.*]] = add <8 x i8> %0, [[VSRA_N]]
  // LLVM: ret <8 x i8> [[TMP0]]
}

int16x4_t test_vsra_n_s16(int16x4_t a, int16x4_t b) {
  return vsra_n_s16(a, b, 3);

  // CIR-LABEL: vsra_n_s16
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s16i x 4>

  // LLVM-LABEL: test_vsra_n_s16
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> %0 to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> %1 to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM: [[VSRA_N:%.*]] = ashr <4 x i16> [[TMP3]], splat (i16 3)
  // LLVM: [[TMP4:%.*]] = add <4 x i16> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <4 x i16> [[TMP4]]
}


int32x2_t test_vsra_n_s32(int32x2_t a, int32x2_t b) {
  return vsra_n_s32(a, b, 3);

  // CIR-LABEL: vsra_n_s32
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s32i x 2>

  // LLVM-LABEL: test_vsra_n_s32
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> %0 to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> %1 to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM: [[VSRA_N:%.*]] = ashr <2 x i32> [[TMP3]], splat (i32 3)
  // LLVM: [[TMP4:%.*]] = add <2 x i32> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <2 x i32> [[TMP4]]
}

int8x16_t test_vsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vsraq_n_s8(a, b, 3);

  // CIR-LABEL: vsraq_n_s8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s8i x 16>

  // LLVM-LABEL: test_vsraq_n_s8
  // LLVM: [[VSRA_N:%.*]] = ashr <16 x i8> %1, splat (i8 3)
  // LLVM: [[TMP0:%.*]] = add <16 x i8> %0, [[VSRA_N]]
  // LLVM: ret <16 x i8> [[TMP0]]
}

int16x8_t test_vsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vsraq_n_s16(a, b, 3);

  // CIR-LABEL: vsraq_n_s16
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_vsraq_n_s16
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM: [[VSRA_N:%.*]] = ashr <8 x i16> [[TMP3]], splat (i16 3)
  // LLVM: [[TMP4:%.*]] = add <8 x i16> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <8 x i16> [[TMP4]]
}

int32x4_t test_vsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vsraq_n_s32(a, b, 3);

  // CIR-LABEL: vsraq_n_s32
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s32i x 4>

  // LLVM-LABEL: test_vsraq_n_s32
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM: [[VSRA_N:%.*]] = ashr <4 x i32> [[TMP3]], splat (i32 3)
  // LLVM: [[TMP4:%.*]] = add <4 x i32> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <4 x i32> [[TMP4]]
}

int64x2_t test_vsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vsraq_n_s64(a, b, 3);

  // CIR-LABEL: vsraq_n_s64
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s64i x 2>

  // LLVM-LABEL: test_vsraq_n_s64
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i64> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM: [[VSRA_N:%.*]] = ashr <2 x i64> [[TMP3]], splat (i64 3)
  // LLVM: [[TMP4:%.*]] = add <2 x i64> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <2 x i64> [[TMP4]]
}

uint8x8_t test_vsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vsra_n_u8(a, b, 3);

  // CIR-LABEL: vsra_n_u8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u8i x 8>) -> !cir.vector<!u8i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u8i x 8>

  // LLVM-LABEL: @test_vsra_n_u8(
  // LLVM: [[VSRA_N:%.*]] = lshr <8 x i8> %1, splat (i8 3)
  // LLVM: [[TMP0:%.*]] = add <8 x i8> %0, [[VSRA_N]]
  // LLVM: ret <8 x i8> [[TMP0]]
}

uint16x4_t test_vsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vsra_n_u16(a, b, 3);

  // CIR-LABEL: vsra_n_u16
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u16i x 4>) -> !cir.vector<!u16i x 4>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u16i x 4>

  // LLVM-LABEL: test_vsra_n_u16
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> %0 to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> %1 to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM: [[VSRA_N:%.*]] = lshr <4 x i16> [[TMP3]], splat (i16 3)
  // LLVM: [[TMP4:%.*]] = add <4 x i16> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <4 x i16> [[TMP4]]
}

uint32x2_t test_vsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vsra_n_u32(a, b, 3);

  // CIR-LABEL: vsra_n_u32
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u32i x 2>) -> !cir.vector<!u32i x 2>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u32i x 2>

  // LLVM-LABEL: test_vsra_n_u32
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> %0 to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> %1 to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM: [[VSRA_N:%.*]] = lshr <2 x i32> [[TMP3]], splat (i32 3)
  // LLVM: [[TMP4:%.*]] = add <2 x i32> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <2 x i32> [[TMP4]]
}

uint8x16_t test_vsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vsraq_n_u8(a, b, 3);

  // CIR-LABEL: vsraq_n_u8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u8i x 16>) -> !cir.vector<!u8i x 16>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u8i x 16>

  // LLVM-LABEL: test_vsraq_n_u8
  // LLVM: [[VSRA_N:%.*]] = lshr <16 x i8> %1, splat (i8 3)
  // LLVM: [[TMP0:%.*]] = add <16 x i8> %0, [[VSRA_N]]
  // LLVM: ret <16 x i8> [[TMP0]]
}

uint16x8_t test_vsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vsraq_n_u16(a, b, 3);

  // CIR-LABEL: vsraq_n_u16
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u16i x 8>

  // LLVM-LABEL: test_vsraq_n_u16
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <8 x i16> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM: [[VSRA_N:%.*]] = lshr <8 x i16> [[TMP3]], splat (i16 3)
  // LLVM: [[TMP4:%.*]] = add <8 x i16> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <8 x i16> [[TMP4]]
}

uint32x4_t test_vsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vsraq_n_u32(a, b, 3);

  // CIR-LABEL: vsraq_n_u32
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u32i x 4>

  // LLVM-LABEL: test_vsraq_n_u32
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i32> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM: [[VSRA_N:%.*]] = lshr <4 x i32> [[TMP3]], splat (i32 3)
  // LLVM: [[TMP4:%.*]] = add <4 x i32> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <4 x i32> [[TMP4]]
}

uint64x2_t test_vsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vsraq_n_u64(a, b, 3);

  // CIR-LABEL: vsraq_n_u64
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u64i x 2>

  // LLVM-LABEL: test_vsraq_n_u64
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> %0 to <16 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i64> %1 to <16 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM: [[VSRA_N:%.*]] = lshr <2 x i64> [[TMP3]], splat (i64 3)
  // LLVM: [[TMP4:%.*]] = add <2 x i64> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <2 x i64> [[TMP4]]
}

int8x8_t test_vrshr_n_s8(int8x8_t a) {
  return vrshr_n_s8(a, 3);

  // CIR-LABEL: vrshr_n_s8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i]> : !cir.vector<!s8i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vrshr_n_s8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> [[A]], <8 x i8> splat (i8 -3))
  // LLVM: ret <8 x i8> [[VRSHR_N]]
}

uint8x8_t test_vrshr_n_u8(uint8x8_t a) {
  return vrshr_n_u8(a, 3);

  // CIR-LABEL: vrshr_n_u8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i]> : !cir.vector<!s8i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vrshr_n_u8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> [[A]], <8 x i8> splat (i8 -3))
  // LLVM: ret <8 x i8> [[VRSHR_N]]
}

int16x4_t test_vrshr_n_s16(int16x4_t a) {
  return vrshr_n_s16(a, 3);

  // CIR-LABEL: vrshr_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s16i, #cir.int<-3> : !s16i,
  // CIR-SAME: #cir.int<-3> : !s16i, #cir.int<-3> : !s16i]> : !cir.vector<!s16i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vrshr_n_s16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
  // LLVM: ret <4 x i16> [[VRSHR_N1]]
}

uint16x4_t test_vrshr_n_u16(uint16x4_t a) {
  return vrshr_n_u16(a, 3);

  // CIR-LABEL: vrshr_n_u16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s16i, #cir.int<-3> : !s16i,
  // CIR-SAME: #cir.int<-3> : !s16i, #cir.int<-3> : !s16i]> : !cir.vector<!s16i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vrshr_n_u16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
  // LLVM: ret <4 x i16> [[VRSHR_N1]]
}

int32x2_t test_vrshr_n_s32(int32x2_t a) {
  return vrshr_n_s32(a, 3);

  // CIR-LABEL: vrshr_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s32i, #cir.int<-3> : !s32i]> : !cir.vector<!s32i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vrshr_n_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
  // LLVM: ret <2 x i32> [[VRSHR_N1]]
}

uint32x2_t test_vrshr_n_u32(uint32x2_t a) {
  return vrshr_n_u32(a, 3);

  // CIR-LABEL: vrshr_n_u32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s32i, #cir.int<-3> : !s32i]> : !cir.vector<!s32i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vrshr_n_u32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
  // LLVM: ret <2 x i32> [[VRSHR_N1]]
}

int64x1_t test_vrshr_n_s64(int64x1_t a) {
  return vrshr_n_s64(a, 3);

  // CIR-LABEL: vrshr_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s64i]> : !cir.vector<!s64i x 1>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>

  // LLVM: {{.*}}@test_vrshr_n_s64(<1 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -3))
  // LLVM: ret <1 x i64> [[VRSHR_N1]]
}

uint64x1_t test_vrshr_n_u64(uint64x1_t a) {
  return vrshr_n_u64(a, 3);

  // CIR-LABEL: vrshr_n_u64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s64i]> : !cir.vector<!s64i x 1>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!u64i x 1>

  // LLVM: {{.*}}@test_vrshr_n_u64(<1 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <1 x i64> [[A]] to <8 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -3))
  // LLVM: ret <1 x i64> [[VRSHR_N1]]
}

int8x16_t test_vrshrq_n_s8(int8x16_t a) {
  return vrshrq_n_s8(a, 3);

  // CIR-LABEL: vrshrq_n_s8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i]> : !cir.vector<!s8i x 16>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>

  // LLVM: {{.*}}@test_vrshrq_n_s8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> [[A]], <16 x i8> splat (i8 -3))
  // LLVM: ret <16 x i8> [[VRSHR_N]]
}

uint8x16_t test_vrshrq_n_u8(uint8x16_t a) {
  return vrshrq_n_u8(a, 3);

  // CIR-LABEL: vrshrq_n_u8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i,
  // CIR-SAME: #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i, #cir.int<-3> : !s8i]> : !cir.vector<!s8i x 16>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}@test_vrshrq_n_u8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> [[A]], <16 x i8> splat (i8 -3))
  // LLVM: ret <16 x i8> [[VRSHR_N]]
}

int16x8_t test_vrshrq_n_s16(int16x8_t a) {
  return vrshrq_n_s16(a, 3);

  // CIR-LABEL: vrshrq_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i,
  // CIR-SAME: #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}@test_vrshrq_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
  // LLVM: ret <8 x i16> [[VRSHR_N1]]
}

uint16x8_t test_vrshrq_n_u16(uint16x8_t a) {
  return vrshrq_n_u16(a, 3);

  // CIR-LABEL: vrshrq_n_u16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i,
  // CIR-SAME: #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i, #cir.int<-3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}@test_vrshrq_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
  // LLVM: ret <8 x i16> [[VRSHR_N1]]
}

int32x4_t test_vrshrq_n_s32(int32x4_t a) {
  return vrshrq_n_s32(a, 3);

  // CIR-LABEL: vrshrq_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s32i, #cir.int<-3> : !s32i, #cir.int<-3> : !s32i, #cir.int<-3> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}@test_vrshrq_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
  // LLVM: ret <4 x i32> [[VRSHR_N1]]
}

uint32x4_t test_vrshrq_n_u32(uint32x4_t a) {
  return vrshrq_n_u32(a, 3);

  // CIR-LABEL: vrshrq_n_u32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s32i, #cir.int<-3> : !s32i,
  // CIR-SAME: #cir.int<-3> : !s32i, #cir.int<-3> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vrshrq_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
  // LLVM: ret <4 x i32> [[VRSHR_N1]]
}

int64x2_t test_vrshrq_n_s64(int64x2_t a) {
  return vrshrq_n_s64(a, 3);

  // CIR-LABEL: vrshrq_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s64i, #cir.int<-3> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}@test_vrshrq_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
  // LLVM: ret <2 x i64> [[VRSHR_N1]]
}

uint64x2_t test_vrshrq_n_u64(uint64x2_t a) {
  return vrshrq_n_u64(a, 3);

  // CIR-LABEL: vrshrq_n_u64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<-3> : !s64i, #cir.int<-3> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!u64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}@test_vrshrq_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
  // LLVM: ret <2 x i64> [[VRSHR_N1]]
}

int8x8_t test_vrsra_n_s8(int8x8_t a, int8x8_t b) {
  return vrsra_n_s8(a, b, 3);

  // CIR-LABEL: vrsra_n_s8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[splat]] : (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s8i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VRSHR_N]]) : !cir.vector<!s8i x 8>

  // LLVM-LABEL: test_vrsra_n_s8
  // LLVM:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.srshl.v8i8(<8 x i8> %1, <8 x i8> splat (i8 -3))
  // LLVM:   [[TMP0:%.*]] = add <8 x i8> %0, [[VRSHR_N]]
  // LLVM:   ret <8 x i8> [[TMP0]]
}

int16x4_t test_vrsra_n_s16(int16x4_t a, int16x4_t b) {
  return vrsra_n_s16(a, b, 3);

  // CIR-LABEL: vrsra_n_s16
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s16i x 4>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s16i x 4>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s16i x 4>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s16i x 4>

  // LLVM-LABEL: test_vrsra_n_s16
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.srshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[TMP3:%.*]] = add <4 x i16> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <4 x i16> [[TMP3]]
}

int32x2_t test_vrsra_n_s32(int32x2_t a, int32x2_t b) {
  return vrsra_n_s32(a, b, 3);

  // CIR-LABEL: vrsra_n_s32
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s32i x 2>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s32i x 2>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s32i x 2>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s32i x 2>

  // LLVM-LABEL: test_vrsra_n_s32
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.srshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[TMP3:%.*]] = add <2 x i32> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <2 x i32> [[TMP3]]
}

int8x16_t test_vrsraq_n_s8(int8x16_t a, int8x16_t b) {
  return vrsraq_n_s8(a, b, 3);

  // CIR-LABEL: vrsraq_n_s8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[splat]] : (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!s8i x 16>
  // CIR: cir.binop(add, {{%.*}}, [[VRSHR_N]]) : !cir.vector<!s8i x 16>

  // LLVM-LABEL: test_vrsraq_n_s8
  // LLVM:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.srshl.v16i8(<16 x i8> %1, <16 x i8> splat (i8 -3))
  // LLVM:   [[TMP0:%.*]] = add <16 x i8> %0, [[VRSHR_N]]
  // LLVM:   ret <16 x i8> [[TMP0]]
}

int16x8_t test_vrsraq_n_s16(int16x8_t a, int16x8_t b) {
  return vrsraq_n_s16(a, b, 3);

  // CIR-LABEL: vrsraq_n_s16
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s16i x 8>

  // LLVM-LABEL: test_vrsraq_n_s16
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.srshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[TMP3:%.*]] = add <8 x i16> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <8 x i16> [[TMP3]]
}

int32x4_t test_vrsraq_n_s32(int32x4_t a, int32x4_t b) {
  return vrsraq_n_s32(a, b, 3);

  // CIR-LABEL: vrsraq_n_s32
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s32i x 4>

  // LLVM-LABEL: test_vrsraq_n_s32
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.srshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[TMP3:%.*]] = add <4 x i32> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <4 x i32> [[TMP3]]
}

int64x2_t test_vrsraq_n_s64(int64x2_t a, int64x2_t b) {
  return vrsraq_n_s64(a, b, 3);

  // CIR-LABEL: vrsraq_n_s64
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s64i x 2>

  // LLVM-LABEL: test_vrsraq_n_s64
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.srshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[TMP3:%.*]] = add <2 x i64> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <2 x i64> [[TMP3]]
}

uint8x8_t test_vrsra_n_u8(uint8x8_t a, uint8x8_t b) {
  return vrsra_n_u8(a, b, 3);

  // CIR-LABEL: vrsra_n_u8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[splat]] : (!cir.vector<!u8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!u8i x 8>
  // CIR: cir.binop(add, {{%.*}}, [[VRSHR_N]]) : !cir.vector<!u8i x 8>

  // LLVM-LABEL: test_vrsra_n_u8
  // LLVM:   [[VRSHR_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.urshl.v8i8(<8 x i8> %1, <8 x i8> splat (i8 -3))
  // LLVM:   [[TMP0:%.*]] = add <8 x i8> %0, [[VRSHR_N]]
  // LLVM:   ret <8 x i8> [[TMP0]]
}

uint16x4_t test_vrsra_n_u16(uint16x4_t a, uint16x4_t b) {
  return vrsra_n_u16(a, b, 3);

  // CIR-LABEL: vrsra_n_u16
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u16i x 4>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!u16i x 4>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u16i x 4>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u16i x 4>

  // LLVM-LABEL: test_vrsra_n_u16
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
  // LLVM:   [[VRSHR_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.urshl.v4i16(<4 x i16> [[VRSHR_N]], <4 x i16> splat (i16 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[TMP3:%.*]] = add <4 x i16> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <4 x i16> [[TMP3]]
}

uint32x2_t test_vrsra_n_u32(uint32x2_t a, uint32x2_t b) {
  return vrsra_n_u32(a, b, 3);

  // CIR-LABEL: vrsra_n_u32
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u32i x 2>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!u32i x 2>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u32i x 2>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u32i x 2>

  // LLVM-LABEL: test_vrsra_n_u32
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
  // LLVM:   [[VRSHR_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.urshl.v2i32(<2 x i32> [[VRSHR_N]], <2 x i32> splat (i32 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[TMP3:%.*]] = add <2 x i32> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <2 x i32> [[TMP3]]
}

uint8x16_t test_vrsraq_n_u8(uint8x16_t a, uint8x16_t b) {
  return vrsraq_n_u8(a, b, 3);

  // CIR-LABEL: vrsraq_n_u8
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" {{%.*}}, [[splat]] : (!cir.vector<!u8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!u8i x 16>
  // CIR: cir.binop(add, {{%.*}}, [[VRSHR_N]]) : !cir.vector<!u8i x 16>

  // LLVM-LABEL: test_vrsraq_n_u8
  // LLVM:   [[VRSHR_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.urshl.v16i8(<16 x i8> %1, <16 x i8> splat (i8 -3))
  // LLVM:   [[TMP0:%.*]] = add <16 x i8> %0, [[VRSHR_N]]
  // LLVM:   ret <16 x i8> [[TMP0]]
}

uint16x8_t test_vrsraq_n_u16(uint16x8_t a, uint16x8_t b) {
  return vrsraq_n_u16(a, b, 3);

  // CIR-LABEL: vrsraq_n_u16
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!u16i x 8>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u16i x 8>

  // LLVM-LABEL: test_vrsraq_n_u16
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i16> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
  // LLVM:   [[VRSHR_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.urshl.v8i16(<8 x i16> [[VRSHR_N]], <8 x i16> splat (i16 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[TMP3:%.*]] = add <8 x i16> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <8 x i16> [[TMP3]]
}

uint32x4_t test_vrsraq_n_u32(uint32x4_t a, uint32x4_t b) {
  return vrsraq_n_u32(a, b, 3);

  // CIR-LABEL: vrsraq_n_u32
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u32i x 4>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!u32i x 4>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u32i x 4>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u32i x 4>

  // LLVM-LABEL: test_vrsraq_n_u32
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i32> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
  // LLVM:   [[VRSHR_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.urshl.v4i32(<4 x i32> [[VRSHR_N]], <4 x i32> splat (i32 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[TMP3:%.*]] = add <4 x i32> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <4 x i32> [[TMP3]]
}

uint64x2_t test_vrsraq_n_u64(uint64x2_t a, uint64x2_t b) {
  return vrsraq_n_u64(a, b, 3);

  // CIR-LABEL: vrsraq_n_u64
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u64i x 2>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!u64i x 2>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u64i x 2>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u64i x 2>

  // LLVM-LABEL: test_vrsraq_n_u64
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> %0 to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i64> %1 to <16 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
  // LLVM:   [[VRSHR_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.urshl.v2i64(<2 x i64> [[VRSHR_N]], <2 x i64> splat (i64 -3))
  // LLVM:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[TMP3:%.*]] = add <2 x i64> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <2 x i64> [[TMP3]]
}

// NYI-LABEL: @test_vsri_n_s8(
// NYI:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSRI_N]]
// int8x8_t test_vsri_n_s8(int8x8_t a, int8x8_t b) {
//   return vsri_n_s8(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 3)
// NYI:   ret <4 x i16> [[VSRI_N2]]
// int16x4_t test_vsri_n_s16(int16x4_t a, int16x4_t b) {
//   return vsri_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   [[VSRI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32> [[VSRI_N]], <2 x i32> [[VSRI_N1]], i32 3)
// NYI:   ret <2 x i32> [[VSRI_N2]]
// int32x2_t test_vsri_n_s32(int32x2_t a, int32x2_t b) {
//   return vsri_n_s32(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_s8(
// NYI:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSRI_N]]
// int8x16_t test_vsriq_n_s8(int8x16_t a, int8x16_t b) {
//   return vsriq_n_s8(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 3)
// NYI:   ret <8 x i16> [[VSRI_N2]]
// int16x8_t test_vsriq_n_s16(int16x8_t a, int16x8_t b) {
//   return vsriq_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// NYI:   [[VSRI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> [[VSRI_N]], <4 x i32> [[VSRI_N1]], i32 3)
// NYI:   ret <4 x i32> [[VSRI_N2]]
// int32x4_t test_vsriq_n_s32(int32x4_t a, int32x4_t b) {
//   return vsriq_n_s32(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// NYI:   [[VSRI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> [[VSRI_N]], <2 x i64> [[VSRI_N1]], i32 3)
// NYI:   ret <2 x i64> [[VSRI_N2]]
// int64x2_t test_vsriq_n_s64(int64x2_t a, int64x2_t b) {
//   return vsriq_n_s64(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_u8(
// NYI:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSRI_N]]
// uint8x8_t test_vsri_n_u8(uint8x8_t a, uint8x8_t b) {
//   return vsri_n_u8(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 3)
// NYI:   ret <4 x i16> [[VSRI_N2]]
// uint16x4_t test_vsri_n_u16(uint16x4_t a, uint16x4_t b) {
//   return vsri_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   [[VSRI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsri.v2i32(<2 x i32> [[VSRI_N]], <2 x i32> [[VSRI_N1]], i32 3)
// NYI:   ret <2 x i32> [[VSRI_N2]]
// uint32x2_t test_vsri_n_u32(uint32x2_t a, uint32x2_t b) {
//   return vsri_n_u32(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_u8(
// NYI:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSRI_N]]
// uint8x16_t test_vsriq_n_u8(uint8x16_t a, uint8x16_t b) {
//   return vsriq_n_u8(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 3)
// NYI:   ret <8 x i16> [[VSRI_N2]]
// uint16x8_t test_vsriq_n_u16(uint16x8_t a, uint16x8_t b) {
//   return vsriq_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// NYI:   [[VSRI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsri.v4i32(<4 x i32> [[VSRI_N]], <4 x i32> [[VSRI_N1]], i32 3)
// NYI:   ret <4 x i32> [[VSRI_N2]]
// uint32x4_t test_vsriq_n_u32(uint32x4_t a, uint32x4_t b) {
//   return vsriq_n_u32(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// NYI:   [[VSRI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> [[VSRI_N]], <2 x i64> [[VSRI_N1]], i32 3)
// NYI:   ret <2 x i64> [[VSRI_N2]]
// uint64x2_t test_vsriq_n_u64(uint64x2_t a, uint64x2_t b) {
//   return vsriq_n_u64(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_p8(
// NYI:   [[VSRI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsri.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSRI_N]]
// poly8x8_t test_vsri_n_p8(poly8x8_t a, poly8x8_t b) {
//   return vsri_n_p8(a, b, 3);
// }

// NYI-LABEL: @test_vsri_n_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsri.v4i16(<4 x i16> [[VSRI_N]], <4 x i16> [[VSRI_N1]], i32 15)
// NYI:   ret <4 x i16> [[VSRI_N2]]
// poly16x4_t test_vsri_n_p16(poly16x4_t a, poly16x4_t b) {
//   return vsri_n_p16(a, b, 15);
// }

// NYI-LABEL: @test_vsriq_n_p8(
// NYI:   [[VSRI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsri.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSRI_N]]
// poly8x16_t test_vsriq_n_p8(poly8x16_t a, poly8x16_t b) {
//   return vsriq_n_p8(a, b, 3);
// }

// NYI-LABEL: @test_vsriq_n_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSRI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsri.v8i16(<8 x i16> [[VSRI_N]], <8 x i16> [[VSRI_N1]], i32 15)
// NYI:   ret <8 x i16> [[VSRI_N2]]
// poly16x8_t test_vsriq_n_p16(poly16x8_t a, poly16x8_t b) {
//   return vsriq_n_p16(a, b, 15);
// }

// NYI-LABEL: @test_vsli_n_s8(
// NYI:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSLI_N]]
// int8x8_t test_vsli_n_s8(int8x8_t a, int8x8_t b) {
//   return vsli_n_s8(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 3)
// NYI:   ret <4 x i16> [[VSLI_N2]]
// int16x4_t test_vsli_n_s16(int16x4_t a, int16x4_t b) {
//   return vsli_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   [[VSLI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> [[VSLI_N]], <2 x i32> [[VSLI_N1]], i32 3)
// NYI:   ret <2 x i32> [[VSLI_N2]]
// int32x2_t test_vsli_n_s32(int32x2_t a, int32x2_t b) {
//   return vsli_n_s32(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_s8(
// NYI:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSLI_N]]
// int8x16_t test_vsliq_n_s8(int8x16_t a, int8x16_t b) {
//   return vsliq_n_s8(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 3)
// NYI:   ret <8 x i16> [[VSLI_N2]]
// int16x8_t test_vsliq_n_s16(int16x8_t a, int16x8_t b) {
//   return vsliq_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// NYI:   [[VSLI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> [[VSLI_N]], <4 x i32> [[VSLI_N1]], i32 3)
// NYI:   ret <4 x i32> [[VSLI_N2]]
// int32x4_t test_vsliq_n_s32(int32x4_t a, int32x4_t b) {
//   return vsliq_n_s32(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 3)
// NYI:   ret <2 x i64> [[VSLI_N2]]
// int64x2_t test_vsliq_n_s64(int64x2_t a, int64x2_t b) {
//   return vsliq_n_s64(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_u8(
// NYI:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSLI_N]]
// uint8x8_t test_vsli_n_u8(uint8x8_t a, uint8x8_t b) {
//   return vsli_n_u8(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 3)
// NYI:   ret <4 x i16> [[VSLI_N2]]
// uint16x4_t test_vsli_n_u16(uint16x4_t a, uint16x4_t b) {
//   return vsli_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   [[VSLI_N2:%.*]] = call <2 x i32> @llvm.aarch64.neon.vsli.v2i32(<2 x i32> [[VSLI_N]], <2 x i32> [[VSLI_N1]], i32 3)
// NYI:   ret <2 x i32> [[VSLI_N2]]
// uint32x2_t test_vsli_n_u32(uint32x2_t a, uint32x2_t b) {
//   return vsli_n_u32(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_u8(
// NYI:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSLI_N]]
// uint8x16_t test_vsliq_n_u8(uint8x16_t a, uint8x16_t b) {
//   return vsliq_n_u8(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 3)
// NYI:   ret <8 x i16> [[VSLI_N2]]
// uint16x8_t test_vsliq_n_u16(uint16x8_t a, uint16x8_t b) {
//   return vsliq_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x i32>
// NYI:   [[VSLI_N2:%.*]] = call <4 x i32> @llvm.aarch64.neon.vsli.v4i32(<4 x i32> [[VSLI_N]], <4 x i32> [[VSLI_N1]], i32 3)
// NYI:   ret <4 x i32> [[VSLI_N2]]
// uint32x4_t test_vsliq_n_u32(uint32x4_t a, uint32x4_t b) {
//   return vsliq_n_u32(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsli.v2i64(<2 x i64> [[VSLI_N]], <2 x i64> [[VSLI_N1]], i32 3)
// NYI:   ret <2 x i64> [[VSLI_N2]]
// uint64x2_t test_vsliq_n_u64(uint64x2_t a, uint64x2_t b) {
//   return vsliq_n_u64(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_p8(
// NYI:   [[VSLI_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.vsli.v8i8(<8 x i8> %a, <8 x i8> %b, i32 3)
// NYI:   ret <8 x i8> [[VSLI_N]]
// poly8x8_t test_vsli_n_p8(poly8x8_t a, poly8x8_t b) {
//   return vsli_n_p8(a, b, 3);
// }

// NYI-LABEL: @test_vsli_n_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <4 x i16> @llvm.aarch64.neon.vsli.v4i16(<4 x i16> [[VSLI_N]], <4 x i16> [[VSLI_N1]], i32 15)
// NYI:   ret <4 x i16> [[VSLI_N2]]
// poly16x4_t test_vsli_n_p16(poly16x4_t a, poly16x4_t b) {
//   return vsli_n_p16(a, b, 15);
// }

// NYI-LABEL: @test_vsliq_n_p8(
// NYI:   [[VSLI_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.vsli.v16i8(<16 x i8> %a, <16 x i8> %b, i32 3)
// NYI:   ret <16 x i8> [[VSLI_N]]
// poly8x16_t test_vsliq_n_p8(poly8x16_t a, poly8x16_t b) {
//   return vsliq_n_p8(a, b, 3);
// }

// NYI-LABEL: @test_vsliq_n_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VSLI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   [[VSLI_N2:%.*]] = call <8 x i16> @llvm.aarch64.neon.vsli.v8i16(<8 x i16> [[VSLI_N]], <8 x i16> [[VSLI_N1]], i32 15)
// NYI:   ret <8 x i16> [[VSLI_N2]]
// poly16x8_t test_vsliq_n_p16(poly16x8_t a, poly16x8_t b) {
//   return vsliq_n_p16(a, b, 15);
// }

uint8x8_t test_vqshlu_n_s8(int8x8_t a) {
  return vqshlu_n_s8(a, 3);

 // CIR-LABEL: vqshlu_n_s8
 // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
 // CIR: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i]> : !cir.vector<!s8i x 8>
 // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
 // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!u8i x 8>

 // LLVM: {{.*}}@test_vqshlu_n_s8(<8 x i8>{{.*}}[[A:%.*]])
 // LLVM: [[VQSHLU_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8> [[A]], <8 x i8> splat (i8 3))
 // LLVM: ret <8 x i8> [[VQSHLU_N]]
}

uint16x4_t test_vqshlu_n_s16(int16x4_t a) {
  return vqshlu_n_s16(a, 3);

  // CIR-LABEL: vqshlu_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME:#cir.int<3> : !s16i, #cir.int<3> : !s16i]> : !cir.vector<!s16i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vqshlu_n_s16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM: [[VQSHLU_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16> [[VQSHLU_N]], <4 x i16> splat (i16 3))
  // LLVM: ret <4 x i16> [[VQSHLU_N1]]
}

uint32x2_t test_vqshlu_n_s32(int32x2_t a) {
  return vqshlu_n_s32(a, 3);

  // CIR-LABEL: vqshlu_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i]> : !cir.vector<!s32i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vqshlu_n_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM: [[VQSHLU_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshlu.v2i32(<2 x i32> [[VQSHLU_N]], <2 x i32> splat (i32 3))
}

uint8x16_t test_vqshluq_n_s8(int8x16_t a) {
  return vqshluq_n_s8(a, 3);

  // CIR-LABEL: vqshluq_n_s8
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i,
  // CIR-SAME: #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i, #cir.int<3> : !s8i]> : !cir.vector<!s8i x 16>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s8i x 16>, !cir.vector<!s8i x 16>) -> !cir.vector<!u8i x 16>

  // LLVM: {{.*}}@test_vqshluq_n_s8(<16 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VQSHLUQ_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshlu.v16i8(<16 x i8> [[A]], <16 x i8> splat (i8 3))
  // LLVM: ret <16 x i8> [[VQSHLUQ_N]]
}

uint16x8_t test_vqshluq_n_s16(int16x8_t a) {
  return vqshluq_n_s16(a, 3);

  // CIR-LABEL: vqshluq_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !cir.vector<!s16i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}@test_vqshluq_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VQSHLUQ_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM: [[VQSHLUQ_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshlu.v8i16(<8 x i16> [[VQSHLUQ_N]], <8 x i16> splat (i16 3))
  // LLVM: ret <8 x i16> [[VQSHLUQ_N1]]
}

uint32x4_t test_vqshluq_n_s32(int32x4_t a) {
  return vqshluq_n_s32(a, 3);

  // CIR-LABEL: vqshluq_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s32i, #cir.int<3> : !s32i,
  // CIR-SAME: #cir.int<3> : !s32i, #cir.int<3> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vqshluq_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VQSHLUQ_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM: [[VQSHLUQ_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshlu.v4i32(<4 x i32> [[VQSHLUQ_N]], <4 x i32> splat (i32 3))
  // LLVM: ret <4 x i32> [[VQSHLUQ_N1]]
}

uint64x2_t test_vqshluq_n_s64(int64x2_t a) {
  return vqshluq_n_s64(a, 3);

  // CIR-LABEL: vqshluq_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s64i, #cir.int<3> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{%.*}}, [[AMT]] :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}@test_vqshluq_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VQSHLUQ_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM: [[VQSHLUQ_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshlu.v2i64(<2 x i64> [[VQSHLUQ_N]], <2 x i64> splat (i64 3))
  // LLVM: ret <2 x i64> [[VQSHLUQ_N1]]
}

int8x8_t test_vshrn_n_s16(int16x8_t a) {
  return vshrn_n_s16(a, 3);

  // CIR-LABEL: vshrn_n_s16
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!s16i x 8>, [[AMT]] : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!s16i x 8>), !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vshrn_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[TMP2:%.*]] = ashr <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
  // LLVM:   ret <8 x i8> [[VSHRN_N]]
}

int16x4_t test_vshrn_n_s32(int32x4_t a) {
  return vshrn_n_s32(a, 9);

  // CIR-LABEL: vshrn_n_s32
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<9> : !s32i, #cir.int<9> : !s32i,
  // CIR-SAME: #cir.int<9> : !s32i, #cir.int<9> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!s32i x 4>, [[AMT]] : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!s32i x 4>), !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vshrn_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], splat (i32 9)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
  // LLVM:   ret <4 x i16> [[VSHRN_N]]
}

int32x2_t test_vshrn_n_s64(int64x2_t a) {
  return vshrn_n_s64(a, 19);

  // CIR-LABEL: vshrn_n_s64
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<19> : !s64i, #cir.int<19> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!s64i x 2>, [[AMT]] : !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!s64i x 2>), !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vshrn_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[TMP2:%.*]] = ashr <2 x i64> [[TMP1]], splat (i64 19)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
  // LLVM:   ret <2 x i32> [[VSHRN_N]]
}

uint8x8_t test_vshrn_n_u16(uint16x8_t a) {
  return vshrn_n_u16(a, 3);

  // CIR-LABEL: vshrn_n_u16
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i,
  // CIR-SAME: #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i]> : !cir.vector<!u16i x 8>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!u16i x 8>, [[AMT]] : !cir.vector<!u16i x 8>) -> !cir.vector<!u16i x 8>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!u16i x 8>), !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vshrn_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[TMP2:%.*]] = lshr <8 x i16> [[TMP1]], splat (i16 3)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
  // LLVM:   ret <8 x i8> [[VSHRN_N]]
}

uint16x4_t test_vshrn_n_u32(uint32x4_t a) {
  return vshrn_n_u32(a, 9);

  // CIR-LABEL: vshrn_n_u32
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u32i x 4>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<9> : !u32i, #cir.int<9> : !u32i,
  // CIR-SAME: #cir.int<9> : !u32i, #cir.int<9> : !u32i]> : !cir.vector<!u32i x 4>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!u32i x 4>, [[AMT]] : !cir.vector<!u32i x 4>) -> !cir.vector<!u32i x 4>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!u32i x 4>), !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vshrn_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[TMP2:%.*]] = lshr <4 x i32> [[TMP1]], splat (i32 9)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
  // LLVM:   ret <4 x i16> [[VSHRN_N]]
}

uint32x2_t test_vshrn_n_u64(uint64x2_t a) {
  return vshrn_n_u64(a, 19);

  // CIR-LABEL: vshrn_n_u64
  // CIR: [[TGT:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u64i x 2>
  // CIR: [[AMT:%.*]] = cir.const #cir.const_vector<[#cir.int<19> : !u64i, #cir.int<19> : !u64i]> : !cir.vector<!u64i x 2>
  // CIR: [[RES:%.*]] = cir.shift(right, [[TGT]] : !cir.vector<!u64i x 2>, [[AMT]] : !cir.vector<!u64i x 2>) -> !cir.vector<!u64i x 2>
  // CIR: {{%.*}} = cir.cast(integral, [[RES]] : !cir.vector<!u64i x 2>), !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vshrn_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[TMP2:%.*]] = lshr <2 x i64> [[TMP1]], splat (i64 19)
  // LLVM:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
}

// NYI-LABEL: @test_vshrn_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[TMP2:%.*]] = ashr <8 x i16> [[TMP1]], <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
// NYI:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VSHRN_N]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vshrn_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vshrn_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vshrn_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[TMP2:%.*]] = ashr <4 x i32> [[TMP1]], <i32 9, i32 9, i32 9, i32 9>
// NYI:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VSHRN_N]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vshrn_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vshrn_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vshrn_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[TMP2:%.*]] = ashr <2 x i64> [[TMP1]], <i64 19, i64 19>
// NYI:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VSHRN_N]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vshrn_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vshrn_high_n_s64(a, b, 19);
// }

// NYI-LABEL: @test_vshrn_high_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[TMP2:%.*]] = lshr <8 x i16> [[TMP1]], <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
// NYI:   [[VSHRN_N:%.*]] = trunc <8 x i16> [[TMP2]] to <8 x i8>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VSHRN_N]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// uint8x16_t test_vshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
//   return vshrn_high_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vshrn_high_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[TMP2:%.*]] = lshr <4 x i32> [[TMP1]], <i32 9, i32 9, i32 9, i32 9>
// NYI:   [[VSHRN_N:%.*]] = trunc <4 x i32> [[TMP2]] to <4 x i16>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VSHRN_N]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// uint16x8_t test_vshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
//   return vshrn_high_n_u32(a, b, 9);
// }

// NYI-LABEL: @test_vshrn_high_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[TMP2:%.*]] = lshr <2 x i64> [[TMP1]], <i64 19, i64 19>
// NYI:   [[VSHRN_N:%.*]] = trunc <2 x i64> [[TMP2]] to <2 x i32>
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VSHRN_N]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// uint32x4_t test_vshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
//   return vshrn_high_n_u64(a, b, 19);
// }

// NYI-LABEL: @test_vqshrun_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[VQSHRUN_N]], i32 3)
// NYI:   ret <8 x i8> [[VQSHRUN_N1]]
// uint8x8_t test_vqshrun_n_s16(int16x8_t a) {
//   return vqshrun_n_s16(a, 3);
// }

// NYI-LABEL: @test_vqshrun_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[VQSHRUN_N]], i32 9)
// NYI:   ret <4 x i16> [[VQSHRUN_N1]]
// uint16x4_t test_vqshrun_n_s32(int32x4_t a) {
//   return vqshrun_n_s32(a, 9);
// }

// NYI-LABEL: @test_vqshrun_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> [[VQSHRUN_N]], i32 19)
// NYI:   ret <2 x i32> [[VQSHRUN_N1]]
// uint32x2_t test_vqshrun_n_s64(int64x2_t a) {
//   return vqshrun_n_s64(a, 19);
// }

// NYI-LABEL: @test_vqshrun_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[VQSHRUN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRUN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vqshrun_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vqshrun_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vqshrun_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[VQSHRUN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRUN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vqshrun_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vqshrun_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vqshrun_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrun.v2i32(<2 x i64> [[VQSHRUN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRUN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vqshrun_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vqshrun_high_n_s64(a, b, 19);
// }

// NYI-LABEL: @test_vrshrn_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// NYI:   ret <8 x i8> [[VRSHRN_N1]]
// int8x8_t test_vrshrn_n_s16(int16x8_t a) {
//   return vrshrn_n_s16(a, 3);
// }

// NYI-LABEL: @test_vrshrn_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// NYI:   ret <4 x i16> [[VRSHRN_N1]]
// int16x4_t test_vrshrn_n_s32(int32x4_t a) {
//   return vrshrn_n_s32(a, 9);
// }

int32x2_t test_vrshrn_n_s64(int64x2_t a) {
  return vrshrn_n_s64(a, 19);

  // CIR-LABEL: vrshrn_n_s64
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.rshrn" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vrshrn_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VRSHRN_N1]]
}

uint8x8_t test_vrshrn_n_u16(uint16x8_t a) {
  return vrshrn_n_u16(a, 3);

  // CIR-LABEL: vrshrn_n_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.rshrn" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vrshrn_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VRSHRN_N1]]
}

uint16x4_t test_vrshrn_n_u32(uint32x4_t a) {
  return vrshrn_n_u32(a, 9);

  // CIR-LABEL: vrshrn_n_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.rshrn" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}vrshrn_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VRSHRN_N1]]
}

uint32x2_t test_vrshrn_n_u64(uint64x2_t a) {
  return vrshrn_n_u64(a, 19);

  // CIR-LABEL: vrshrn_n_u64
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.rshrn" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vrshrn_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VRSHRN_N1]]

}

// NYI-LABEL: @test_vrshrn_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vrshrn_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vrshrn_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vrshrn_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vrshrn_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vrshrn_high_n_s64(a, b, 19);
// }

// NYI-LABEL: @test_vrshrn_high_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.rshrn.v8i8(<8 x i16> [[VRSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// uint8x16_t test_vrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
//   return vrshrn_high_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vrshrn_high_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.rshrn.v4i16(<4 x i32> [[VRSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// uint16x8_t test_vrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
//   return vrshrn_high_n_u32(a, b, 9);
// }

// NYI-LABEL: @test_vrshrn_high_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.rshrn.v2i32(<2 x i64> [[VRSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// uint32x4_t test_vrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
//   return vrshrn_high_n_u64(a, b, 19);
// }

uint8x8_t test_vqrshrun_n_s16(int16x8_t a) {
  return vqrshrun_n_s16(a, 3);
  // CIR-LABEL: test_vqrshrun_n_s16
  // CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<3> : !s32i
  // CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

  // LLVM-LABEL: @test_vqrshrun_n_s16(
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> {{%.*}} to <16 x i8>
  // LLVM:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VQRSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[VQRSHRUN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VQRSHRUN_N1]]
}

uint16x4_t test_vqrshrun_n_s32(int32x4_t a) {
  return vqrshrun_n_s32(a, 9);
  // CIR-LABEL: test_vqrshrun_n_s32
  // CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<9> : !s32i
  // CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

  // LLVM-LABEL: @test_vqrshrun_n_s32(
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> {{%.*}} to <16 x i8>
  // LLVM:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VQRSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[VQRSHRUN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VQRSHRUN_N1]]
}

uint32x2_t test_vqrshrun_n_s64(int64x2_t a) {
  return vqrshrun_n_s64(a, 19);
  // CIR-LABEL: test_vqrshrun_n_s64
  // CIR: [[INTRN_ARG1:%.*]] = cir.const #cir.int<19> : !s32i
  // CIR: [[INTRN_ARG0:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqrshrun" [[INTRN_ARG0]], [[INTRN_ARG1]] :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

  // LLVM-LABEL: @test_vqrshrun_n_s64(
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> {{%.*}} to <16 x i8>
  // LLVM:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VQRSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[VQRSHRUN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VQRSHRUN_N1]]
}

// NYI-LABEL: @test_vqrshrun_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQRSHRUN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[VQRSHRUN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRUN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vqrshrun_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vqrshrun_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vqrshrun_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQRSHRUN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[VQRSHRUN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRUN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vqrshrun_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vqrshrun_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vqrshrun_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQRSHRUN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQRSHRUN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrun.v2i32(<2 x i64> [[VQRSHRUN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRUN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vqrshrun_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vqrshrun_high_n_s64(a, b, 19);
// }

int8x8_t test_vqshrn_n_s16(int16x8_t a) {
  return vqshrn_n_s16(a, 3);

  // CIR-LABEL: vqshrn_n_s16
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!s16i x 8>, !s32i) -> !cir.vector<!s8i x 8>

  // LLVM:{{.*}}test_vqshrn_n_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VQSHRN_N1]]
}

int16x4_t test_vqshrn_n_s32(int32x4_t a) {
  return vqshrn_n_s32(a, 9);

  // CIR-LABEL: vqshrn_n_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!s32i x 4>, !s32i) -> !cir.vector<!s16i x 4>

  // LLVM:{{.*}}test_vqshrn_n_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VQSHRN_N1]]
}

int32x2_t test_vqshrn_n_s64(int64x2_t a) {
  return vqshrn_n_s64(a, 19);

  // CIR-LABEL: vqshrn_n_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!s32i x 2>

  // LLVM:{{.*}}test_vqshrn_n_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VQSHRN_N1]]
}

uint8x8_t test_vqshrn_n_u16(uint16x8_t a) {
  return vqshrn_n_u16(a, 3);

  // CIR-LABEL: vqshrn_n_u16
  // CIR: cir.llvm.intrinsic "aarch64.neon.uqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!u16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

  // LLVM:{{.*}}test_vqshrn_n_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VQSHRN_N1]]
}

uint16x4_t test_vqshrn_n_u32(uint32x4_t a) {
  return vqshrn_n_u32(a, 9);

  // CIR-LABEL: vqshrn_n_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.uqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!u32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

  // LLVM:{{.*}}test_vqshrn_n_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VQSHRN_N1]]
}

uint32x2_t test_vqshrn_n_u64(uint64x2_t a) {
  return vqshrn_n_u64(a, 19);

  // CIR-LABEL: vqshrn_n_u64
  // CIR: cir.llvm.intrinsic "aarch64.neon.uqshrn" {{%.*}}, {{%.*}} : 
  // CIR-SAME: (!cir.vector<!u64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

  // LLVM:{{.*}}test_vqshrn_n_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VQSHRN_N1]]
}

// NYI-LABEL: @test_vqshrn_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vqshrn_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vqshrn_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vqshrn_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vqshrn_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vqshrn_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vqshrn_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vqshrn_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vqshrn_high_n_s64(a, b, 19);
// }

// NYI-LABEL: @test_vqshrn_high_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[VQSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// uint8x16_t test_vqshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
//   return vqshrn_high_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vqshrn_high_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[VQSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// uint16x8_t test_vqshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
//   return vqshrn_high_n_u32(a, b, 9);
// }

// NYI-LABEL: @test_vqshrn_high_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshrn.v2i32(<2 x i64> [[VQSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// uint32x4_t test_vqshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
//   return vqshrn_high_n_u64(a, b, 19);
// }

int8x8_t test_vqrshrn_n_s16(int16x8_t a) {
  return vqrshrn_n_s16(a, 3);

  // CIR-LABEL: vqrshrn_n_s16
  // CIR: [[AMT:%.*]] = cir.const #cir.int<3> : !s32i
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!s16i x 8>, !s32i) -> !cir.vector<!s8i x 8>

  // LLVM: {{.*}}test_vqrshrn_n_s16(<8 x i16>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VQRSHRN_N1]]
}

int16x4_t test_vqrshrn_n_s32(int32x4_t a) {
  return vqrshrn_n_s32(a, 9);

  // CIR-LABEL: vqrshrn_n_s32
  // CIR: [[AMT:%.*]] = cir.const #cir.int<9> : !s32i
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !s32i) -> !cir.vector<!s16i x 4>

  // LLVM: {{.*}}test_vqrshrn_n_s32(<4 x i32>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VQRSHRN_N1]]

}

int32x2_t test_vqrshrn_n_s64(int64x2_t a) {
  return vqrshrn_n_s64(a, 19);

  // CIR-LABEL: vqrshrn_n_s64
  // CIR: [[AMT:%.*]] = cir.const #cir.int<19> : !s32
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !s32i) -> !cir.vector<!s32i x 2>

  // LLVM: {{.*}}test_vqrshrn_n_s64(<2 x i64>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VQRSHRN_N1]]
}

uint8x8_t test_vqrshrn_n_u16(uint16x8_t a) {
  return vqrshrn_n_u16(a, 3);

  // CIR-LABEL: vqrshrn_n_u16
  // CIR: [[AMT:%.*]] = cir.const #cir.int<3> : !s32
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.uqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!u16i x 8>, !s32i) -> !cir.vector<!u8i x 8>

  // LLVM: {{.*}}test_vqrshrn_n_u16(<8 x i16>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <8 x i16> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
  // LLVM:   ret <8 x i8> [[VQRSHRN_N1]]
}

uint16x4_t test_vqrshrn_n_u32(uint32x4_t a) {
  return vqrshrn_n_u32(a, 9);

  // CIR-LABEL: vqrshrn_n_u32
  // CIR: [[AMT:%.*]] = cir.const #cir.int<9> : !s32
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u32i x 4>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.uqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!u32i x 4>, !s32i) -> !cir.vector<!u16i x 4>

  // LLVM: {{.*}}test_vqrshrn_n_u32(<4 x i32>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
  // LLVM:   ret <4 x i16> [[VQRSHRN_N1]]
}

uint32x2_t test_vqrshrn_n_u64(uint64x2_t a) {
  return vqrshrn_n_u64(a, 19);

  // CIR-LABEL: vqrshrn_n_u64
  // CIR: [[AMT:%.*]] = cir.const #cir.int<19> : !s32
  // CIR: [[VQRSHRN_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u64i x 2>
  // CIR: [[VQRSHRN_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.uqrshrn" [[VQRSHRN_N]], [[AMT]] :
  // CIR-SAME: (!cir.vector<!u64i x 2>, !s32i) -> !cir.vector<!u32i x 2>

  // LLVM: {{.*}}test_vqrshrn_n_u64(<2 x i64>{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
  // LLVM:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
  // LLVM:   ret <2 x i32> [[VQRSHRN_N1]]
}

// NYI-LABEL: @test_vqrshrn_high_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// int8x16_t test_vqrshrn_high_n_s16(int8x8_t a, int16x8_t b) {
//   return vqrshrn_high_n_s16(a, b, 3);
// }

// NYI-LABEL: @test_vqrshrn_high_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// int16x8_t test_vqrshrn_high_n_s32(int16x4_t a, int32x4_t b) {
//   return vqrshrn_high_n_s32(a, b, 9);
// }

// NYI-LABEL: @test_vqrshrn_high_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// int32x4_t test_vqrshrn_high_n_s64(int32x2_t a, int64x2_t b) {
//   return vqrshrn_high_n_s64(a, b, 19);
// }

// NYI-LABEL: @test_vqrshrn_high_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQRSHRN_N1:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[VQRSHRN_N]], i32 3)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> [[VQRSHRN_N1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I]]
// uint8x16_t test_vqrshrn_high_n_u16(uint8x8_t a, uint16x8_t b) {
//   return vqrshrn_high_n_u16(a, b, 3);
// }

// NYI-LABEL: @test_vqrshrn_high_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQRSHRN_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[VQRSHRN_N]], i32 9)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i16> %a, <4 x i16> [[VQRSHRN_N1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I]]
// uint16x8_t test_vqrshrn_high_n_u32(uint16x4_t a, uint32x4_t b) {
//   return vqrshrn_high_n_u32(a, b, 9);
// }

// NYI-LABEL: @test_vqrshrn_high_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VQRSHRN_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQRSHRN_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqrshrn.v2i32(<2 x i64> [[VQRSHRN_N]], i32 19)
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i32> %a, <2 x i32> [[VQRSHRN_N1]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I]]
// uint32x4_t test_vqrshrn_high_n_u64(uint32x2_t a, uint64x2_t b) {
//   return vqrshrn_high_n_u64(a, b, 19);
// }

int16x8_t test_vshll_n_s8(int8x8_t a) {
  return vshll_n_s8(a, 3);

  // CIR-LABEL: vshll_n_s8
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s16i x 8>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i,
  // CIR-SAME: #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i, #cir.int<3> : !s16i]> : !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.shift(left, [[SHIFT_TGT]] : !cir.vector<!s16i x 8>, [[SHIFT_AMT]] : !cir.vector<!s16i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}@test_vshll_n_s8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = sext <8 x i8> [[A]] to <8 x i16>
  // LLVM:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
  // LLVM:   ret <8 x i16> [[VSHLL_N]]
}

int32x4_t test_vshll_n_s16(int16x4_t a) {
  return vshll_n_s16(a, 9);

  // CIR-LABEL: vshll_n_s16
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!s16i x 4>), !cir.vector<!s32i x 4>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<9> : !s32i, #cir.int<9> : !s32i, #cir.int<9> :
  // CIR-SAME: !s32i, #cir.int<9> : !s32i]> : !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.shift(left, [[SHIFT_TGT]] : !cir.vector<!s32i x 4>, [[SHIFT_AMT]] : !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}@test_vshll_n_s16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
  // LLVM:   ret <4 x i32> [[VSHLL_N]]
}

int64x2_t test_vshll_n_s32(int32x2_t a) {
  return vshll_n_s32(a, 19);

  // CIR-LABEL: vshll_n_s32
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!s32i x 2>), !cir.vector<!s64i x 2>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<19> : !s64i, #cir.int<19> : !s64i]> : !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.shift(left, [[SHIFT_TGT]] : !cir.vector<!s64i x 2>, [[SHIFT_AMT]] : !cir.vector<!s64i x 2>)

  // LLVM: {{.*}}@test_vshll_n_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
  // LLVM:   ret <2 x i64> [[VSHLL_N]]
}

uint16x8_t test_vshll_n_u8(uint8x8_t a) {
  return vshll_n_u8(a, 3);

  // CIR-LABEL: vshll_n_u8
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!u8i x 8>), !cir.vector<!u16i x 8>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i,
  // CIR-SAME: #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i, #cir.int<3> : !u16i]> : !cir.vector<!u16i x 8>
  // CIR: {{%.*}} = cir.shift(left, [[SHIFT_TGT]] : !cir.vector<!u16i x 8>, [[SHIFT_AMT]] : !cir.vector<!u16i x 8>)

  // LLVM: {{.*}}@test_vshll_n_u8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = zext <8 x i8> [[A]] to <8 x i16>
  // LLVM:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], splat (i16 3)
}

uint32x4_t test_vshll_n_u16(uint16x4_t a) {
  return vshll_n_u16(a, 9);

  // CIR-LABEL: vshll_n_u16
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!u16i x 4>), !cir.vector<!u32i x 4>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<9> : !u32i, #cir.int<9> : !u32i,
  // CIR-SAME: #cir.int<9> : !u32i, #cir.int<9> : !u32i]> : !cir.vector<!u32i x 4>

  // LLVM: {{.*}}@test_vshll_n_u16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
  // LLVM:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
  // LLVM:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], splat (i32 9)
  // LLVM:   ret <4 x i32> [[VSHLL_N]]
}

uint64x2_t test_vshll_n_u32(uint32x2_t a) {
  return vshll_n_u32(a, 19);

  // CIR-LABEL: vshll_n_u32
  // CIR: [[SHIFT_TGT:%.*]] = cir.cast(integral, {{%.*}} : !cir.vector<!u32i x 2>), !cir.vector<!u64i x 2>
  // CIR: [[SHIFT_AMT:%.*]] =  cir.const #cir.const_vector<[#cir.int<19> : !u64i, #cir.int<19> : !u64i]> : !cir.vector<!u64i x 2>
  // CIR: {{%.*}} = cir.shift(left, [[SHIFT_TGT]] : !cir.vector<!u64i x 2>, [[SHIFT_AMT]] : !cir.vector<!u64i x 2>)

  // LLVM: {{.*}}@test_vshll_n_u32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
  // LLVM:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
  // LLVM:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], splat (i64 19)
  // LLVM:   ret <2 x i64> [[VSHLL_N]]
}

// NYI-LABEL: @test_vshll_high_n_s8(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// NYI:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
// NYI:   ret <8 x i16> [[VSHLL_N]]
// int16x8_t test_vshll_high_n_s8(int8x16_t a) {
//   return vshll_high_n_s8(a, 3);
// }

// NYI-LABEL: @test_vshll_high_n_s16(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[TMP2:%.*]] = sext <4 x i16> [[TMP1]] to <4 x i32>
// NYI:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 9, i32 9, i32 9, i32 9>
// NYI:   ret <4 x i32> [[VSHLL_N]]
// int32x4_t test_vshll_high_n_s16(int16x8_t a) {
//   return vshll_high_n_s16(a, 9);
// }

// NYI-LABEL: @test_vshll_high_n_s32(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[TMP2:%.*]] = sext <2 x i32> [[TMP1]] to <2 x i64>
// NYI:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 19, i64 19>
// NYI:   ret <2 x i64> [[VSHLL_N]]
// int64x2_t test_vshll_high_n_s32(int32x4_t a) {
//   return vshll_high_n_s32(a, 19);
// }

// NYI-LABEL: @test_vshll_high_n_u8(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I]] to <8 x i16>
// NYI:   [[VSHLL_N:%.*]] = shl <8 x i16> [[TMP0]], <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
// NYI:   ret <8 x i16> [[VSHLL_N]]
// uint16x8_t test_vshll_high_n_u8(uint8x16_t a) {
//   return vshll_high_n_u8(a, 3);
// }

// NYI-LABEL: @test_vshll_high_n_u16(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[TMP2:%.*]] = zext <4 x i16> [[TMP1]] to <4 x i32>
// NYI:   [[VSHLL_N:%.*]] = shl <4 x i32> [[TMP2]], <i32 9, i32 9, i32 9, i32 9>
// NYI:   ret <4 x i32> [[VSHLL_N]]
// uint32x4_t test_vshll_high_n_u16(uint16x8_t a) {
//   return vshll_high_n_u16(a, 9);
// }

// NYI-LABEL: @test_vshll_high_n_u32(
// NYI:   [[SHUFFLE_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[TMP2:%.*]] = zext <2 x i32> [[TMP1]] to <2 x i64>
// NYI:   [[VSHLL_N:%.*]] = shl <2 x i64> [[TMP2]], <i64 19, i64 19>
// NYI:   ret <2 x i64> [[VSHLL_N]]
// uint64x2_t test_vshll_high_n_u32(uint32x4_t a) {
//   return vshll_high_n_u32(a, 19);
// }

int16x8_t test_vmovl_s8(int8x8_t a) {
  return vmovl_s8(a);

  // CIR-LABEL: vmovl_s8
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vmovl_s8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = sext <8 x i8> [[A]] to <8 x i16>
  // LLVM:  ret <8 x i16> [[VMOVL_I]]
}

int32x4_t test_vmovl_s16(int16x4_t a) {
  return vmovl_s16(a);

  // CIR-LABEL: vmovl_s16
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!s16i x 4>), !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vmovl_s16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = sext <4 x i16> [[A]] to <4 x i32>
  // LLVM:  ret <4 x i32> [[VMOVL_I]]
}

int64x2_t test_vmovl_s32(int32x2_t a) {
  return vmovl_s32(a);

  // CIR-LABEL: vmovl_s32
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!s32i x 2>), !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vmovl_s32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = sext <2 x i32> [[A]] to <2 x i64>
  // LLVM:  ret <2 x i64> [[VMOVL_I]]
}

uint16x8_t test_vmovl_u8(uint8x8_t a) {
  return vmovl_u8(a);

  // CIR-LABEL: vmovl_u8
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!u8i x 8>), !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vmovl_u8(<8 x i8>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = zext <8 x i8> [[A]] to <8 x i16>
  // LLVM:  ret <8 x i16> [[VMOVL_I]]
}

uint32x4_t test_vmovl_u16(uint16x4_t a) {
  return vmovl_u16(a);

  // CIR-LABEL: vmovl_u16
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!u16i x 4>), !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vmovl_u16(<4 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = zext <4 x i16> [[A]] to <4 x i32>
  // LLVM:  ret <4 x i32> [[VMOVL_I]]
}

uint64x2_t test_vmovl_u32(uint32x2_t a) {
  return vmovl_u32(a);

  // CIR-LABEL: vmovl_u32
  // CIR: {{%.*}} = cir.cast(integral, {{%.*}} : !cir.vector<!u32i x 2>), !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vmovl_u32(<2 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVL_I:%.*]] = zext <2 x i32> [[A]] to <2 x i64>
  // LLVM:  ret <2 x i64> [[VMOVL_I]]
}

// NYI-LABEL: @test_vmovl_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vmovl_high_s8(int8x16_t a) {
//   return vmovl_high_s8(a);
// }

// NYI-LABEL: @test_vmovl_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[TMP1]]
// int32x4_t test_vmovl_high_s16(int16x8_t a) {
//   return vmovl_high_s16(a);
// }

// NYI-LABEL: @test_vmovl_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[TMP1]]
// int64x2_t test_vmovl_high_s32(int32x4_t a) {
//   return vmovl_high_s32(a);
// }

// NYI-LABEL: @test_vmovl_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vmovl_high_u8(uint8x16_t a) {
//   return vmovl_high_u8(a);
// }

// NYI-LABEL: @test_vmovl_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[TMP1]]
// uint32x4_t test_vmovl_high_u16(uint16x8_t a) {
//   return vmovl_high_u16(a);
// }

// NYI-LABEL: @test_vmovl_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[TMP1]]
// uint64x2_t test_vmovl_high_u32(uint32x4_t a) {
//   return vmovl_high_u32(a);
// }

// NYI-LABEL: @test_vcvt_n_f32_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VCVT_N1:%.*]] = call <2 x float> @llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32(<2 x i32> [[VCVT_N]], i32 31)
// NYI:   ret <2 x float> [[VCVT_N1]]
// float32x2_t test_vcvt_n_f32_s32(int32x2_t a) {
//   return vcvt_n_f32_s32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_f32_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VCVT_N1:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32(<4 x i32> [[VCVT_N]], i32 31)
// NYI:   ret <4 x float> [[VCVT_N1]]
// float32x4_t test_vcvtq_n_f32_s32(int32x4_t a) {
//   return vcvtq_n_f32_s32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_f64_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VCVT_N1:%.*]] = call <2 x double> @llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64(<2 x i64> [[VCVT_N]], i32 50)
// NYI:   ret <2 x double> [[VCVT_N1]]
// float64x2_t test_vcvtq_n_f64_s64(int64x2_t a) {
//   return vcvtq_n_f64_s64(a, 50);
// }

// NYI-LABEL: @test_vcvt_n_f32_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VCVT_N1:%.*]] = call <2 x float> @llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32(<2 x i32> [[VCVT_N]], i32 31)
// NYI:   ret <2 x float> [[VCVT_N1]]
// float32x2_t test_vcvt_n_f32_u32(uint32x2_t a) {
//   return vcvt_n_f32_u32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_f32_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VCVT_N1:%.*]] = call <4 x float> @llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32(<4 x i32> [[VCVT_N]], i32 31)
// NYI:   ret <4 x float> [[VCVT_N1]]
// float32x4_t test_vcvtq_n_f32_u32(uint32x4_t a) {
//   return vcvtq_n_f32_u32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_f64_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VCVT_N1:%.*]] = call <2 x double> @llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64(<2 x i64> [[VCVT_N]], i32 50)
// NYI:   ret <2 x double> [[VCVT_N1]]
// float64x2_t test_vcvtq_n_f64_u64(uint64x2_t a) {
//   return vcvtq_n_f64_u64(a, 50);
// }

// NYI-LABEL: @test_vcvt_n_s32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// NYI:   [[VCVT_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v2i32.v2f32(<2 x float> [[VCVT_N]], i32 31)
// NYI:   ret <2 x i32> [[VCVT_N1]]
// int32x2_t test_vcvt_n_s32_f32(float32x2_t a) {
//   return vcvt_n_s32_f32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_s32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// NYI:   [[VCVT_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxs.v4i32.v4f32(<4 x float> [[VCVT_N]], i32 31)
// NYI:   ret <4 x i32> [[VCVT_N1]]
// int32x4_t test_vcvtq_n_s32_f32(float32x4_t a) {
//   return vcvtq_n_s32_f32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// NYI:   [[VCVT_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64(<2 x double> [[VCVT_N]], i32 50)
// NYI:   ret <2 x i64> [[VCVT_N1]]
// int64x2_t test_vcvtq_n_s64_f64(float64x2_t a) {
//   return vcvtq_n_s64_f64(a, 50);
// }

// NYI-LABEL: @test_vcvt_n_u32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x float>
// NYI:   [[VCVT_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v2i32.v2f32(<2 x float> [[VCVT_N]], i32 31)
// NYI:   ret <2 x i32> [[VCVT_N1]]
// uint32x2_t test_vcvt_n_u32_f32(float32x2_t a) {
//   return vcvt_n_u32_f32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_u32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x float>
// NYI:   [[VCVT_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.vcvtfp2fxu.v4i32.v4f32(<4 x float> [[VCVT_N]], i32 31)
// NYI:   ret <4 x i32> [[VCVT_N1]]
// uint32x4_t test_vcvtq_n_u32_f32(float32x4_t a) {
//   return vcvtq_n_u32_f32(a, 31);
// }

// NYI-LABEL: @test_vcvtq_n_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x double>
// NYI:   [[VCVT_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64(<2 x double> [[VCVT_N]], i32 50)
// NYI:   ret <2 x i64> [[VCVT_N1]]
// uint64x2_t test_vcvtq_n_u64_f64(float64x2_t a) {
//   return vcvtq_n_u64_f64(a, 50);
// }

// NYI-LABEL: @test_vaddl_s8(
// NYI:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %a to <8 x i16>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vaddl_s8(int8x8_t a, int8x8_t b) {
//   return vaddl_s8(a, b);
// }

// NYI-LABEL: @test_vaddl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %a to <4 x i32>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vaddl_s16(int16x4_t a, int16x4_t b) {
//   return vaddl_s16(a, b);
// }

// NYI-LABEL: @test_vaddl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %a to <2 x i64>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vaddl_s32(int32x2_t a, int32x2_t b) {
//   return vaddl_s32(a, b);
// }

// NYI-LABEL: @test_vaddl_u8(
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %a to <8 x i16>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vaddl_u8(uint8x8_t a, uint8x8_t b) {
//   return vaddl_u8(a, b);
// }

// NYI-LABEL: @test_vaddl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %a to <4 x i32>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vaddl_u16(uint16x4_t a, uint16x4_t b) {
//   return vaddl_u16(a, b);
// }

// NYI-LABEL: @test_vaddl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %a to <2 x i64>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vaddl_u32(uint32x2_t a, uint32x2_t b) {
//   return vaddl_u32(a, b);
// }

// NYI-LABEL: @test_vaddl_high_s8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vaddl_high_s8(int8x16_t a, int8x16_t b) {
//   return vaddl_high_s8(a, b);
// }

// NYI-LABEL: @test_vaddl_high_s16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = sext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> [[TMP1]], [[TMP3]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vaddl_high_s16(int16x8_t a, int16x8_t b) {
//   return vaddl_high_s16(a, b);
// }

// NYI-LABEL: @test_vaddl_high_s32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = sext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> [[TMP1]], [[TMP3]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vaddl_high_s32(int32x4_t a, int32x4_t b) {
//   return vaddl_high_s32(a, b);
// }

// NYI-LABEL: @test_vaddl_high_u8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> [[TMP0]], [[TMP1]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vaddl_high_u8(uint8x16_t a, uint8x16_t b) {
//   return vaddl_high_u8(a, b);
// }

// NYI-LABEL: @test_vaddl_high_u16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = zext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> [[TMP1]], [[TMP3]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vaddl_high_u16(uint16x8_t a, uint16x8_t b) {
//   return vaddl_high_u16(a, b);
// }

// NYI-LABEL: @test_vaddl_high_u32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = zext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> [[TMP1]], [[TMP3]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vaddl_high_u32(uint32x4_t a, uint32x4_t b) {
//   return vaddl_high_u32(a, b);
// }

// NYI-LABEL: @test_vaddw_s8(
// NYI:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vaddw_s8(int16x8_t a, int8x8_t b) {
//   return vaddw_s8(a, b);
// }

// NYI-LABEL: @test_vaddw_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vaddw_s16(int32x4_t a, int16x4_t b) {
//   return vaddw_s16(a, b);
// }

// NYI-LABEL: @test_vaddw_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vaddw_s32(int64x2_t a, int32x2_t b) {
//   return vaddw_s32(a, b);
// }

// NYI-LABEL: @test_vaddw_u8(
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vaddw_u8(uint16x8_t a, uint8x8_t b) {
//   return vaddw_u8(a, b);
// }

// NYI-LABEL: @test_vaddw_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vaddw_u16(uint32x4_t a, uint16x4_t b) {
//   return vaddw_u16(a, b);
// }

// NYI-LABEL: @test_vaddw_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vaddw_u32(uint64x2_t a, uint32x2_t b) {
//   return vaddw_u32(a, b);
// }

// NYI-LABEL: @test_vaddw_high_s8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[TMP0]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vaddw_high_s8(int16x8_t a, int8x16_t b) {
//   return vaddw_high_s8(a, b);
// }

// NYI-LABEL: @test_vaddw_high_s16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[TMP1]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vaddw_high_s16(int32x4_t a, int16x8_t b) {
//   return vaddw_high_s16(a, b);
// }

// NYI-LABEL: @test_vaddw_high_s32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[TMP1]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vaddw_high_s32(int64x2_t a, int32x4_t b) {
//   return vaddw_high_s32(a, b);
// }

// NYI-LABEL: @test_vaddw_high_u8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[TMP0]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vaddw_high_u8(uint16x8_t a, uint8x16_t b) {
//   return vaddw_high_u8(a, b);
// }

// NYI-LABEL: @test_vaddw_high_u16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[TMP1]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vaddw_high_u16(uint32x4_t a, uint16x8_t b) {
//   return vaddw_high_u16(a, b);
// }

// NYI-LABEL: @test_vaddw_high_u32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[TMP1]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vaddw_high_u32(uint64x2_t a, uint32x4_t b) {
//   return vaddw_high_u32(a, b);
// }

// NYI-LABEL: @test_vsubl_s8(
// NYI:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %a to <8 x i16>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vsubl_s8(int8x8_t a, int8x8_t b) {
//   return vsubl_s8(a, b);
// }

// NYI-LABEL: @test_vsubl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %a to <4 x i32>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vsubl_s16(int16x4_t a, int16x4_t b) {
//   return vsubl_s16(a, b);
// }

// NYI-LABEL: @test_vsubl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %a to <2 x i64>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vsubl_s32(int32x2_t a, int32x2_t b) {
//   return vsubl_s32(a, b);
// }

// NYI-LABEL: @test_vsubl_u8(
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %a to <8 x i16>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vsubl_u8(uint8x8_t a, uint8x8_t b) {
//   return vsubl_u8(a, b);
// }

// NYI-LABEL: @test_vsubl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %a to <4 x i32>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vsubl_u16(uint16x4_t a, uint16x4_t b) {
//   return vsubl_u16(a, b);
// }

// NYI-LABEL: @test_vsubl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %a to <2 x i64>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I4_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> [[VMOVL_I_I]], [[VMOVL_I4_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vsubl_u32(uint32x2_t a, uint32x2_t b) {
//   return vsubl_u32(a, b);
// }

// NYI-LABEL: @test_vsubl_high_s8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP1:%.*]] = sext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> [[TMP0]], [[TMP1]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vsubl_high_s8(int8x16_t a, int8x16_t b) {
//   return vsubl_high_s8(a, b);
// }

// NYI-LABEL: @test_vsubl_high_s16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = sext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> [[TMP1]], [[TMP3]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vsubl_high_s16(int16x8_t a, int16x8_t b) {
//   return vsubl_high_s16(a, b);
// }

// NYI-LABEL: @test_vsubl_high_s32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = sext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> [[TMP1]], [[TMP3]]
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vsubl_high_s32(int32x4_t a, int32x4_t b) {
//   return vsubl_high_s32(a, b);
// }

// NYI-LABEL: @test_vsubl_high_u8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP1:%.*]] = zext <8 x i8> [[SHUFFLE_I_I10_I]] to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> [[TMP0]], [[TMP1]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vsubl_high_u8(uint8x16_t a, uint8x16_t b) {
//   return vsubl_high_u8(a, b);
// }

// NYI-LABEL: @test_vsubl_high_u16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = zext <4 x i16> [[SHUFFLE_I_I10_I]] to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> [[TMP1]], [[TMP3]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vsubl_high_u16(uint16x8_t a, uint16x8_t b) {
//   return vsubl_high_u16(a, b);
// }

// NYI-LABEL: @test_vsubl_high_u32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SHUFFLE_I_I10_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I10_I]] to <8 x i8>
// NYI:   [[TMP3:%.*]] = zext <2 x i32> [[SHUFFLE_I_I10_I]] to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> [[TMP1]], [[TMP3]]
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vsubl_high_u32(uint32x4_t a, uint32x4_t b) {
//   return vsubl_high_u32(a, b);
// }

// NYI-LABEL: @test_vsubw_s8(
// NYI:   [[VMOVL_I_I:%.*]] = sext <8 x i8> %b to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMOVL_I_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vsubw_s8(int16x8_t a, int8x8_t b) {
//   return vsubw_s8(a, b);
// }

// NYI-LABEL: @test_vsubw_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <4 x i16> %b to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMOVL_I_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vsubw_s16(int32x4_t a, int16x4_t b) {
//   return vsubw_s16(a, b);
// }

// NYI-LABEL: @test_vsubw_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = sext <2 x i32> %b to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMOVL_I_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vsubw_s32(int64x2_t a, int32x2_t b) {
//   return vsubw_s32(a, b);
// }

// NYI-LABEL: @test_vsubw_u8(
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> %b to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMOVL_I_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vsubw_u8(uint16x8_t a, uint8x8_t b) {
//   return vsubw_u8(a, b);
// }

// NYI-LABEL: @test_vsubw_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> %b to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMOVL_I_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vsubw_u16(uint32x4_t a, uint16x4_t b) {
//   return vsubw_u16(a, b);
// }

// NYI-LABEL: @test_vsubw_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> %b to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMOVL_I_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vsubw_u32(uint64x2_t a, uint32x2_t b) {
//   return vsubw_u32(a, b);
// }

// NYI-LABEL: @test_vsubw_high_s8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = sext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[TMP0]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vsubw_high_s8(int16x8_t a, int8x16_t b) {
//   return vsubw_high_s8(a, b);
// }

// NYI-LABEL: @test_vsubw_high_s16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[TMP1]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vsubw_high_s16(int32x4_t a, int16x8_t b) {
//   return vsubw_high_s16(a, b);
// }

// NYI-LABEL: @test_vsubw_high_s32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = sext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[TMP1]]
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vsubw_high_s32(int64x2_t a, int32x4_t b) {
//   return vsubw_high_s32(a, b);
// }

// NYI-LABEL: @test_vsubw_high_u8(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[TMP0:%.*]] = zext <8 x i8> [[SHUFFLE_I_I_I]] to <8 x i16>
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[TMP0]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vsubw_high_u8(uint16x8_t a, uint8x16_t b) {
//   return vsubw_high_u8(a, b);
// }

// NYI-LABEL: @test_vsubw_high_u16(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <4 x i16> [[SHUFFLE_I_I_I]] to <4 x i32>
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[TMP1]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vsubw_high_u16(uint32x4_t a, uint16x8_t b) {
//   return vsubw_high_u16(a, b);
// }

// NYI-LABEL: @test_vsubw_high_u32(
// NYI:   [[SHUFFLE_I_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = zext <2 x i32> [[SHUFFLE_I_I_I]] to <2 x i64>
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[TMP1]]
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vsubw_high_u32(uint64x2_t a, uint32x4_t b) {
//   return vsubw_high_u32(a, b);
// }

// NYI-LABEL: @test_vaddhn_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <8 x i16> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <8 x i16> [[VADDHN_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VADDHN2_I:%.*]] = trunc <8 x i16> [[VADDHN1_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[VADDHN2_I]]
// int8x8_t test_vaddhn_s16(int16x8_t a, int16x8_t b) {
//   return vaddhn_s16(a, b);
// }

// NYI-LABEL: @test_vaddhn_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <4 x i32> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <4 x i32> [[VADDHN_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VADDHN2_I:%.*]] = trunc <4 x i32> [[VADDHN1_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[VADDHN2_I]]
// int16x4_t test_vaddhn_s32(int32x4_t a, int32x4_t b) {
//   return vaddhn_s32(a, b);
// }

// NYI-LABEL: @test_vaddhn_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <2 x i64> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <2 x i64> [[VADDHN_I]], <i64 32, i64 32>
// NYI:   [[VADDHN2_I:%.*]] = trunc <2 x i64> [[VADDHN1_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[VADDHN2_I]]
// int32x2_t test_vaddhn_s64(int64x2_t a, int64x2_t b) {
//   return vaddhn_s64(a, b);
// }

// NYI-LABEL: @test_vaddhn_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <8 x i16> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <8 x i16> [[VADDHN_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VADDHN2_I:%.*]] = trunc <8 x i16> [[VADDHN1_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[VADDHN2_I]]
// uint8x8_t test_vaddhn_u16(uint16x8_t a, uint16x8_t b) {
//   return vaddhn_u16(a, b);
// }

// NYI-LABEL: @test_vaddhn_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <4 x i32> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <4 x i32> [[VADDHN_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VADDHN2_I:%.*]] = trunc <4 x i32> [[VADDHN1_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[VADDHN2_I]]
// uint16x4_t test_vaddhn_u32(uint32x4_t a, uint32x4_t b) {
//   return vaddhn_u32(a, b);
// }

// NYI-LABEL: @test_vaddhn_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VADDHN_I:%.*]] = add <2 x i64> %a, %b
// NYI:   [[VADDHN1_I:%.*]] = lshr <2 x i64> [[VADDHN_I]], <i64 32, i64 32>
// NYI:   [[VADDHN2_I:%.*]] = trunc <2 x i64> [[VADDHN1_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[VADDHN2_I]]
// uint32x2_t test_vaddhn_u64(uint64x2_t a, uint64x2_t b) {
//   return vaddhn_u64(a, b);
// }

// NYI-LABEL: @test_vaddhn_high_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <8 x i16> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <8 x i16> [[VADDHN_I_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <8 x i16> [[VADDHN1_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VADDHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// int8x16_t test_vaddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
//   return vaddhn_high_s16(r, a, b);
// }

// NYI-LABEL: @test_vaddhn_high_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <4 x i32> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <4 x i32> [[VADDHN_I_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <4 x i32> [[VADDHN1_I_I]] to <4 x i16>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VADDHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// int16x8_t test_vaddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
//   return vaddhn_high_s32(r, a, b);
// }

// NYI-LABEL: @test_vaddhn_high_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <2 x i64> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <2 x i64> [[VADDHN_I_I]], <i64 32, i64 32>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <2 x i64> [[VADDHN1_I_I]] to <2 x i32>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VADDHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// int32x4_t test_vaddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
//   return vaddhn_high_s64(r, a, b);
// }

// NYI-LABEL: @test_vaddhn_high_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <8 x i16> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <8 x i16> [[VADDHN_I_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <8 x i16> [[VADDHN1_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VADDHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// uint8x16_t test_vaddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
//   return vaddhn_high_u16(r, a, b);
// }

// NYI-LABEL: @test_vaddhn_high_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <4 x i32> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <4 x i32> [[VADDHN_I_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <4 x i32> [[VADDHN1_I_I]] to <4 x i16>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VADDHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// uint16x8_t test_vaddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
//   return vaddhn_high_u32(r, a, b);
// }

// NYI-LABEL: @test_vaddhn_high_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VADDHN_I_I:%.*]] = add <2 x i64> %a, %b
// NYI:   [[VADDHN1_I_I:%.*]] = lshr <2 x i64> [[VADDHN_I_I]], <i64 32, i64 32>
// NYI:   [[VADDHN2_I_I:%.*]] = trunc <2 x i64> [[VADDHN1_I_I]] to <2 x i32>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VADDHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// uint32x4_t test_vaddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
//   return vaddhn_high_u64(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i8> [[VRADDHN_V2_I]]
// int8x8_t test_vraddhn_s16(int16x8_t a, int16x8_t b) {
//   return vraddhn_s16(a, b);
// }

// NYI-LABEL: @test_vraddhn_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRADDHN_V3_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VRADDHN_V2_I]]
// int16x4_t test_vraddhn_s32(int32x4_t a, int32x4_t b) {
//   return vraddhn_s32(a, b);
// }

// NYI-LABEL: @test_vraddhn_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRADDHN_V3_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VRADDHN_V2_I]]
// int32x2_t test_vraddhn_s64(int64x2_t a, int64x2_t b) {
//   return vraddhn_s64(a, b);
// }

// NYI-LABEL: @test_vraddhn_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i8> [[VRADDHN_V2_I]]
// uint8x8_t test_vraddhn_u16(uint16x8_t a, uint16x8_t b) {
//   return vraddhn_u16(a, b);
// }

// NYI-LABEL: @test_vraddhn_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRADDHN_V3_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VRADDHN_V2_I]]
// uint16x4_t test_vraddhn_u32(uint32x4_t a, uint32x4_t b) {
//   return vraddhn_u32(a, b);
// }

// NYI-LABEL: @test_vraddhn_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRADDHN_V3_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VRADDHN_V2_I]]
// uint32x2_t test_vraddhn_u64(uint64x2_t a, uint64x2_t b) {
//   return vraddhn_u64(a, b);
// }

// NYI-LABEL: @test_vraddhn_high_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRADDHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// int8x16_t test_vraddhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
//   return vraddhn_high_s16(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_high_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRADDHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRADDHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// int16x8_t test_vraddhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
//   return vraddhn_high_s32(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_high_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRADDHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRADDHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// int32x4_t test_vraddhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
//   return vraddhn_high_s64(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_high_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.raddhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRADDHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// uint8x16_t test_vraddhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
//   return vraddhn_high_u16(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_high_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.raddhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRADDHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRADDHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRADDHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// uint16x8_t test_vraddhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
//   return vraddhn_high_u32(r, a, b);
// }

// NYI-LABEL: @test_vraddhn_high_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRADDHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.raddhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRADDHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRADDHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRADDHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// uint32x4_t test_vraddhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
//   return vraddhn_high_u64(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <8 x i16> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <8 x i16> [[VSUBHN_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[VSUBHN2_I]]
// int8x8_t test_vsubhn_s16(int16x8_t a, int16x8_t b) {
//   return vsubhn_s16(a, b);
// }

// NYI-LABEL: @test_vsubhn_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <4 x i32> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <4 x i32> [[VSUBHN_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[VSUBHN2_I]]
// int16x4_t test_vsubhn_s32(int32x4_t a, int32x4_t b) {
//   return vsubhn_s32(a, b);
// }

// NYI-LABEL: @test_vsubhn_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <2 x i64> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <2 x i64> [[VSUBHN_I]], <i64 32, i64 32>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[VSUBHN2_I]]
// int32x2_t test_vsubhn_s64(int64x2_t a, int64x2_t b) {
//   return vsubhn_s64(a, b);
// }

// NYI-LABEL: @test_vsubhn_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <8 x i16> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <8 x i16> [[VSUBHN_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I]] to <8 x i8>
// NYI:   ret <8 x i8> [[VSUBHN2_I]]
// uint8x8_t test_vsubhn_u16(uint16x8_t a, uint16x8_t b) {
//   return vsubhn_u16(a, b);
// }

// NYI-LABEL: @test_vsubhn_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <4 x i32> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <4 x i32> [[VSUBHN_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I]] to <4 x i16>
// NYI:   ret <4 x i16> [[VSUBHN2_I]]
// uint16x4_t test_vsubhn_u32(uint32x4_t a, uint32x4_t b) {
//   return vsubhn_u32(a, b);
// }

// NYI-LABEL: @test_vsubhn_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSUBHN_I:%.*]] = sub <2 x i64> %a, %b
// NYI:   [[VSUBHN1_I:%.*]] = lshr <2 x i64> [[VSUBHN_I]], <i64 32, i64 32>
// NYI:   [[VSUBHN2_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I]] to <2 x i32>
// NYI:   ret <2 x i32> [[VSUBHN2_I]]
// uint32x2_t test_vsubhn_u64(uint64x2_t a, uint64x2_t b) {
//   return vsubhn_u64(a, b);
// }

// NYI-LABEL: @test_vsubhn_high_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <8 x i16> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <8 x i16> [[VSUBHN_I_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VSUBHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// int8x16_t test_vsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
//   return vsubhn_high_s16(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_high_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <4 x i32> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <4 x i32> [[VSUBHN_I_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I_I]] to <4 x i16>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VSUBHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// int16x8_t test_vsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
//   return vsubhn_high_s32(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_high_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <2 x i64> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <2 x i64> [[VSUBHN_I_I]], <i64 32, i64 32>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I_I]] to <2 x i32>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VSUBHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// int32x4_t test_vsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
//   return vsubhn_high_s64(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_high_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <8 x i16> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <8 x i16> [[VSUBHN_I_I]], <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <8 x i16> [[VSUBHN1_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VSUBHN2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// uint8x16_t test_vsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
//   return vsubhn_high_u16(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_high_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <4 x i32> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <4 x i32> [[VSUBHN_I_I]], <i32 16, i32 16, i32 16, i32 16>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <4 x i32> [[VSUBHN1_I_I]] to <4 x i16>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VSUBHN2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// uint16x8_t test_vsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
//   return vsubhn_high_u32(r, a, b);
// }

// NYI-LABEL: @test_vsubhn_high_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSUBHN_I_I:%.*]] = sub <2 x i64> %a, %b
// NYI:   [[VSUBHN1_I_I:%.*]] = lshr <2 x i64> [[VSUBHN_I_I]], <i64 32, i64 32>
// NYI:   [[VSUBHN2_I_I:%.*]] = trunc <2 x i64> [[VSUBHN1_I_I]] to <2 x i32>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VSUBHN2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// uint32x4_t test_vsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
//   return vsubhn_high_u64(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i8> [[VRSUBHN_V2_I]]
// int8x8_t test_vrsubhn_s16(int16x8_t a, int16x8_t b) {
//   return vrsubhn_s16(a, b);
// }

// NYI-LABEL: @test_vrsubhn_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRSUBHN_V3_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VRSUBHN_V2_I]]
// int16x4_t test_vrsubhn_s32(int32x4_t a, int32x4_t b) {
//   return vrsubhn_s32(a, b);
// }

// NYI-LABEL: @test_vrsubhn_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRSUBHN_V3_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VRSUBHN_V2_I]]
// int32x2_t test_vrsubhn_s64(int64x2_t a, int64x2_t b) {
//   return vrsubhn_s64(a, b);
// }

// NYI-LABEL: @test_vrsubhn_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i8> [[VRSUBHN_V2_I]]
// uint8x8_t test_vrsubhn_u16(uint16x8_t a, uint16x8_t b) {
//   return vrsubhn_u16(a, b);
// }

// NYI-LABEL: @test_vrsubhn_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRSUBHN_V3_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I]] to <8 x i8>
// NYI:   ret <4 x i16> [[VRSUBHN_V2_I]]
// uint16x4_t test_vrsubhn_u32(uint32x4_t a, uint32x4_t b) {
//   return vrsubhn_u32(a, b);
// }

// NYI-LABEL: @test_vrsubhn_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRSUBHN_V3_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I]] to <8 x i8>
// NYI:   ret <2 x i32> [[VRSUBHN_V2_I]]
// uint32x2_t test_vrsubhn_u64(uint64x2_t a, uint64x2_t b) {
//   return vrsubhn_u64(a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRSUBHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// int8x16_t test_vrsubhn_high_s16(int8x8_t r, int16x8_t a, int16x8_t b) {
//   return vrsubhn_high_s16(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRSUBHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// int16x8_t test_vrsubhn_high_s32(int16x4_t r, int32x4_t a, int32x4_t b) {
//   return vrsubhn_high_s32(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRSUBHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// int32x4_t test_vrsubhn_high_s64(int32x2_t r, int64x2_t a, int64x2_t b) {
//   return vrsubhn_high_s64(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.rsubhn.v8i8(<8 x i16> %a, <8 x i16> %b)
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i8> %r, <8 x i8> [[VRSUBHN_V2_I_I]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   ret <16 x i8> [[SHUFFLE_I_I]]
// uint8x16_t test_vrsubhn_high_u16(uint8x8_t r, uint16x8_t a, uint16x8_t b) {
//   return vrsubhn_high_u16(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.rsubhn.v4i16(<4 x i32> %a, <4 x i32> %b)
// NYI:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <4 x i16> [[VRSUBHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i16> %r, <4 x i16> [[VRSUBHN_V2_I_I]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
// NYI:   ret <8 x i16> [[SHUFFLE_I_I]]
// uint16x8_t test_vrsubhn_high_u32(uint16x4_t r, uint32x4_t a, uint32x4_t b) {
//   return vrsubhn_high_u32(r, a, b);
// }

// NYI-LABEL: @test_vrsubhn_high_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VRSUBHN_V2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.rsubhn.v2i32(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VRSUBHN_V3_I_I:%.*]] = bitcast <2 x i32> [[VRSUBHN_V2_I_I]] to <8 x i8>
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i32> %r, <2 x i32> [[VRSUBHN_V2_I_I]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
// NYI:   ret <4 x i32> [[SHUFFLE_I_I]]
// uint32x4_t test_vrsubhn_high_u64(uint32x2_t r, uint64x2_t a, uint64x2_t b) {
//   return vrsubhn_high_u64(r, a, b);
// }

// NYI-LABEL: @test_vabdl_s8(
// NYI:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> [[VABD_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[VMOVL_I_I]]
// int16x8_t test_vabdl_s8(int8x8_t a, int8x8_t b) {
//   return vabdl_s8(a, b);
// }

// NYI-LABEL: @test_vabdl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[VMOVL_I_I]]
// int32x4_t test_vabdl_s16(int16x4_t a, int16x4_t b) {
//   return vabdl_s16(a, b);
// }

// NYI-LABEL: @test_vabdl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[VMOVL_I_I]]
// int64x2_t test_vabdl_s32(int32x2_t a, int32x2_t b) {
//   return vabdl_s32(a, b);
// }

// NYI-LABEL: @test_vabdl_u8(
// NYI:   [[VABD_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   [[VMOVL_I_I:%.*]] = zext <8 x i8> [[VABD_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[VMOVL_I_I]]
// uint16x8_t test_vabdl_u8(uint8x8_t a, uint8x8_t b) {
//   return vabdl_u8(a, b);
// }

// NYI-LABEL: @test_vabdl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[VMOVL_I_I]]
// uint32x4_t test_vabdl_u16(uint16x4_t a, uint16x4_t b) {
//   return vabdl_u16(a, b);
// }

// NYI-LABEL: @test_vabdl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VABD2_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[VMOVL_I_I]]
// uint64x2_t test_vabdl_u32(uint32x2_t a, uint32x2_t b) {
//   return vabdl_u32(a, b);
// }

// NYI-LABEL: @test_vabal_s8(
// NYI:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vabal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
//   return vabal_s8(a, b, c);
// }

// NYI-LABEL: @test_vabal_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vabal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
//   return vabal_s16(a, b, c);
// }

// NYI-LABEL: @test_vabal_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vabal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
//   return vabal_s32(a, b, c);
// }

// NYI-LABEL: @test_vabal_u8(
// NYI:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vabal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
//   return vabal_u8(a, b, c);
// }

// NYI-LABEL: @test_vabal_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vabal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
//   return vabal_u16(a, b, c);
// }

// NYI-LABEL: @test_vabal_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vabal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
//   return vabal_u32(a, b, c);
// }

// NYI-LABEL: @test_vabdl_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[VMOVL_I_I_I]]
// int16x8_t test_vabdl_high_s8(int8x16_t a, int8x16_t b) {
//   return vabdl_high_s8(a, b);
// }

// NYI-LABEL: @test_vabdl_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[VMOVL_I_I_I]]
// int32x4_t test_vabdl_high_s16(int16x8_t a, int16x8_t b) {
//   return vabdl_high_s16(a, b);
// }

// NYI-LABEL: @test_vabdl_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[VMOVL_I_I_I]]
// int64x2_t test_vabdl_high_s32(int32x4_t a, int32x4_t b) {
//   return vabdl_high_s32(a, b);
// }

// NYI-LABEL: @test_vabdl_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VABD_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I]] to <8 x i16>
// NYI:   ret <8 x i16> [[VMOVL_I_I_I]]
// uint16x8_t test_vabdl_high_u8(uint8x16_t a, uint8x16_t b) {
//   return vabdl_high_u8(a, b);
// }

// NYI-LABEL: @test_vabdl_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I]] to <4 x i32>
// NYI:   ret <4 x i32> [[VMOVL_I_I_I]]
// uint32x4_t test_vabdl_high_u16(uint16x8_t a, uint16x8_t b) {
//   return vabdl_high_u16(a, b);
// }

// NYI-LABEL: @test_vabdl_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I]] to <2 x i64>
// NYI:   ret <2 x i64> [[VMOVL_I_I_I]]
// uint64x2_t test_vabdl_high_u32(uint32x4_t a, uint32x4_t b) {
//   return vabdl_high_u32(a, b);
// }

// NYI-LABEL: @test_vabal_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VABD_I_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I_I]]
// int16x8_t test_vabal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
//   return vabal_high_s8(a, b, c);
// }

// NYI-LABEL: @test_vabal_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I_I]]
// int32x4_t test_vabal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
//   return vabal_high_s16(a, b, c);
// }

// NYI-LABEL: @test_vabal_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.sabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I_I]]
// int64x2_t test_vabal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
//   return vabal_high_s32(a, b, c);
// }

// NYI-LABEL: @test_vabal_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VABD_I_I_I_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uabd.v8i8(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <8 x i8> [[VABD_I_I_I_I]] to <8 x i16>
// NYI:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I_I]]
// uint16x8_t test_vabal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
//   return vabal_high_u8(a, b, c);
// }

// NYI-LABEL: @test_vabal_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uabd.v4i16(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[VABD2_I_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <4 x i16> [[VABD2_I_I_I_I]] to <4 x i32>
// NYI:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I_I]]
// uint32x4_t test_vabal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
//   return vabal_high_u16(a, b, c);
// }

// NYI-LABEL: @test_vabal_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VABD2_I_I_I_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.uabd.v2i32(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[VABD2_I_I_I_I]] to <8 x i8>
// NYI:   [[VMOVL_I_I_I_I:%.*]] = zext <2 x i32> [[VABD2_I_I_I_I]] to <2 x i64>
// NYI:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMOVL_I_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I_I]]
// uint64x2_t test_vabal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
//   return vabal_high_u32(a, b, c);
// }

int16x8_t test_vmull_s8(int8x8_t a, int8x8_t b) {
  return vmull_s8(a, b);

  // CIR-LABEL: vmull_s8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vmull_s8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
  // LLVM: ret <8 x i16> [[VMULL_I]]
}

int32x4_t test_vmull_s16(int16x4_t a, int16x4_t b) {
  return vmull_s16(a, b);

  // CIR-LABEL: vmull_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vmull_s16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM: [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM: ret <4 x i32> [[VMULL2_I]]
}

int64x2_t test_vmull_s32(int32x2_t a, int32x2_t b) {
  return vmull_s32(a, b);

  // CIR-LABEL: vmull_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.smull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vmull_s32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM: [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM: ret <2 x i64> [[VMULL2_I]]
}

uint16x8_t test_vmull_u8(uint8x8_t a, uint8x8_t b) {
  return vmull_u8(a, b);

  // CIR-LABEL: vmull_u8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u8i x 8>, !cir.vector<!u8i x 8>) -> !cir.vector<!u16i x 8>

  // LLVM: {{.*}}test_vmull_u8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
  // LLVM: ret <8 x i16> [[VMULL_I]]
}

uint32x4_t test_vmull_u16(uint16x4_t a, uint16x4_t b) {
  return vmull_u16(a, b);

  // CIR-LABEL: vmull_u16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u16i x 4>, !cir.vector<!u16i x 4>) -> !cir.vector<!u32i x 4>

  // LLVM: {{.*}}test_vmull_u16(<4 x i16>{{.*}}[[A:%.*]], <4 x i16>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <4 x i16> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <4 x i16> [[B]] to <8 x i8>
  // LLVM: [[VMULL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[A]], <4 x i16> [[B]])
  // LLVM: ret <4 x i32> [[VMULL2_I]]
}

uint64x2_t test_vmull_u32(uint32x2_t a, uint32x2_t b) {
  return vmull_u32(a, b);

  // CIR-LABEL: vmull_u32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.umull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!u32i x 2>, !cir.vector<!u32i x 2>) -> !cir.vector<!u64i x 2>

  // LLVM: {{.*}}test_vmull_u32(<2 x i32>{{.*}}[[A:%.*]], <2 x i32>{{.*}}[[B:%.*]])
  // LLVM: [[TMP0:%.*]] = bitcast <2 x i32> [[A]] to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <2 x i32> [[B]] to <8 x i8>
  // LLVM: [[VMULL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[A]], <2 x i32> [[B]])
  // LLVM: ret <2 x i64> [[VMULL2_I]]
}

// NYI-LABEL: @test_vmull_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   ret <8 x i16> [[VMULL_I_I]]
// int16x8_t test_vmull_high_s8(int8x16_t a, int8x16_t b) {
//   return vmull_high_s8(a, b);
// }

// NYI-LABEL: @test_vmull_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   ret <4 x i32> [[VMULL2_I_I]]
// int32x4_t test_vmull_high_s16(int16x8_t a, int16x8_t b) {
//   return vmull_high_s16(a, b);
// }

// NYI-LABEL: @test_vmull_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   ret <2 x i64> [[VMULL2_I_I]]
// int64x2_t test_vmull_high_s32(int32x4_t a, int32x4_t b) {
//   return vmull_high_s32(a, b);
// }

// NYI-LABEL: @test_vmull_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   ret <8 x i16> [[VMULL_I_I]]
// uint16x8_t test_vmull_high_u8(uint8x16_t a, uint8x16_t b) {
//   return vmull_high_u8(a, b);
// }

// NYI-LABEL: @test_vmull_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   ret <4 x i32> [[VMULL2_I_I]]
// uint32x4_t test_vmull_high_u16(uint16x8_t a, uint16x8_t b) {
//   return vmull_high_u16(a, b);
// }

// NYI-LABEL: @test_vmull_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   ret <2 x i64> [[VMULL2_I_I]]
// uint64x2_t test_vmull_high_u32(uint32x4_t a, uint32x4_t b) {
//   return vmull_high_u32(a, b);
// }

// NYI-LABEL: @test_vmlal_s8(
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// int16x8_t test_vmlal_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
//   return vmlal_s8(a, b, c);
// }

// NYI-LABEL: @test_vmlal_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// int32x4_t test_vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
//   return vmlal_s16(a, b, c);
// }

// NYI-LABEL: @test_vmlal_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// int64x2_t test_vmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
//   return vmlal_s32(a, b, c);
// }

// NYI-LABEL: @test_vmlal_u8(
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[ADD_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I]]
// NYI:   ret <8 x i16> [[ADD_I]]
// uint16x8_t test_vmlal_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
//   return vmlal_u8(a, b, c);
// }

// NYI-LABEL: @test_vmlal_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[ADD_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I]]
// NYI:   ret <4 x i32> [[ADD_I]]
// uint32x4_t test_vmlal_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
//   return vmlal_u16(a, b, c);
// }

// NYI-LABEL: @test_vmlal_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[ADD_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I]]
// NYI:   ret <2 x i64> [[ADD_I]]
// uint64x2_t test_vmlal_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
//   return vmlal_u32(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I_I]]
// int16x8_t test_vmlal_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
//   return vmlal_high_s8(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I_I]]
// int32x4_t test_vmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
//   return vmlal_high_s16(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I_I]]
// int64x2_t test_vmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
//   return vmlal_high_s32(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <8 x i16> %a, [[VMULL_I_I_I]]
// NYI:   ret <8 x i16> [[ADD_I_I]]
// uint16x8_t test_vmlal_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
//   return vmlal_high_u8(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <4 x i32> %a, [[VMULL2_I_I_I]]
// NYI:   ret <4 x i32> [[ADD_I_I]]
// uint32x4_t test_vmlal_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
//   return vmlal_high_u16(a, b, c);
// }

// NYI-LABEL: @test_vmlal_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[ADD_I_I:%.*]] = add <2 x i64> %a, [[VMULL2_I_I_I]]
// NYI:   ret <2 x i64> [[ADD_I_I]]
// uint64x2_t test_vmlal_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
//   return vmlal_high_u32(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_s8(
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// int16x8_t test_vmlsl_s8(int16x8_t a, int8x8_t b, int8x8_t c) {
//   return vmlsl_s8(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// int32x4_t test_vmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
//   return vmlsl_s16(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// int64x2_t test_vmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
//   return vmlsl_s32(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_u8(
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> %b, <8 x i8> %c)
// NYI:   [[SUB_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I]]
// NYI:   ret <8 x i16> [[SUB_I]]
// uint16x8_t test_vmlsl_u8(uint16x8_t a, uint8x8_t b, uint8x8_t c) {
//   return vmlsl_u8(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> %b, <4 x i16> %c)
// NYI:   [[SUB_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I]]
// NYI:   ret <4 x i32> [[SUB_I]]
// uint32x4_t test_vmlsl_u16(uint32x4_t a, uint16x4_t b, uint16x4_t c) {
//   return vmlsl_u16(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %c to <8 x i8>
// NYI:   [[VMULL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> %b, <2 x i32> %c)
// NYI:   [[SUB_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I]]
// NYI:   ret <2 x i64> [[SUB_I]]
// uint64x2_t test_vmlsl_u32(uint64x2_t a, uint32x2_t b, uint32x2_t c) {
//   return vmlsl_u32(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_s8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.smull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I_I]]
// NYI:   ret <8 x i16> [[SUB_I_I]]
// int16x8_t test_vmlsl_high_s8(int16x8_t a, int8x16_t b, int8x16_t c) {
//   return vmlsl_high_s8(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.smull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// NYI:   ret <4 x i32> [[SUB_I_I]]
// int32x4_t test_vmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
//   return vmlsl_high_s16(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.smull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// NYI:   ret <2 x i64> [[SUB_I_I]]
// int64x2_t test_vmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
//   return vmlsl_high_s32(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_u8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %c, <16 x i8> %c, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.umull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <8 x i16> %a, [[VMULL_I_I_I]]
// NYI:   ret <8 x i16> [[SUB_I_I]]
// uint16x8_t test_vmlsl_high_u8(uint16x8_t a, uint8x16_t b, uint8x16_t c) {
//   return vmlsl_high_u8(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_u16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.umull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <4 x i32> %a, [[VMULL2_I_I_I]]
// NYI:   ret <4 x i32> [[SUB_I_I]]
// uint32x4_t test_vmlsl_high_u16(uint32x4_t a, uint16x8_t b, uint16x8_t c) {
//   return vmlsl_high_u16(a, b, c);
// }

// NYI-LABEL: @test_vmlsl_high_u32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VMULL2_I_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.umull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[SUB_I_I:%.*]] = sub <2 x i64> %a, [[VMULL2_I_I_I]]
// NYI:   ret <2 x i64> [[SUB_I_I]]
// uint64x2_t test_vmlsl_high_u32(uint64x2_t a, uint32x4_t b, uint32x4_t c) {
//   return vmlsl_high_u32(a, b, c);
// }

// NYI-LABEL: @test_vqdmull_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VQDMULL_V2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> %a, <4 x i16> %b)
// NYI:   [[VQDMULL_V3_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQDMULL_V2_I]]
// int32x4_t test_vqdmull_s16(int16x4_t a, int16x4_t b) {
//   return vqdmull_s16(a, b);
// }

// NYI-LABEL: @test_vqdmull_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VQDMULL_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> %a, <2 x i32> %b)
// NYI:   [[VQDMULL_V3_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQDMULL_V2_I]]
// int64x2_t test_vqdmull_s32(int32x2_t a, int32x2_t b) {
//   return vqdmull_s32(a, b);
// }

int32x4_t test_vqdmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlal_s16(a, b, c);

  // CIR-LABEL: vqdmlal_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vqdmlal_s16(<4 x i32>{{.*}}[[a:%.*]], <4 x i16>{{.*}}[[b:%.*]], <4 x i16>{{.*}}[[c:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <4 x i16> [[c]] to <8 x i8>
  // LLVM:   [[VQDMLAL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[b]], <4 x i16> [[c]])
  // LLVM:   [[VQDMLAL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> [[a]], <4 x i32> [[VQDMLAL2_I]])
  // LLVM:   ret <4 x i32> [[VQDMLAL_V3_I]]
}

int64x2_t test_vqdmlal_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlal_s32(a, b, c);

  // CIR-LABEL: vqdmlal_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vqdmlal_s32(<2 x i64>{{.*}}[[a:%.*]], <2 x i32>{{.*}}[[b:%.*]], <2 x i32>{{.*}}[[c:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <2 x i32> [[c]] to <8 x i8>
  // LLVM:   [[VQDMLAL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[b]], <2 x i32> [[c]])
  // LLVM:   [[VQDMLAL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> [[a]], <2 x i64> [[VQDMLAL2_I]])
  // LLVM:   ret <2 x i64> [[VQDMLAL_V3_I]]
}


int32x4_t test_vqdmlsl_s16(int32x4_t a, int16x4_t b, int16x4_t c) {
  return vqdmlsl_s16(a, b, c);

  // CIR-LABEL: vqdmlsl_s16
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s16i x 4>, !cir.vector<!s16i x 4>) -> !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 4>, !cir.vector<!s32i x 4>) -> !cir.vector<!s32i x 4>

  // LLVM: {{.*}}test_vqdmlsl_s16(<4 x i32>{{.*}}[[a:%.*]], <4 x i16>{{.*}}[[b:%.*]], <4 x i16>{{.*}}[[c:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <4 x i32> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <4 x i16> [[b]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <4 x i16> [[c]] to <8 x i8>
  // LLVM:   [[VQDMLSL2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[b]], <4 x i16> [[c]])
  // LLVM:   [[VQDMLSL_V3_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> [[a]], <4 x i32> [[VQDMLSL2_I]])
  // LLVM:   ret <4 x i32> [[VQDMLSL_V3_I]]
}

int64x2_t test_vqdmlsl_s32(int64x2_t a, int32x2_t b, int32x2_t c) {
  return vqdmlsl_s32(a, b, c);

  // CIR-LABEL: vqdmlsl_s32
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqdmull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s32i x 2>, !cir.vector<!s32i x 2>) -> !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.sqsub" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s64i x 2>, !cir.vector<!s64i x 2>) -> !cir.vector<!s64i x 2>

  // LLVM: {{.*}}test_vqdmlsl_s32(<2 x i64>{{.*}}[[a:%.*]], <2 x i32>{{.*}}[[b:%.*]], <2 x i32>{{.*}}[[c:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <2 x i64> [[a]] to <16 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <2 x i32> [[b]] to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <2 x i32> [[c]] to <8 x i8>
  // LLVM:   [[VQDMLSL2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[b]], <2 x i32> [[c]])
  // LLVM:   [[VQDMLSL_V3_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> [[a]], <2 x i64> [[VQDMLSL2_I]])
  // LLVM:   ret <2 x i64> [[VQDMLSL_V3_I]]
}

// NYI-LABEL: @test_vqdmull_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %a, <8 x i16> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMULL_V2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMULL_V3_I_I:%.*]] = bitcast <4 x i32> [[VQDMULL_V2_I_I]] to <16 x i8>
// NYI:   ret <4 x i32> [[VQDMULL_V2_I_I]]
// int32x4_t test_vqdmull_high_s16(int16x8_t a, int16x8_t b) {
//   return vqdmull_high_s16(a, b);
// }

// NYI-LABEL: @test_vqdmull_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %a, <4 x i32> %a, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMULL_V2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMULL_V3_I_I:%.*]] = bitcast <2 x i64> [[VQDMULL_V2_I_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VQDMULL_V2_I_I]]
// int64x2_t test_vqdmull_high_s32(int32x4_t a, int32x4_t b) {
//   return vqdmull_high_s32(a, b);
// }

// NYI-LABEL: @test_vqdmlal_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMLAL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMLAL_V3_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqadd.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I_I]])
// NYI:   ret <4 x i32> [[VQDMLAL_V3_I_I]]
// int32x4_t test_vqdmlal_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
//   return vqdmlal_high_s16(a, b, c);
// }

// NYI-LABEL: @test_vqdmlal_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMLAL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMLAL_V3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqadd.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I_I]])
// NYI:   ret <2 x i64> [[VQDMLAL_V3_I_I]]
// int64x2_t test_vqdmlal_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
//   return vqdmlal_high_s32(a, b, c);
// }

// NYI-LABEL: @test_vqdmlsl_high_s16(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <8 x i16> %b, <8 x i16> %b, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <8 x i16> %c, <8 x i16> %c, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <4 x i16> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMLAL2_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[SHUFFLE_I_I]], <4 x i16> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMLSL_V3_I_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqsub.v4i32(<4 x i32> %a, <4 x i32> [[VQDMLAL2_I_I]])
// NYI:   ret <4 x i32> [[VQDMLSL_V3_I_I]]
// int32x4_t test_vqdmlsl_high_s16(int32x4_t a, int16x8_t b, int16x8_t c) {
//   return vqdmlsl_high_s16(a, b, c);
// }

// NYI-LABEL: @test_vqdmlsl_high_s32(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <4 x i32> %b, <4 x i32> %b, <2 x i32> <i32 2, i32 3>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <4 x i32> %c, <4 x i32> %c, <2 x i32> <i32 2, i32 3>
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> [[SHUFFLE_I_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <2 x i32> [[SHUFFLE_I7_I]] to <8 x i8>
// NYI:   [[VQDMLAL2_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqdmull.v2i64(<2 x i32> [[SHUFFLE_I_I]], <2 x i32> [[SHUFFLE_I7_I]])
// NYI:   [[VQDMLSL_V3_I_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqsub.v2i64(<2 x i64> %a, <2 x i64> [[VQDMLAL2_I_I]])
// NYI:   ret <2 x i64> [[VQDMLSL_V3_I_I]]
// int64x2_t test_vqdmlsl_high_s32(int64x2_t a, int32x4_t b, int32x4_t c) {
//   return vqdmlsl_high_s32(a, b, c);
// }

poly16x8_t test_vmull_p8(poly8x8_t a, poly8x8_t b) {
  return vmull_p8(a, b);

  // CIR-LABEL: vmull_p8
  // CIR: {{%.*}} = cir.llvm.intrinsic "aarch64.neon.pmull" {{%.*}}, {{%.*}} :
  // CIR-SAME: (!cir.vector<!s8i x 8>, !cir.vector<!s8i x 8>) -> !cir.vector<!s16i x 8>

  // LLVM: {{.*}}test_vmull_p8(<8 x i8>{{.*}}[[A:%.*]], <8 x i8>{{.*}}[[B:%.*]])
  // LLVM: [[VMULL_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> [[A]], <8 x i8> [[B]])
  // LLVM: ret <8 x i16> [[VMULL_I]]
}

// NYI-LABEL: @test_vmull_high_p8(
// NYI:   [[SHUFFLE_I_I:%.*]] = shufflevector <16 x i8> %a, <16 x i8> %a, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[SHUFFLE_I7_I:%.*]] = shufflevector <16 x i8> %b, <16 x i8> %b, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// NYI:   [[VMULL_I_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.pmull.v8i16(<8 x i8> [[SHUFFLE_I_I]], <8 x i8> [[SHUFFLE_I7_I]])
// NYI:   ret <8 x i16> [[VMULL_I_I]]
// poly16x8_t test_vmull_high_p8(poly8x16_t a, poly8x16_t b) {
//   return vmull_high_p8(a, b);
// }

int64_t test_vaddd_s64(int64_t a, int64_t b) {
  return vaddd_s64(a, b);

  // CIR-LABEL: vaddd_s64
  // CIR: {{%.*}} = cir.binop(add, {{%.*}}, {{%.*}}) : !s64i

  // LLVM-LABEL: @test_vaddd_s64
  // LLVM-SAME: (i64 [[a:%.]], i64 [[b:%.]])
  // LLVM:   [[VADDD_I:%.*]]  = add i64 [[a]], [[b]]
  // LLVM:   ret i64 [[VADDD_I]]
}

uint64_t test_vaddd_u64(uint64_t a, uint64_t b) {
   return vaddd_u64(a, b);

  // CIR-LABEL: vaddd_u64
  // CIR: {{%.*}} = cir.binop(add, {{%.*}}, {{%.*}}) : !u64i

  // LLVM-LABEL: @test_vaddd_u64
  // LLVM-SAME: (i64 [[a:%.]], i64 [[b:%.]])
  // LLVM:   [[VADDD_I:%.*]]  = add i64 [[a]], [[b]]
  // LLVM:   ret i64 [[VADDD_I]]
}

int64_t test_vsubd_s64(int64_t a, int64_t b) {
  return vsubd_s64(a, b);

  // CIR-LABEL: vsubd_s64
  // CIR: {{%.*}} = cir.binop(sub, {{%.*}}, {{%.*}}) : !s64i

  // LLVM-LABEL: @test_vsubd_s64
  // LLVM-SAME: (i64 [[a:%.]], i64 [[b:%.]])
  // LLVM:   [[VSUBD_I:%.*]]  = sub i64 [[a]], [[b]]
  // LLVM:   ret i64 [[VSUBD_I]]
}

uint64_t test_vsubd_u64(uint64_t a, uint64_t b) {
  return vsubd_u64(a, b);

  // CIR-LABEL: vsubd_u64
  // CIR: {{%.*}} = cir.binop(sub, {{%.*}}, {{%.*}}) : !u64i

  // LLVM-LABEL: @test_vsubd_u64
  // LLVM-SAME: (i64 [[a:%.]], i64 [[b:%.]])
  // LLVM:   [[VSUBD_I:%.*]]  = sub i64 [[a]], [[b]]
  // LLVM:   ret i64 [[VSUBD_I]]
}

// NYI-LABEL: @test_vqaddb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQADDB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQADDB_S8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// int8_t test_vqaddb_s8(int8_t a, int8_t b) {
//   return vqaddb_s8(a, b);
// }

// NYI-LABEL: @test_vqaddh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQADDH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQADDH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqaddh_s16(int16_t a, int16_t b) {
//   return vqaddh_s16(a, b);
// }

int32_t test_vqadds_s32(int32_t a, int32_t b) {
  return vqadds_s32(a, b);

  // CIR: vqadds_s32
  // CIR: cir.binop(add, {{%.*}}, {{%.*}}) sat : !s32i

  // LLVM:{{.*}}test_vqadds_s32(i32{{.*}}[[a:%.*]], i32{{.*}}[[b:%.*]])
  // LLVM:   [[VQADDS_S32_I:%.*]] = call i32 @llvm.sadd.sat.i32(i32 [[a]], i32 [[b]])
  // LLVM:   ret i32 [[VQADDS_S32_I]]
}

int64_t test_vqaddd_s64(int64_t a, int64_t b) {
  return vqaddd_s64(a, b);

  // CIR: vqaddd_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqadd" {{%.*}}, {{%.*}} : (!s64i, !s64i) -> !s64i

  // LLVM-LABEL: @test_vqaddd_s64
  // LLVM-SAME: (i64{{.*}}[[a:%.*]], i64{{.*}}[[b:%.*]])
  // LLVM:   [[VQADD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 [[a]], i64 [[b]])
  // LLVM:   ret i64 [[VQADD_S64_I]]
}

uint64_t test_vqaddd_u64(uint64_t a, uint64_t b) {
  return vqaddd_u64(a, b);

  // CIR: vqaddd_u64
  // CIR: cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} : (!u64i, !u64i) -> !u64i

  // LLVM-LABEL: @test_vqaddd_u64
  // LLVM-SAME: (i64{{.*}}[[a:%.*]], i64{{.*}}[[b:%.*]])
  // LLVM:   [[VQADD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqadd.i64(i64 [[a]], i64 [[b]])
  // LLVM:   ret i64 [[VQADD_U64_I]]
}

// NYI-LABEL: @test_vqaddb_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQADDB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQADDB_U8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// uint8_t test_vqaddb_u8(uint8_t a, uint8_t b) {
//   return vqaddb_u8(a, b);
// }

// NYI-LABEL: @test_vqaddh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQADDH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQADDH_U16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// uint16_t test_vqaddh_u16(uint16_t a, uint16_t b) {
//   return vqaddh_u16(a, b);
// }

uint32_t test_vqadds_u32(uint32_t a, uint32_t b) {
  return vqadds_u32(a, b);

  // CIR: vqadds_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.uqadd" {{%.*}}, {{%.*}} : (!u32i, !u32i) -> !u32i

  // LLVM-LABEL: @test_vqadds_u32
  // LLVM-SAME: (i32{{.*}}[[a:%.*]], i32{{.*}}[[b:%.*]])
  // LLVM:   [[VQADDS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqadd.i32(i32 [[a]], i32 [[b]])
  // LLVM:   ret i32 [[VQADDS_U32_I]]
}


// NYI-LABEL: @test_vqsubb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQSUBB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqsub.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSUBB_S8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// int8_t test_vqsubb_s8(int8_t a, int8_t b) {
//   return vqsubb_s8(a, b);
// }

// NYI-LABEL: @test_vqsubh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQSUBH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqsub.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSUBH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqsubh_s16(int16_t a, int16_t b) {
//   return vqsubh_s16(a, b);
// }

// NYI-LABEL: @test_vqsubs_s32(
// NYI:   [[VQSUBS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQSUBS_S32_I]]
int32_t test_vqsubs_s32(int32_t a, int32_t b) {
  return vqsubs_s32(a, b);

  // CIR: vqsubs_s32
  // CIR: cir.binop(sub, {{%.*}}, {{%.*}}) sat : !s32i

  // LLVM:{{.*}}test_vqsubs_s32(i32{{.*}}[[a:%.*]], i32{{.*}}[[b:%.*]])
  // LLVM:   [[VQSUBS_S32_I:%.*]] = call i32 @llvm.ssub.sat.i32(i32 [[a]], i32 [[b]])
  // LLVM:   ret i32 [[VQSUBS_S32_I]]
}

int64_t test_vqsubd_s64(int64_t a, int64_t b) {
  return vqsubd_s64(a, b);

  // CIR: vqsubd_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqsub" {{%.*}}, {{%.*}} : (!s64i, !s64i) -> !s64i

  // LLVM-LABEL: @test_vqsubd_s64
  // LLVM-SAME: (i64{{.*}}[[a:%.*]], i64{{.*}}[[b:%.*]])
  // LLVM:   [[VQSUBD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 [[a]], i64 [[b]])
  // LLVM:   ret i64 [[VQSUBD_S64_I]]
}

// NYI-LABEL: @test_vqsubb_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQSUBB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqsub.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSUBB_U8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// uint8_t test_vqsubb_u8(uint8_t a, uint8_t b) {
//   return vqsubb_u8(a, b);
// }

// NYI-LABEL: @test_vqsubh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQSUBH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqsub.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSUBH_U16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// uint16_t test_vqsubh_u16(uint16_t a, uint16_t b) {
//   return vqsubh_u16(a, b);
// }

// NYI-LABEL: @test_vqsubs_u32(
// NYI:   [[VQSUBS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqsub.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQSUBS_U32_I]]
// uint32_t test_vqsubs_u32(uint32_t a, uint32_t b) {
//   return vqsubs_u32(a, b);
// }

// NYI-LABEL: @test_vqsubd_u64(
// NYI:   [[VQSUBD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqsub.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VQSUBD_U64_I]]
// uint64_t test_vqsubd_u64(uint64_t a, uint64_t b) {
//   return vqsubd_u64(a, b);
// }

// NYI-LABEL: @test_vshld_s64(
// NYI:   [[VSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VSHLD_S64_I]]
// int64_t test_vshld_s64(int64_t a, int64_t b) {
//   return vshld_s64(a, b);
// }

// NYI-LABEL: @test_vshld_u64(
// NYI:   [[VSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.ushl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VSHLD_U64_I]]
// uint64_t test_vshld_u64(uint64_t a, int64_t b) {
//   return vshld_u64(a, b);
// }

// NYI-LABEL: @test_vqshlb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQSHLB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSHLB_S8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// int8_t test_vqshlb_s8(int8_t a, int8_t b) {
//   return vqshlb_s8(a, b);
// }

// NYI-LABEL: @test_vqshlh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQSHLH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSHLH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqshlh_s16(int16_t a, int16_t b) {
//   return vqshlh_s16(a, b);
// }

// NYI-LABEL: @test_vqshls_s32(
// NYI:   [[VQSHLS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqshl.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQSHLS_S32_I]]
// int32_t test_vqshls_s32(int32_t a, int32_t b) {
//   return vqshls_s32(a, b);
// }

// NYI-LABEL: @test_vqshld_s64(
// NYI:   [[VQSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VQSHLD_S64_I]]
// int64_t test_vqshld_s64(int64_t a, int64_t b) {
//   return vqshld_s64(a, b);
// }

// NYI-LABEL: @test_vqshlb_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQSHLB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQSHLB_U8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// uint8_t test_vqshlb_u8(uint8_t a, int8_t b) {
//   return vqshlb_u8(a, b);
// }

// NYI-LABEL: @test_vqshlh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQSHLH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQSHLH_U16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// uint16_t test_vqshlh_u16(uint16_t a, int16_t b) {
//   return vqshlh_u16(a, b);
// }

// NYI-LABEL: @test_vqshls_u32(
// NYI:   [[VQSHLS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqshl.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQSHLS_U32_I]]
// uint32_t test_vqshls_u32(uint32_t a, int32_t b) {
//   return vqshls_u32(a, b);
// }

// NYI-LABEL: @test_vqshld_u64(
// NYI:   [[VQSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VQSHLD_U64_I]]
// uint64_t test_vqshld_u64(uint64_t a, int64_t b) {
//   return vqshld_u64(a, b);
// }

// NYI-LABEL: @test_vrshld_s64(
// NYI:   [[VRSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VRSHLD_S64_I]]
// int64_t test_vrshld_s64(int64_t a, int64_t b) {
//   return vrshld_s64(a, b);
// }

// NYI-LABEL: @test_vrshld_u64(
// NYI:   [[VRSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VRSHLD_U64_I]]
// uint64_t test_vrshld_u64(uint64_t a, int64_t b) {
//   return vrshld_u64(a, b);
// }

// NYI-LABEL: @test_vqrshlb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQRSHLB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQRSHLB_S8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// int8_t test_vqrshlb_s8(int8_t a, int8_t b) {
//   return vqrshlb_s8(a, b);
// }

// NYI-LABEL: @test_vqrshlh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQRSHLH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRSHLH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqrshlh_s16(int16_t a, int16_t b) {
//   return vqrshlh_s16(a, b);
// }

// NYI-LABEL: @test_vqrshls_s32(
// NYI:   [[VQRSHLS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrshl.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQRSHLS_S32_I]]
// int32_t test_vqrshls_s32(int32_t a, int32_t b) {
//   return vqrshls_s32(a, b);
// }

// NYI-LABEL: @test_vqrshld_s64(
// NYI:   [[VQRSHLD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqrshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VQRSHLD_S64_I]]
// int64_t test_vqrshld_s64(int64_t a, int64_t b) {
//   return vqrshld_s64(a, b);
// }

// NYI-LABEL: @test_vqrshlb_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VQRSHLB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VQRSHLB_U8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// uint8_t test_vqrshlb_u8(uint8_t a, int8_t b) {
//   return vqrshlb_u8(a, b);
// }

// NYI-LABEL: @test_vqrshlh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQRSHLH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRSHLH_U16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// uint16_t test_vqrshlh_u16(uint16_t a, int16_t b) {
//   return vqrshlh_u16(a, b);
// }

// NYI-LABEL: @test_vqrshls_u32(
// NYI:   [[VQRSHLS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uqrshl.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQRSHLS_U32_I]]
// uint32_t test_vqrshls_u32(uint32_t a, int32_t b) {
//   return vqrshls_u32(a, b);
// }

// NYI-LABEL: @test_vqrshld_u64(
// NYI:   [[VQRSHLD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uqrshl.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VQRSHLD_U64_I]]
// uint64_t test_vqrshld_u64(uint64_t a, int64_t b) {
//   return vqrshld_u64(a, b);
// }

// NYI-LABEL: @test_vpaddd_s64(
// NYI:   [[VPADDD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// NYI:   ret i64 [[VPADDD_S64_I]]
// int64_t test_vpaddd_s64(int64x2_t a) {
//   return vpaddd_s64(a);
// }

// NYI-LABEL: @test_vpadds_f32(
// NYI:   [[LANE0_I:%.*]] = extractelement <2 x float> %a, i64 0
// NYI:   [[LANE1_I:%.*]] = extractelement <2 x float> %a, i64 1
// NYI:   [[VPADDD_I:%.*]] = fadd float [[LANE0_I]], [[LANE1_I]]
// NYI:   ret float [[VPADDD_I]]
// float32_t test_vpadds_f32(float32x2_t a) {
//   return vpadds_f32(a);
// }

// NYI-LABEL: @test_vpaddd_f64(
// NYI:   [[LANE0_I:%.*]] = extractelement <2 x double> %a, i64 0
// NYI:   [[LANE1_I:%.*]] = extractelement <2 x double> %a, i64 1
// NYI:   [[VPADDD_I:%.*]] = fadd double [[LANE0_I]], [[LANE1_I]]
// NYI:   ret double [[VPADDD_I]]
// float64_t test_vpaddd_f64(float64x2_t a) {
//   return vpaddd_f64(a);
// }

// NYI-LABEL: @test_vpmaxnms_f32(
// NYI:   [[VPMAXNMS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float> %a)
// NYI:   ret float [[VPMAXNMS_F32_I]]
// float32_t test_vpmaxnms_f32(float32x2_t a) {
//   return vpmaxnms_f32(a);
// }

// NYI-LABEL: @test_vpmaxnmqd_f64(
// NYI:   [[VPMAXNMQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double> %a)
// NYI:   ret double [[VPMAXNMQD_F64_I]]
// float64_t test_vpmaxnmqd_f64(float64x2_t a) {
//   return vpmaxnmqd_f64(a);
// }

// NYI-LABEL: @test_vpmaxs_f32(
// NYI:   [[VPMAXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> %a)
// NYI:   ret float [[VPMAXS_F32_I]]
// float32_t test_vpmaxs_f32(float32x2_t a) {
//   return vpmaxs_f32(a);
// }

// NYI-LABEL: @test_vpmaxqd_f64(
// NYI:   [[VPMAXQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> %a)
// NYI:   ret double [[VPMAXQD_F64_I]]
// float64_t test_vpmaxqd_f64(float64x2_t a) {
//   return vpmaxqd_f64(a);
// }

// NYI-LABEL: @test_vpminnms_f32(
// NYI:   [[VPMINNMS_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float> %a)
// NYI:   ret float [[VPMINNMS_F32_I]]
// float32_t test_vpminnms_f32(float32x2_t a) {
//   return vpminnms_f32(a);
// }

// NYI-LABEL: @test_vpminnmqd_f64(
// NYI:   [[VPMINNMQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double> %a)
// NYI:   ret double [[VPMINNMQD_F64_I]]
// float64_t test_vpminnmqd_f64(float64x2_t a) {
//   return vpminnmqd_f64(a);
// }

// NYI-LABEL: @test_vpmins_f32(
// NYI:   [[VPMINS_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float> %a)
// NYI:   ret float [[VPMINS_F32_I]]
// float32_t test_vpmins_f32(float32x2_t a) {
//   return vpmins_f32(a);
// }

// NYI-LABEL: @test_vpminqd_f64(
// NYI:   [[VPMINQD_F64_I:%.*]] = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> %a)
// NYI:   ret double [[VPMINQD_F64_I]]
// float64_t test_vpminqd_f64(float64x2_t a) {
//   return vpminqd_f64(a);
// }

// NYI-LABEL: @test_vqdmulhh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqdmulh.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQDMULHH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqdmulhh_s16(int16_t a, int16_t b) {
//   return vqdmulhh_s16(a, b);
// }

// NYI-LABEL: @test_vqdmulhs_s32(
// NYI:   [[VQDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqdmulh.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VQDMULHS_S32_I]]
// int32_t test_vqdmulhs_s32(int32_t a, int32_t b) {
//   return vqdmulhs_s32(a, b);
// }

// NYI-LABEL: @test_vqrdmulhh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQRDMULHH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrdmulh.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VQRDMULHH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vqrdmulhh_s16(int16_t a, int16_t b) {
//   return vqrdmulhh_s16(a, b);
// }

int32_t test_vqrdmulhs_s32(int32_t a, int32_t b) {
  return vqrdmulhs_s32(a, b);

  // CIR-LABEL: vqrdmulhs_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.sqrdmulh" {{%.*}}, {{%.*}} : (!s32i, !s32i) -> !s32i

  // LLVM: {{.*}}test_vqrdmulhs_s32(i32{{.*}}[[a:%.*]], i32{{.*}}[[b:%.*]])
  // LLVM:   [[VQRDMULHS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqrdmulh.i32(i32 [[a]], i32 [[b]])
  // LLVM:   ret i32 [[VQRDMULHS_S32_I]]
}

// NYI-LABEL: @test_vmulxs_f32(
// NYI:   [[VMULXS_F32_I:%.*]] = call float @llvm.aarch64.neon.fmulx.f32(float %a, float %b)
// NYI:   ret float [[VMULXS_F32_I]]
// float32_t test_vmulxs_f32(float32_t a, float32_t b) {
//   return vmulxs_f32(a, b);
// }

// NYI-LABEL: @test_vmulxd_f64(
// NYI:   [[VMULXD_F64_I:%.*]] = call double @llvm.aarch64.neon.fmulx.f64(double %a, double %b)
// NYI:   ret double [[VMULXD_F64_I]]
// float64_t test_vmulxd_f64(float64_t a, float64_t b) {
//   return vmulxd_f64(a, b);
// }

// NYI-LABEL: @test_vmulx_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VMULX2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmulx.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x double> [[VMULX2_I]]
// float64x1_t test_vmulx_f64(float64x1_t a, float64x1_t b) {
//   return vmulx_f64(a, b);
// }

// NYI-LABEL: @test_vrecpss_f32(
// NYI:   [[VRECPS_I:%.*]] = call float @llvm.aarch64.neon.frecps.f32(float %a, float %b)
// NYI:   ret float [[VRECPS_I]]
// float32_t test_vrecpss_f32(float32_t a, float32_t b) {
//   return vrecpss_f32(a, b);
// }

// NYI-LABEL: @test_vrecpsd_f64(
// NYI:   [[VRECPS_I:%.*]] = call double @llvm.aarch64.neon.frecps.f64(double %a, double %b)
// NYI:   ret double [[VRECPS_I]]
// float64_t test_vrecpsd_f64(float64_t a, float64_t b) {
//   return vrecpsd_f64(a, b);
// }

// NYI-LABEL: @test_vrsqrtss_f32(
// NYI:   [[VRSQRTSS_F32_I:%.*]] = call float @llvm.aarch64.neon.frsqrts.f32(float %a, float %b)
// NYI:   ret float [[VRSQRTSS_F32_I]]
// float32_t test_vrsqrtss_f32(float32_t a, float32_t b) {
//   return vrsqrtss_f32(a, b);
// }

// NYI-LABEL: @test_vrsqrtsd_f64(
// NYI:   [[VRSQRTSD_F64_I:%.*]] = call double @llvm.aarch64.neon.frsqrts.f64(double %a, double %b)
// NYI:   ret double [[VRSQRTSD_F64_I]]
// float64_t test_vrsqrtsd_f64(float64_t a, float64_t b) {
//   return vrsqrtsd_f64(a, b);
// }

// NYI-LABEL: @test_vcvts_f32_s32(
// NYI:   [[TMP0:%.*]] = sitofp i32 %a to float
// NYI:   ret float [[TMP0]]
// float32_t test_vcvts_f32_s32(int32_t a) {
//   return vcvts_f32_s32(a);
// }

// NYI-LABEL: @test_vcvtd_f64_s64(
// NYI:   [[TMP0:%.*]] = sitofp i64 %a to double
// NYI:   ret double [[TMP0]]
// float64_t test_vcvtd_f64_s64(int64_t a) {
//   return vcvtd_f64_s64(a);
// }

// NYI-LABEL: @test_vcvts_f32_u32(
// NYI:   [[TMP0:%.*]] = uitofp i32 %a to float
// NYI:   ret float [[TMP0]]
// float32_t test_vcvts_f32_u32(uint32_t a) {
//   return vcvts_f32_u32(a);
// }

// NYI-LABEL: @test_vcvtd_f64_u64(
// NYI:   [[TMP0:%.*]] = uitofp i64 %a to double
// NYI:   ret double [[TMP0]]
// float64_t test_vcvtd_f64_u64(uint64_t a) {
//   return vcvtd_f64_u64(a);
// }

// NYI-LABEL: @test_vrecpes_f32(
// NYI:   [[VRECPES_F32_I:%.*]] = call float @llvm.aarch64.neon.frecpe.f32(float %a)
// NYI:   ret float [[VRECPES_F32_I]]
// float32_t test_vrecpes_f32(float32_t a) {
//   return vrecpes_f32(a);
// }

// NYI-LABEL: @test_vrecped_f64(
// NYI:   [[VRECPED_F64_I:%.*]] = call double @llvm.aarch64.neon.frecpe.f64(double %a)
// NYI:   ret double [[VRECPED_F64_I]]
// float64_t test_vrecped_f64(float64_t a) {
//   return vrecped_f64(a);
// }

// NYI-LABEL: @test_vrecpxs_f32(
// NYI:   [[VRECPXS_F32_I:%.*]] = call float @llvm.aarch64.neon.frecpx.f32(float %a)
// NYI:   ret float [[VRECPXS_F32_I]]
// float32_t test_vrecpxs_f32(float32_t a) {
//   return vrecpxs_f32(a);
// }

// NYI-LABEL: @test_vrecpxd_f64(
// NYI:   [[VRECPXD_F64_I:%.*]] = call double @llvm.aarch64.neon.frecpx.f64(double %a)
// NYI:   ret double [[VRECPXD_F64_I]]
// float64_t test_vrecpxd_f64(float64_t a) {
//   return vrecpxd_f64(a);
// }

// NYI-LABEL: @test_vrsqrte_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VRSQRTE_V1_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.ursqrte.v2i32(<2 x i32> %a)
// NYI:   ret <2 x i32> [[VRSQRTE_V1_I]]
// uint32x2_t test_vrsqrte_u32(uint32x2_t a) {
//   return vrsqrte_u32(a);
// }

// NYI-LABEL: @test_vrsqrteq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VRSQRTEQ_V1_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.ursqrte.v4i32(<4 x i32> %a)
// NYI:   ret <4 x i32> [[VRSQRTEQ_V1_I]]
// uint32x4_t test_vrsqrteq_u32(uint32x4_t a) {
//   return vrsqrteq_u32(a);
// }

// NYI-LABEL: @test_vrsqrtes_f32(
// NYI:   [[VRSQRTES_F32_I:%.*]] = call float @llvm.aarch64.neon.frsqrte.f32(float %a)
// NYI:   ret float [[VRSQRTES_F32_I]]
// float32_t test_vrsqrtes_f32(float32_t a) {
//   return vrsqrtes_f32(a);
// }

// NYI-LABEL: @test_vrsqrted_f64(
// NYI:   [[VRSQRTED_F64_I:%.*]] = call double @llvm.aarch64.neon.frsqrte.f64(double %a)
// NYI:   ret double [[VRSQRTED_F64_I]]
// float64_t test_vrsqrted_f64(float64_t a) {
//   return vrsqrted_f64(a);
// }

uint8x16_t test_vld1q_u8(uint8_t const *a) {
  return vld1q_u8(a);
  // CIR-LABEL: @test_vld1q_u8
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: cir.load align(1) %[[CAST]] : !cir.ptr<!cir.vector<!u8i x 16>>, !cir.vector<!u8i x 16>

  // LLVM-LABEL: @test_vld1q_u8
  // LLVM:   [[TMP1:%.*]] = load <16 x i8>, ptr %0, align 1
}

uint16x8_t test_vld1q_u16(uint16_t const *a) {
  return vld1q_u16(a);
  // CIR-LABEL: @test_vld1q_u16
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u16i x 8>>
  // CIR: cir.load align(2) %[[CAST]] : !cir.ptr<!cir.vector<!u16i x 8>>, !cir.vector<!u16i x 8>

  // LLVM-LABEL: @test_vld1q_u16
  // LLVM:   [[TMP1:%.*]] = load <8 x i16>, ptr %0, align 2
}

uint32x4_t test_vld1q_u32(uint32_t const *a) {
  return vld1q_u32(a);
  // CIR-LABEL: @test_vld1q_u32
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: cir.load align(4) %[[CAST]] : !cir.ptr<!cir.vector<!u32i x 4>>, !cir.vector<!u32i x 4>

  // LLVM-LABEL: @test_vld1q_u32
  // LLVM:   [[TMP1:%.*]] = load <4 x i32>, ptr %0, align 4
}

uint64x2_t test_vld1q_u64(uint64_t const *a) {
  return vld1q_u64(a);
  // CIR-LABEL: @test_vld1q_u64
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u64i x 2>>
  // CIR: cir.load align(8) %[[CAST]] : !cir.ptr<!cir.vector<!u64i x 2>>, !cir.vector<!u64i x 2>

  // LLVM-LABEL: @test_vld1q_u64
  // LLVM:   [[TMP1:%.*]] = load <2 x i64>, ptr %0, align 8
}

int8x16_t test_vld1q_s8(int8_t const *a) {
  return vld1q_s8(a);
  // CIR-LABEL: @test_vld1q_s8
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s8i x 16>>
  // CIR: cir.load align(1) %[[CAST]] : !cir.ptr<!cir.vector<!s8i x 16>>, !cir.vector<!s8i x 16>

  // LLVM-LABEL: @test_vld1q_s8
  // LLVM:   [[TMP1:%.*]] = load <16 x i8>, ptr %0, align 1
}

int16x8_t test_vld1q_s16(int16_t const *a) {
  return vld1q_s16(a);
  // CIR-LABEL: @test_vld1q_s16
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: cir.load align(2) %[[CAST]] : !cir.ptr<!cir.vector<!s16i x 8>>, !cir.vector<!s16i x 8>

  // LLVM-LABEL: @test_vld1q_s16
  // LLVM:   [[TMP1:%.*]] = load <8 x i16>, ptr %0, align 2
}

int32x4_t test_vld1q_s32(int32_t const *a) {
  return vld1q_s32(a);
  // CIR-LABEL: @test_vld1q_s32
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR: cir.load align(4) %[[CAST]] : !cir.ptr<!cir.vector<!s32i x 4>>, !cir.vector<!s32i x 4>

  // LLVM-LABEL: @test_vld1q_s32
  // LLVM:   [[TMP1:%.*]] = load <4 x i32>, ptr %0, align 4
}

int64x2_t test_vld1q_s64(int64_t const *a) {
  return vld1q_s64(a);
  // CIR-LABEL: @test_vld1q_s64
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s64i x 2>>
  // CIR: cir.load align(8) %[[CAST]] : !cir.ptr<!cir.vector<!s64i x 2>>, !cir.vector<!s64i x 2>

  // LLVM-LABEL: @test_vld1q_s64
  // LLVM:   [[TMP1:%.*]] = load <2 x i64>, ptr %0, align 8
}

// NYI-LABEL: @test_vld1q_f16(
// NYI:   [[TMP2:%.*]] = load <8 x half>, ptr %a, align 2
// NYI:   ret <8 x half> [[TMP2]]
// float16x8_t test_vld1q_f16(float16_t const *a) {
//   return vld1q_f16(a);
// }

// NYI-LABEL: @test_vld1q_f32(
// NYI:   [[TMP2:%.*]] = load <4 x float>, ptr %a, align 4
// NYI:   ret <4 x float> [[TMP2]]
// float32x4_t test_vld1q_f32(float32_t const *a) {
//   return vld1q_f32(a);
// }

// NYI-LABEL: @test_vld1q_f64(
// NYI:   [[TMP2:%.*]] = load <2 x double>, ptr %a, align 8
// NYI:   ret <2 x double> [[TMP2]]
// float64x2_t test_vld1q_f64(float64_t const *a) {
//   return vld1q_f64(a);
// }

// NYI-LABEL: @test_vld1q_p8(
// NYI:   [[TMP1:%.*]] = load <16 x i8>, ptr %a, align 1
// NYI:   ret <16 x i8> [[TMP1]]
// poly8x16_t test_vld1q_p8(poly8_t const *a) {
//   return vld1q_p8(a);
// }

// NYI-LABEL: @test_vld1q_p16(
// NYI:   [[TMP2:%.*]] = load <8 x i16>, ptr %a, align 2
// NYI:   ret <8 x i16> [[TMP2]]
// poly16x8_t test_vld1q_p16(poly16_t const *a) {
//   return vld1q_p16(a);
// }

// NYI-LABEL: @test_vld1_u8(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// uint8x8_t test_vld1_u8(uint8_t const *a) {
//   return vld1_u8(a);
// }

// NYI-LABEL: @test_vld1_u16(
// NYI:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// NYI:   ret <4 x i16> [[TMP2]]
// uint16x4_t test_vld1_u16(uint16_t const *a) {
//   return vld1_u16(a);
// }

// NYI-LABEL: @test_vld1_u32(
// NYI:   [[TMP2:%.*]] = load <2 x i32>, ptr %a, align 4
// NYI:   ret <2 x i32> [[TMP2]]
// uint32x2_t test_vld1_u32(uint32_t const *a) {
//   return vld1_u32(a);
// }

// NYI-LABEL: @test_vld1_u64(
// NYI:   [[TMP2:%.*]] = load <1 x i64>, ptr %a, align 8
// NYI:   ret <1 x i64> [[TMP2]]
// uint64x1_t test_vld1_u64(uint64_t const *a) {
//   return vld1_u64(a);
// }

// NYI-LABEL: @test_vld1_s8(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// int8x8_t test_vld1_s8(int8_t const *a) {
//   return vld1_s8(a);
// }

// NYI-LABEL: @test_vld1_s16(
// NYI:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// NYI:   ret <4 x i16> [[TMP2]]
// int16x4_t test_vld1_s16(int16_t const *a) {
//   return vld1_s16(a);
// }

// NYI-LABEL: @test_vld1_s32(
// NYI:   [[TMP2:%.*]] = load <2 x i32>, ptr %a, align 4
// NYI:   ret <2 x i32> [[TMP2]]
// int32x2_t test_vld1_s32(int32_t const *a) {
//   return vld1_s32(a);
// }

// NYI-LABEL: @test_vld1_s64(
// NYI:   [[TMP2:%.*]] = load <1 x i64>, ptr %a, align 8
// NYI:   ret <1 x i64> [[TMP2]]
// int64x1_t test_vld1_s64(int64_t const *a) {
//   return vld1_s64(a);
// }

// NYI-LABEL: @test_vld1_f16(
// NYI:   [[TMP2:%.*]] = load <4 x half>, ptr %a, align 2
// NYI:   ret <4 x half> [[TMP2]]
// float16x4_t test_vld1_f16(float16_t const *a) {
//   return vld1_f16(a);
// }

// NYI-LABEL: @test_vld1_f32(
// NYI:   [[TMP2:%.*]] = load <2 x float>, ptr %a, align 4
// NYI:   ret <2 x float> [[TMP2]]
// float32x2_t test_vld1_f32(float32_t const *a) {
//   return vld1_f32(a);
// }

// NYI-LABEL: @test_vld1_f64(
// NYI:   [[TMP2:%.*]] = load <1 x double>, ptr %a, align 8
// NYI:   ret <1 x double> [[TMP2]]
// float64x1_t test_vld1_f64(float64_t const *a) {
//   return vld1_f64(a);
// }

// NYI-LABEL: @test_vld1_p8(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// poly8x8_t test_vld1_p8(poly8_t const *a) {
//   return vld1_p8(a);
// }

// NYI-LABEL: @test_vld1_p16(
// NYI:   [[TMP2:%.*]] = load <4 x i16>, ptr %a, align 2
// NYI:   ret <4 x i16> [[TMP2]]
// poly16x4_t test_vld1_p16(poly16_t const *a) {
//   return vld1_p16(a);
// }

// NYI-LABEL: @test_vld1_u8_void(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// uint8x8_t test_vld1_u8_void(void *a) {
//   return vld1_u8(a);
// }

// NYI-LABEL: @test_vld1_u16_void(
// NYI:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// NYI:   ret <4 x i16> [[TMP1]]
// uint16x4_t test_vld1_u16_void(void *a) {
//   return vld1_u16(a);
// }

// NYI-LABEL: @test_vld1_u32_void(
// NYI:   [[TMP1:%.*]] = load <2 x i32>, ptr %a, align 1
// NYI:   ret <2 x i32> [[TMP1]]
// uint32x2_t test_vld1_u32_void(void *a) {
//   return vld1_u32(a);
// }

// NYI-LABEL: @test_vld1_u64_void(
// NYI:   [[TMP1:%.*]] = load <1 x i64>, ptr %a, align 1
// NYI:   ret <1 x i64> [[TMP1]]
// uint64x1_t test_vld1_u64_void(void *a) {
//   return vld1_u64(a);
// }

// NYI-LABEL: @test_vld1_s8_void(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// int8x8_t test_vld1_s8_void(void *a) {
//   return vld1_s8(a);
// }

// NYI-LABEL: @test_vld1_s16_void(
// NYI:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// NYI:   ret <4 x i16> [[TMP1]]
// int16x4_t test_vld1_s16_void(void *a) {
//   return vld1_s16(a);
// }

// NYI-LABEL: @test_vld1_s32_void(
// NYI:   [[TMP1:%.*]] = load <2 x i32>, ptr %a, align 1
// NYI:   ret <2 x i32> [[TMP1]]
// int32x2_t test_vld1_s32_void(void *a) {
//   return vld1_s32(a);
// }

// NYI-LABEL: @test_vld1_s64_void(
// NYI:   [[TMP1:%.*]] = load <1 x i64>, ptr %a, align 1
// NYI:   ret <1 x i64> [[TMP1]]
// int64x1_t test_vld1_s64_void(void *a) {
//   return vld1_s64(a);
// }

// NYI-LABEL: @test_vld1_f16_void(
// NYI:   [[TMP1:%.*]] = load <4 x half>, ptr %a, align 1
// NYI:   ret <4 x half> [[TMP1]]
// float16x4_t test_vld1_f16_void(void *a) {
//   return vld1_f16(a);
// }

// NYI-LABEL: @test_vld1_f32_void(
// NYI:   [[TMP1:%.*]] = load <2 x float>, ptr %a, align 1
// NYI:   ret <2 x float> [[TMP1]]
// float32x2_t test_vld1_f32_void(void *a) {
//   return vld1_f32(a);
// }

// NYI-LABEL: @test_vld1_f64_void(
// NYI:   [[TMP1:%.*]] = load <1 x double>, ptr %a, align 1
// NYI:   ret <1 x double> [[TMP1]]
// float64x1_t test_vld1_f64_void(void *a) {
//   return vld1_f64(a);
// }

// NYI-LABEL: @test_vld1_p8_void(
// NYI:   [[TMP1:%.*]] = load <8 x i8>, ptr %a, align 1
// NYI:   ret <8 x i8> [[TMP1]]
// poly8x8_t test_vld1_p8_void(void *a) {
//   return vld1_p8(a);
// }

// NYI-LABEL: @test_vld1_p16_void(
// NYI:   [[TMP1:%.*]] = load <4 x i16>, ptr %a, align 1
// NYI:   ret <4 x i16> [[TMP1]]
// poly16x4_t test_vld1_p16_void(void *a) {
//   return vld1_p16(a);
// }

// NYI-LABEL: @test_vld2q_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x16x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x16x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x16x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint8x16x2_t [[TMP5]]
// uint8x16x2_t test_vld2q_u8(uint8_t const *a) {
//   return vld2q_u8(a);
// }

// NYI-LABEL: @test_vld2q_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x8x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x8x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x8x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint16x8x2_t [[TMP6]]
// uint16x8x2_t test_vld2q_u16(uint16_t const *a) {
//   return vld2q_u16(a);
// }

// NYI-LABEL: @test_vld2q_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x4x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x4x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x4x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint32x4x2_t [[TMP6]]
// uint32x4x2_t test_vld2q_u32(uint32_t const *a) {
//   return vld2q_u32(a);
// }

// NYI-LABEL: @test_vld2q_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x2x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x2x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x2x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint64x2x2_t [[TMP6]]
// uint64x2x2_t test_vld2q_u64(uint64_t const *a) {
//   return vld2q_u64(a);
// }

// NYI-LABEL: @test_vld2q_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x16x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int8x16x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x16x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int8x16x2_t [[TMP5]]
// int8x16x2_t test_vld2q_s8(int8_t const *a) {
//   return vld2q_s8(a);
// }

// NYI-LABEL: @test_vld2q_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x8x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int16x8x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x8x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int16x8x2_t [[TMP6]]
// int16x8x2_t test_vld2q_s16(int16_t const *a) {
//   return vld2q_s16(a);
// }

// NYI-LABEL: @test_vld2q_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x4x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int32x4x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld2.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x4x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int32x4x2_t [[TMP6]]
// int32x4x2_t test_vld2q_s32(int32_t const *a) {
//   return vld2q_s32(a);
// }

// NYI-LABEL: @test_vld2q_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x2x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int64x2x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x2x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int64x2x2_t [[TMP6]]
// int64x2x2_t test_vld2q_s64(int64_t const *a) {
//   return vld2q_s64(a);
// }

// NYI-LABEL: @test_vld2q_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x8x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float16x8x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <8 x half>, <8 x half> } @llvm.aarch64.neon.ld2.v8f16.p0(ptr %a)
// NYI:   store { <8 x half>, <8 x half> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x8x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float16x8x2_t [[TMP6]]
// float16x8x2_t test_vld2q_f16(float16_t const *a) {
//   return vld2q_f16(a);
// }

// NYI-LABEL: @test_vld2q_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x4x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float32x4x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <4 x float>, <4 x float> } @llvm.aarch64.neon.ld2.v4f32.p0(ptr %a)
// NYI:   store { <4 x float>, <4 x float> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x4x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float32x4x2_t [[TMP6]]
// float32x4x2_t test_vld2q_f32(float32_t const *a) {
//   return vld2q_f32(a);
// }

// NYI-LABEL: @test_vld2q_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld2.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x2_t [[TMP6]]
// float64x2x2_t test_vld2q_f64(float64_t const *a) {
//   return vld2q_f64(a);
// }

// NYI-LABEL: @test_vld2q_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x16x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x16x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld2.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x16x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly8x16x2_t [[TMP5]]
// poly8x16x2_t test_vld2q_p8(poly8_t const *a) {
//   return vld2q_p8(a);
// }

// NYI-LABEL: @test_vld2q_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x8x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x8x2_t, align 16
// NYI:   [[VLD2:%.*]] = call { <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld2.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x8x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly16x8x2_t [[TMP6]]
// poly16x8x2_t test_vld2q_p16(poly16_t const *a) {
//   return vld2q_p16(a);
// }

// NYI-LABEL: @test_vld2_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x8x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x8x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x8x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint8x8x2_t [[TMP5]]
// uint8x8x2_t test_vld2_u8(uint8_t const *a) {
//   return vld2_u8(a);
// }

// NYI-LABEL: @test_vld2_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x4x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x4x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x4x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint16x4x2_t [[TMP6]]
// uint16x4x2_t test_vld2_u16(uint16_t const *a) {
//   return vld2_u16(a);
// }

// NYI-LABEL: @test_vld2_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x2x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x2x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x2x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint32x2x2_t [[TMP6]]
// uint32x2x2_t test_vld2_u32(uint32_t const *a) {
//   return vld2_u32(a);
// }

// NYI-LABEL: @test_vld2_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x1x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x1x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x1x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint64x1x2_t [[TMP6]]
// uint64x1x2_t test_vld2_u64(uint64_t const *a) {
//   return vld2_u64(a);
// }

// NYI-LABEL: @test_vld2_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x8x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int8x8x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x8x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int8x8x2_t [[TMP5]]
// int8x8x2_t test_vld2_s8(int8_t const *a) {
//   return vld2_s8(a);
// }

// NYI-LABEL: @test_vld2_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x4x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int16x4x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x4x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int16x4x2_t [[TMP6]]
// int16x4x2_t test_vld2_s16(int16_t const *a) {
//   return vld2_s16(a);
// }

// NYI-LABEL: @test_vld2_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x2x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int32x2x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld2.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x2x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int32x2x2_t [[TMP6]]
// int32x2x2_t test_vld2_s32(int32_t const *a) {
//   return vld2_s32(a);
// }

// NYI-LABEL: @test_vld2_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x1x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int64x1x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x1x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int64x1x2_t [[TMP6]]
// int64x1x2_t test_vld2_s64(int64_t const *a) {
//   return vld2_s64(a);
// }

// NYI-LABEL: @test_vld2_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x4x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float16x4x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <4 x half>, <4 x half> } @llvm.aarch64.neon.ld2.v4f16.p0(ptr %a)
// NYI:   store { <4 x half>, <4 x half> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x4x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float16x4x2_t [[TMP6]]
// float16x4x2_t test_vld2_f16(float16_t const *a) {
//   return vld2_f16(a);
// }

// NYI-LABEL: @test_vld2_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x2x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float32x2x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <2 x float>, <2 x float> } @llvm.aarch64.neon.ld2.v2f32.p0(ptr %a)
// NYI:   store { <2 x float>, <2 x float> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x2x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float32x2x2_t [[TMP6]]
// float32x2x2_t test_vld2_f32(float32_t const *a) {
//   return vld2_f32(a);
// }

// NYI-LABEL: @test_vld2_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld2.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x2_t [[TMP6]]
// float64x1x2_t test_vld2_f64(float64_t const *a) {
//   return vld2_f64(a);
// }

// NYI-LABEL: @test_vld2_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x8x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x8x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld2.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x8x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly8x8x2_t [[TMP5]]
// poly8x8x2_t test_vld2_p8(poly8_t const *a) {
//   return vld2_p8(a);
// }

// NYI-LABEL: @test_vld2_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x4x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x4x2_t, align 8
// NYI:   [[VLD2:%.*]] = call { <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld2.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16> } [[VLD2]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x4x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly16x4x2_t [[TMP6]]
// poly16x4x2_t test_vld2_p16(poly16_t const *a) {
//   return vld2_p16(a);
// }

// NYI-LABEL: @test_vld3q_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x16x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x16x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x16x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint8x16x3_t [[TMP5]]
// uint8x16x3_t test_vld3q_u8(uint8_t const *a) {
//   return vld3q_u8(a);
// }

// NYI-LABEL: @test_vld3q_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x8x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x8x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x8x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint16x8x3_t [[TMP6]]
// uint16x8x3_t test_vld3q_u16(uint16_t const *a) {
//   return vld3q_u16(a);
// }

// NYI-LABEL: @test_vld3q_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x4x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x4x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x4x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint32x4x3_t [[TMP6]]
// uint32x4x3_t test_vld3q_u32(uint32_t const *a) {
//   return vld3q_u32(a);
// }

// NYI-LABEL: @test_vld3q_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x2x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x2x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x2x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint64x2x3_t [[TMP6]]
// uint64x2x3_t test_vld3q_u64(uint64_t const *a) {
//   return vld3q_u64(a);
// }

// NYI-LABEL: @test_vld3q_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x16x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int8x16x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x16x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int8x16x3_t [[TMP5]]
// int8x16x3_t test_vld3q_s8(int8_t const *a) {
//   return vld3q_s8(a);
// }

// NYI-LABEL: @test_vld3q_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x8x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int16x8x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x8x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int16x8x3_t [[TMP6]]
// int16x8x3_t test_vld3q_s16(int16_t const *a) {
//   return vld3q_s16(a);
// }

// NYI-LABEL: @test_vld3q_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x4x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int32x4x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld3.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32>, <4 x i32> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x4x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int32x4x3_t [[TMP6]]
// int32x4x3_t test_vld3q_s32(int32_t const *a) {
//   return vld3q_s32(a);
// }

// NYI-LABEL: @test_vld3q_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x2x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int64x2x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x2x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int64x2x3_t [[TMP6]]
// int64x2x3_t test_vld3q_s64(int64_t const *a) {
//   return vld3q_s64(a);
// }

// NYI-LABEL: @test_vld3q_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x8x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float16x8x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld3.v8f16.p0(ptr %a)
// NYI:   store { <8 x half>, <8 x half>, <8 x half> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x8x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float16x8x3_t [[TMP6]]
// float16x8x3_t test_vld3q_f16(float16_t const *a) {
//   return vld3q_f16(a);
// }

// NYI-LABEL: @test_vld3q_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x4x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float32x4x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld3.v4f32.p0(ptr %a)
// NYI:   store { <4 x float>, <4 x float>, <4 x float> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x4x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float32x4x3_t [[TMP6]]
// float32x4x3_t test_vld3q_f32(float32_t const *a) {
//   return vld3q_f32(a);
// }

// NYI-LABEL: @test_vld3q_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld3.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x3_t [[TMP6]]
// float64x2x3_t test_vld3q_f64(float64_t const *a) {
//   return vld3q_f64(a);
// }

// NYI-LABEL: @test_vld3q_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x16x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x16x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld3.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x16x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly8x16x3_t [[TMP5]]
// poly8x16x3_t test_vld3q_p8(poly8_t const *a) {
//   return vld3q_p8(a);
// }

// NYI-LABEL: @test_vld3q_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x8x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x8x3_t, align 16
// NYI:   [[VLD3:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld3.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x8x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly16x8x3_t [[TMP6]]
// poly16x8x3_t test_vld3q_p16(poly16_t const *a) {
//   return vld3q_p16(a);
// }

// NYI-LABEL: @test_vld3_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x8x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x8x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x8x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint8x8x3_t [[TMP5]]
// uint8x8x3_t test_vld3_u8(uint8_t const *a) {
//   return vld3_u8(a);
// }

// NYI-LABEL: @test_vld3_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x4x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x4x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x4x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint16x4x3_t [[TMP6]]
// uint16x4x3_t test_vld3_u16(uint16_t const *a) {
//   return vld3_u16(a);
// }

// NYI-LABEL: @test_vld3_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x2x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x2x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x2x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint32x2x3_t [[TMP6]]
// uint32x2x3_t test_vld3_u32(uint32_t const *a) {
//   return vld3_u32(a);
// }

// NYI-LABEL: @test_vld3_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x1x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x1x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x1x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint64x1x3_t [[TMP6]]
// uint64x1x3_t test_vld3_u64(uint64_t const *a) {
//   return vld3_u64(a);
// }

// NYI-LABEL: @test_vld3_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x8x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int8x8x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x8x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int8x8x3_t [[TMP5]]
// int8x8x3_t test_vld3_s8(int8_t const *a) {
//   return vld3_s8(a);
// }

// NYI-LABEL: @test_vld3_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x4x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int16x4x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x4x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int16x4x3_t [[TMP6]]
// int16x4x3_t test_vld3_s16(int16_t const *a) {
//   return vld3_s16(a);
// }

// NYI-LABEL: @test_vld3_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x2x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int32x2x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld3.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32>, <2 x i32> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x2x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int32x2x3_t [[TMP6]]
// int32x2x3_t test_vld3_s32(int32_t const *a) {
//   return vld3_s32(a);
// }

// NYI-LABEL: @test_vld3_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x1x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int64x1x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x1x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int64x1x3_t [[TMP6]]
// int64x1x3_t test_vld3_s64(int64_t const *a) {
//   return vld3_s64(a);
// }

// NYI-LABEL: @test_vld3_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x4x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float16x4x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld3.v4f16.p0(ptr %a)
// NYI:   store { <4 x half>, <4 x half>, <4 x half> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x4x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float16x4x3_t [[TMP6]]
// float16x4x3_t test_vld3_f16(float16_t const *a) {
//   return vld3_f16(a);
// }

// NYI-LABEL: @test_vld3_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x2x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float32x2x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld3.v2f32.p0(ptr %a)
// NYI:   store { <2 x float>, <2 x float>, <2 x float> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x2x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float32x2x3_t [[TMP6]]
// float32x2x3_t test_vld3_f32(float32_t const *a) {
//   return vld3_f32(a);
// }

// NYI-LABEL: @test_vld3_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld3.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x3_t [[TMP6]]
// float64x1x3_t test_vld3_f64(float64_t const *a) {
//   return vld3_f64(a);
// }

// NYI-LABEL: @test_vld3_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x8x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x8x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld3.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x8x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly8x8x3_t [[TMP5]]
// poly8x8x3_t test_vld3_p8(poly8_t const *a) {
//   return vld3_p8(a);
// }

// NYI-LABEL: @test_vld3_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x4x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x4x3_t, align 8
// NYI:   [[VLD3:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld3.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16> } [[VLD3]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x4x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly16x4x3_t [[TMP6]]
// poly16x4x3_t test_vld3_p16(poly16_t const *a) {
//   return vld3_p16(a);
// }

// NYI-LABEL: @test_vld4q_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x16x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x16x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x16x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint8x16x4_t [[TMP5]]
// uint8x16x4_t test_vld4q_u8(uint8_t const *a) {
//   return vld4q_u8(a);
// }

// NYI-LABEL: @test_vld4q_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x8x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x8x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x8x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint16x8x4_t [[TMP6]]
// uint16x8x4_t test_vld4q_u16(uint16_t const *a) {
//   return vld4q_u16(a);
// }

// NYI-LABEL: @test_vld4q_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x4x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x4x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x4x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint32x4x4_t [[TMP6]]
// uint32x4x4_t test_vld4q_u32(uint32_t const *a) {
//   return vld4q_u32(a);
// }

// NYI-LABEL: @test_vld4q_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x2x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x2x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x2x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.uint64x2x4_t [[TMP6]]
// uint64x2x4_t test_vld4q_u64(uint64_t const *a) {
//   return vld4q_u64(a);
// }

// NYI-LABEL: @test_vld4q_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x16x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int8x16x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x16x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int8x16x4_t [[TMP5]]
// int8x16x4_t test_vld4q_s8(int8_t const *a) {
//   return vld4q_s8(a);
// }

// NYI-LABEL: @test_vld4q_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x8x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int16x8x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x8x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int16x8x4_t [[TMP6]]
// int16x8x4_t test_vld4q_s16(int16_t const *a) {
//   return vld4q_s16(a);
// }

// NYI-LABEL: @test_vld4q_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x4x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int32x4x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } @llvm.aarch64.neon.ld4.v4i32.p0(ptr %a)
// NYI:   store { <4 x i32>, <4 x i32>, <4 x i32>, <4 x i32> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x4x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int32x4x4_t [[TMP6]]
// int32x4x4_t test_vld4q_s32(int32_t const *a) {
//   return vld4q_s32(a);
// }

// NYI-LABEL: @test_vld4q_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x2x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.int64x2x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x2x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.int64x2x4_t [[TMP6]]
// int64x2x4_t test_vld4q_s64(int64_t const *a) {
//   return vld4q_s64(a);
// }

// NYI-LABEL: @test_vld4q_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x8x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float16x8x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <8 x half>, <8 x half>, <8 x half>, <8 x half> } @llvm.aarch64.neon.ld4.v8f16.p0(ptr %a)
// NYI:   store { <8 x half>, <8 x half>, <8 x half>, <8 x half> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x8x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float16x8x4_t [[TMP6]]
// float16x8x4_t test_vld4q_f16(float16_t const *a) {
//   return vld4q_f16(a);
// }

// NYI-LABEL: @test_vld4q_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x4x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float32x4x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <4 x float>, <4 x float>, <4 x float>, <4 x float> } @llvm.aarch64.neon.ld4.v4f32.p0(ptr %a)
// NYI:   store { <4 x float>, <4 x float>, <4 x float>, <4 x float> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x4x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float32x4x4_t [[TMP6]]
// float32x4x4_t test_vld4q_f32(float32_t const *a) {
//   return vld4q_f32(a);
// }

// NYI-LABEL: @test_vld4q_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld4.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x4_t [[TMP6]]
// float64x2x4_t test_vld4q_f64(float64_t const *a) {
//   return vld4q_f64(a);
// }

// NYI-LABEL: @test_vld4q_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x16x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x16x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.aarch64.neon.ld4.v16i8.p0(ptr %a)
// NYI:   store { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x16x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly8x16x4_t [[TMP5]]
// poly8x16x4_t test_vld4q_p8(poly8_t const *a) {
//   return vld4q_p8(a);
// }

// NYI-LABEL: @test_vld4q_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x8x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x8x4_t, align 16
// NYI:   [[VLD4:%.*]] = call { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } @llvm.aarch64.neon.ld4.v8i16.p0(ptr %a)
// NYI:   store { <8 x i16>, <8 x i16>, <8 x i16>, <8 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x8x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly16x8x4_t [[TMP6]]
// poly16x8x4_t test_vld4q_p16(poly16_t const *a) {
//   return vld4q_p16(a);
// }

// NYI-LABEL: @test_vld4_u8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint8x8x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint8x8x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.uint8x8x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint8x8x4_t [[TMP5]]
// uint8x8x4_t test_vld4_u8(uint8_t const *a) {
//   return vld4_u8(a);
// }

// NYI-LABEL: @test_vld4_u16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint16x4x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint16x4x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint16x4x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint16x4x4_t [[TMP6]]
// uint16x4x4_t test_vld4_u16(uint16_t const *a) {
//   return vld4_u16(a);
// }

// NYI-LABEL: @test_vld4_u32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint32x2x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint32x2x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint32x2x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint32x2x4_t [[TMP6]]
// uint32x2x4_t test_vld4_u32(uint32_t const *a) {
//   return vld4_u32(a);
// }

// NYI-LABEL: @test_vld4_u64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.uint64x1x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.uint64x1x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.uint64x1x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.uint64x1x4_t [[TMP6]]
// uint64x1x4_t test_vld4_u64(uint64_t const *a) {
//   return vld4_u64(a);
// }

// NYI-LABEL: @test_vld4_s8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int8x8x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int8x8x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.int8x8x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int8x8x4_t [[TMP5]]
// int8x8x4_t test_vld4_s8(int8_t const *a) {
//   return vld4_s8(a);
// }

// NYI-LABEL: @test_vld4_s16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int16x4x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int16x4x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int16x4x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int16x4x4_t [[TMP6]]
// int16x4x4_t test_vld4_s16(int16_t const *a) {
//   return vld4_s16(a);
// }

// NYI-LABEL: @test_vld4_s32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int32x2x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int32x2x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.aarch64.neon.ld4.v2i32.p0(ptr %a)
// NYI:   store { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int32x2x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int32x2x4_t [[TMP6]]
// int32x2x4_t test_vld4_s32(int32_t const *a) {
//   return vld4_s32(a);
// }

// NYI-LABEL: @test_vld4_s64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.int64x1x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.int64x1x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.int64x1x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.int64x1x4_t [[TMP6]]
// int64x1x4_t test_vld4_s64(int64_t const *a) {
//   return vld4_s64(a);
// }

// NYI-LABEL: @test_vld4_f16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float16x4x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float16x4x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <4 x half>, <4 x half>, <4 x half>, <4 x half> } @llvm.aarch64.neon.ld4.v4f16.p0(ptr %a)
// NYI:   store { <4 x half>, <4 x half>, <4 x half>, <4 x half> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float16x4x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float16x4x4_t [[TMP6]]
// float16x4x4_t test_vld4_f16(float16_t const *a) {
//   return vld4_f16(a);
// }

// NYI-LABEL: @test_vld4_f32(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float32x2x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float32x2x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <2 x float>, <2 x float>, <2 x float>, <2 x float> } @llvm.aarch64.neon.ld4.v2f32.p0(ptr %a)
// NYI:   store { <2 x float>, <2 x float>, <2 x float>, <2 x float> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float32x2x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float32x2x4_t [[TMP6]]
// float32x2x4_t test_vld4_f32(float32_t const *a) {
//   return vld4_f32(a);
// }

// NYI-LABEL: @test_vld4_f64(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld4.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x4_t [[TMP6]]
// float64x1x4_t test_vld4_f64(float64_t const *a) {
//   return vld4_f64(a);
// }

// NYI-LABEL: @test_vld4_p8(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly8x8x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly8x8x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } @llvm.aarch64.neon.ld4.v8i8.p0(ptr %a)
// NYI:   store { <8 x i8>, <8 x i8>, <8 x i8>, <8 x i8> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP5:%.*]] = load %struct.poly8x8x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly8x8x4_t [[TMP5]]
// poly8x8x4_t test_vld4_p8(poly8_t const *a) {
//   return vld4_p8(a);
// }

// NYI-LABEL: @test_vld4_p16(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly16x4x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly16x4x4_t, align 8
// NYI:   [[VLD4:%.*]] = call { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } @llvm.aarch64.neon.ld4.v4i16.p0(ptr %a)
// NYI:   store { <4 x i16>, <4 x i16>, <4 x i16>, <4 x i16> } [[VLD4]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly16x4x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly16x4x4_t [[TMP6]]
// poly16x4x4_t test_vld4_p16(poly16_t const *a) {
//   return vld4_p16(a);
// }

void test_vst1q_u8(uint8_t *a, uint8x16_t b) {
  vst1q_u8(a, b);
  // CIR-LABEL: @test_vst1q_u8
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u8i x 16>>
  // CIR: cir.store align(1) %{{.*}}, %[[CAST]] : !cir.vector<!u8i x 16>, !cir.ptr<!cir.vector<!u8i x 16>>

  // LLVM-LABEL: @test_vst1q_u8
  // LLVM:   store <16 x i8> %{{.*}}, ptr %0, align 1
}

void test_vst1q_u16(uint16_t *a, uint16x8_t b) {
  vst1q_u16(a, b);
  // CIR-LABEL: @test_vst1q_u16
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u16i x 8>>
  // CIR: cir.store align(2) %{{.*}}, %[[CAST]] : !cir.vector<!u16i x 8>, !cir.ptr<!cir.vector<!u16i x 8>>

  // LLVM-LABEL: @test_vst1q_u16
  // LLVM:   store <8 x i16> %{{.*}}, ptr %0, align 2
}

void test_vst1q_u32(uint32_t *a, uint32x4_t b) {
  vst1q_u32(a, b);
  // CIR-LABEL: @test_vst1q_u32
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u32i x 4>>
  // CIR: cir.store align(4) %{{.*}}, %[[CAST]] : !cir.vector<!u32i x 4>, !cir.ptr<!cir.vector<!u32i x 4>>

  // LLVM-LABEL: @test_vst1q_u32
  // LLVM:   store <4 x i32> %{{.*}}, ptr %0, align 4
}

void test_vst1q_u64(uint64_t *a, uint64x2_t b) {
  vst1q_u64(a, b);
  // CIR-LABEL: @test_vst1q_u64
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!u64i x 2>>
  // CIR: cir.store align(8) %{{.*}}, %[[CAST]] : !cir.vector<!u64i x 2>, !cir.ptr<!cir.vector<!u64i x 2>>

  // LLVM-LABEL: @test_vst1q_u64
  // LLVM:   store <2 x i64> %{{.*}}, ptr %0, align 8
}

void test_vst1q_s8(int8_t *a, int8x16_t b) {
  vst1q_s8(a, b);
  // CIR-LABEL: @test_vst1q_s8
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s8i x 16>>
  // CIR: cir.store align(1) %{{.*}}, %[[CAST]] : !cir.vector<!s8i x 16>, !cir.ptr<!cir.vector<!s8i x 16>>

  // LLVM-LABEL: @test_vst1q_s8
  // LLVM:   store <16 x i8> %{{.*}}, ptr %0, align 1
}

void test_vst1q_s16(int16_t *a, int16x8_t b) {
  vst1q_s16(a, b);
  // CIR-LABEL: @test_vst1q_s16
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s16i x 8>>
  // CIR: cir.store align(2) %{{.*}}, %[[CAST]] : !cir.vector<!s16i x 8>, !cir.ptr<!cir.vector<!s16i x 8>>

  // LLVM-LABEL: @test_vst1q_s16
  // LLVM:   store <8 x i16> %{{.*}}, ptr %0, align 2
}

void test_vst1q_s32(int32_t *a, int32x4_t b) {
  vst1q_s32(a, b);
  // CIR-LABEL: @test_vst1q_s32
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s32i x 4>>
  // CIR: cir.store align(4) %{{.*}}, %[[CAST]] : !cir.vector<!s32i x 4>, !cir.ptr<!cir.vector<!s32i x 4>>

  // LLVM-LABEL: @test_vst1q_s32
  // LLVM:   store <4 x i32> %{{.*}}, ptr %0, align 4
}

void test_vst1q_s64(int64_t *a, int64x2_t b) {
  vst1q_s64(a, b);
  // CIR-LABEL: @test_vst1q_s64
  // CIR: %[[CAST:.*]] = cir.cast(bitcast, {{.*}} : !cir.ptr<!void>), !cir.ptr<!cir.vector<!s64i x 2>>
  // CIR: cir.store align(8) %{{.*}}, %[[CAST]] : !cir.vector<!s64i x 2>, !cir.ptr<!cir.vector<!s64i x 2>>

  // LLVM-LABEL: @test_vst1q_s64
  // LLVM:   store <2 x i64> %{{.*}}, ptr %0, align 8
}

// NYI-LABEL: @test_vst1q_f16(
// NYI:   [[TMP1:%.*]] = bitcast <8 x half> %b to <16 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x half>
// NYI:   store <8 x half> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1q_f16(float16_t *a, float16x8_t b) {
//   vst1q_f16(a, b);
// }

// NYI-LABEL: @test_vst1q_f32(
// NYI:   [[TMP1:%.*]] = bitcast <4 x float> %b to <16 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <4 x float>
// NYI:   store <4 x float> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1q_f32(float32_t *a, float32x4_t b) {
//   vst1q_f32(a, b);
// }

// NYI-LABEL: @test_vst1q_f64(
// NYI:   [[TMP1:%.*]] = bitcast <2 x double> %b to <16 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x double>
// NYI:   store <2 x double> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1q_f64(float64_t *a, float64x2_t b) {
//   vst1q_f64(a, b);
// }

// NYI-LABEL: @test_vst1q_p8(
// NYI:   store <16 x i8> %b, ptr %a
// NYI:   ret void
// void test_vst1q_p8(poly8_t *a, poly8x16_t b) {
//   vst1q_p8(a, b);
// }

// NYI-LABEL: @test_vst1q_p16(
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <8 x i16>
// NYI:   store <8 x i16> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1q_p16(poly16_t *a, poly16x8_t b) {
//   vst1q_p16(a, b);
// }

// NYI-LABEL: @test_vst1_u8(
// NYI:   store <8 x i8> %b, ptr %a
// NYI:   ret void
// void test_vst1_u8(uint8_t *a, uint8x8_t b) {
//   vst1_u8(a, b);
// }

// NYI-LABEL: @test_vst1_u16(
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   store <4 x i16> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_u16(uint16_t *a, uint16x4_t b) {
//   vst1_u16(a, b);
// }

// NYI-LABEL: @test_vst1_u32(
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   store <2 x i32> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_u32(uint32_t *a, uint32x2_t b) {
//   vst1_u32(a, b);
// }

// NYI-LABEL: @test_vst1_u64(
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   store <1 x i64> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_u64(uint64_t *a, uint64x1_t b) {
//   vst1_u64(a, b);
// }

// NYI-LABEL: @test_vst1_s8(
// NYI:   store <8 x i8> %b, ptr %a
// NYI:   ret void
// void test_vst1_s8(int8_t *a, int8x8_t b) {
//   vst1_s8(a, b);
// }

// NYI-LABEL: @test_vst1_s16(
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   store <4 x i16> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_s16(int16_t *a, int16x4_t b) {
//   vst1_s16(a, b);
// }

// NYI-LABEL: @test_vst1_s32(
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x i32>
// NYI:   store <2 x i32> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_s32(int32_t *a, int32x2_t b) {
//   vst1_s32(a, b);
// }

// NYI-LABEL: @test_vst1_s64(
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   store <1 x i64> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_s64(int64_t *a, int64x1_t b) {
//   vst1_s64(a, b);
// }

// NYI-LABEL: @test_vst1_f16(
// NYI:   [[TMP1:%.*]] = bitcast <4 x half> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x half>
// NYI:   store <4 x half> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_f16(float16_t *a, float16x4_t b) {
//   vst1_f16(a, b);
// }

// NYI-LABEL: @test_vst1_f32(
// NYI:   [[TMP1:%.*]] = bitcast <2 x float> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <2 x float>
// NYI:   store <2 x float> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_f32(float32_t *a, float32x2_t b) {
//   vst1_f32(a, b);
// }

// NYI-LABEL: @test_vst1_f64(
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x double>
// NYI:   store <1 x double> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_f64(float64_t *a, float64x1_t b) {
//   vst1_f64(a, b);
// }

// NYI-LABEL: @test_vst1_p8(
// NYI:   store <8 x i8> %b, ptr %a
// NYI:   ret void
// void test_vst1_p8(poly8_t *a, poly8x8_t b) {
//   vst1_p8(a, b);
// }

// NYI-LABEL: @test_vst1_p16(
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <4 x i16>
// NYI:   store <4 x i16> [[TMP3]], ptr %a
// NYI:   ret void
// void test_vst1_p16(poly16_t *a, poly16x4_t b) {
//   vst1_p16(a, b);
// }

// NYI-LABEL: @test_vst2q_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x16x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x16x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2q_u8(uint8_t *a, uint8x16x2_t b) {
//   vst2q_u8(a, b);
// }

// NYI-LABEL: @test_vst2q_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x8x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x8x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_u16(uint16_t *a, uint16x8x2_t b) {
//   vst2q_u16(a, b);
// }

// NYI-LABEL: @test_vst2q_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x4x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x4x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_u32(uint32_t *a, uint32x4x2_t b) {
//   vst2q_u32(a, b);
// }

// NYI-LABEL: @test_vst2q_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x2x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x2x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_u64(uint64_t *a, uint64x2x2_t b) {
//   vst2q_u64(a, b);
// }

// NYI-LABEL: @test_vst2q_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x16x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int8x16x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2q_s8(int8_t *a, int8x16x2_t b) {
//   vst2q_s8(a, b);
// }

// NYI-LABEL: @test_vst2q_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x8x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int16x8x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_s16(int16_t *a, int16x8x2_t b) {
//   vst2q_s16(a, b);
// }

// NYI-LABEL: @test_vst2q_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x4x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int32x4x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st2.v4i32.p0(<4 x i32> [[TMP7]], <4 x i32> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_s32(int32_t *a, int32x4x2_t b) {
//   vst2q_s32(a, b);
// }

// NYI-LABEL: @test_vst2q_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x2x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int64x2x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_s64(int64_t *a, int64x2x2_t b) {
//   vst2q_s64(a, b);
// }

// NYI-LABEL: @test_vst2q_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x8x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float16x8x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// NYI:   call void @llvm.aarch64.neon.st2.v8f16.p0(<8 x half> [[TMP7]], <8 x half> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_f16(float16_t *a, float16x8x2_t b) {
//   vst2q_f16(a, b);
// }

// NYI-LABEL: @test_vst2q_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x4x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float32x4x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// NYI:   call void @llvm.aarch64.neon.st2.v4f32.p0(<4 x float> [[TMP7]], <4 x float> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_f32(float32_t *a, float32x4x2_t b) {
//   vst2q_f32(a, b);
// }

// NYI-LABEL: @test_vst2q_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st2.v2f64.p0(<2 x double> [[TMP7]], <2 x double> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_f64(float64_t *a, float64x2x2_t b) {
//   vst2q_f64(a, b);
// }

// NYI-LABEL: @test_vst2q_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x16x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x16x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   call void @llvm.aarch64.neon.st2.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2q_p8(poly8_t *a, poly8x16x2_t b) {
//   vst2q_p8(a, b);
// }

// NYI-LABEL: @test_vst2q_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x8x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x8x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v8i16.p0(<8 x i16> [[TMP7]], <8 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2q_p16(poly16_t *a, poly16x8x2_t b) {
//   vst2q_p16(a, b);
// }

// NYI-LABEL: @test_vst2_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x8x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2_u8(uint8_t *a, uint8x8x2_t b) {
//   vst2_u8(a, b);
// }

// NYI-LABEL: @test_vst2_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x4x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x4x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_u16(uint16_t *a, uint16x4x2_t b) {
//   vst2_u16(a, b);
// }

// NYI-LABEL: @test_vst2_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x2x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x2x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_u32(uint32_t *a, uint32x2x2_t b) {
//   vst2_u32(a, b);
// }

// NYI-LABEL: @test_vst2_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x1x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x1x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_u64(uint64_t *a, uint64x1x2_t b) {
//   vst2_u64(a, b);
// }

// NYI-LABEL: @test_vst2_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int8x8x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2_s8(int8_t *a, int8x8x2_t b) {
//   vst2_s8(a, b);
// }

// NYI-LABEL: @test_vst2_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x4x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int16x4x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_s16(int16_t *a, int16x4x2_t b) {
//   vst2_s16(a, b);
// }

// NYI-LABEL: @test_vst2_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x2x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int32x2x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st2.v2i32.p0(<2 x i32> [[TMP7]], <2 x i32> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_s32(int32_t *a, int32x2x2_t b) {
//   vst2_s32(a, b);
// }

// NYI-LABEL: @test_vst2_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x1x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int64x1x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_s64(int64_t *a, int64x1x2_t b) {
//   vst2_s64(a, b);
// }

// NYI-LABEL: @test_vst2_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x4x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float16x4x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// NYI:   call void @llvm.aarch64.neon.st2.v4f16.p0(<4 x half> [[TMP7]], <4 x half> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_f16(float16_t *a, float16x4x2_t b) {
//   vst2_f16(a, b);
// }

// NYI-LABEL: @test_vst2_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x2x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float32x2x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// NYI:   call void @llvm.aarch64.neon.st2.v2f32.p0(<2 x float> [[TMP7]], <2 x float> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_f32(float32_t *a, float32x2x2_t b) {
//   vst2_f32(a, b);
// }

// NYI-LABEL: @test_vst2_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st2.v1f64.p0(<1 x double> [[TMP7]], <1 x double> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_f64(float64_t *a, float64x1x2_t b) {
//   vst2_f64(a, b);
// }

// NYI-LABEL: @test_vst2_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x8x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   call void @llvm.aarch64.neon.st2.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], ptr %a)
// NYI:   ret void
// void test_vst2_p8(poly8_t *a, poly8x8x2_t b) {
//   vst2_p8(a, b);
// }

// NYI-LABEL: @test_vst2_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x4x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x4x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st2.v4i16.p0(<4 x i16> [[TMP7]], <4 x i16> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst2_p16(poly16_t *a, poly16x4x2_t b) {
//   vst2_p16(a, b);
// }

// NYI-LABEL: @test_vst3q_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x16x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x16x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3q_u8(uint8_t *a, uint8x16x3_t b) {
//   vst3q_u8(a, b);
// }

// NYI-LABEL: @test_vst3q_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x8x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x8x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_u16(uint16_t *a, uint16x8x3_t b) {
//   vst3q_u16(a, b);
// }

// NYI-LABEL: @test_vst3q_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x4x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x4x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_u32(uint32_t *a, uint32x4x3_t b) {
//   vst3q_u32(a, b);
// }

// NYI-LABEL: @test_vst3q_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x2x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x2x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_u64(uint64_t *a, uint64x2x3_t b) {
//   vst3q_u64(a, b);
// }

// NYI-LABEL: @test_vst3q_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x16x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int8x16x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3q_s8(int8_t *a, int8x16x3_t b) {
//   vst3q_s8(a, b);
// }

// NYI-LABEL: @test_vst3q_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x8x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int16x8x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_s16(int16_t *a, int16x8x3_t b) {
//   vst3q_s16(a, b);
// }

// NYI-LABEL: @test_vst3q_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x4x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int32x4x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st3.v4i32.p0(<4 x i32> [[TMP9]], <4 x i32> [[TMP10]], <4 x i32> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_s32(int32_t *a, int32x4x3_t b) {
//   vst3q_s32(a, b);
// }

// NYI-LABEL: @test_vst3q_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x2x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int64x2x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_s64(int64_t *a, int64x2x3_t b) {
//   vst3q_s64(a, b);
// }

// NYI-LABEL: @test_vst3q_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x8x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float16x8x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x half>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// NYI:   call void @llvm.aarch64.neon.st3.v8f16.p0(<8 x half> [[TMP9]], <8 x half> [[TMP10]], <8 x half> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_f16(float16_t *a, float16x8x3_t b) {
//   vst3q_f16(a, b);
// }

// NYI-LABEL: @test_vst3q_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x4x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float32x4x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x float>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// NYI:   call void @llvm.aarch64.neon.st3.v4f32.p0(<4 x float> [[TMP9]], <4 x float> [[TMP10]], <4 x float> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_f32(float32_t *a, float32x4x3_t b) {
//   vst3q_f32(a, b);
// }

// NYI-LABEL: @test_vst3q_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st3.v2f64.p0(<2 x double> [[TMP9]], <2 x double> [[TMP10]], <2 x double> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_f64(float64_t *a, float64x2x3_t b) {
//   vst3q_f64(a, b);
// }

// NYI-LABEL: @test_vst3q_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x16x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   call void @llvm.aarch64.neon.st3.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3q_p8(poly8_t *a, poly8x16x3_t b) {
//   vst3q_p8(a, b);
// }

// NYI-LABEL: @test_vst3q_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x8x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x8x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v8i16.p0(<8 x i16> [[TMP9]], <8 x i16> [[TMP10]], <8 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3q_p16(poly16_t *a, poly16x8x3_t b) {
//   vst3q_p16(a, b);
// }

// NYI-LABEL: @test_vst3_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x8x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3_u8(uint8_t *a, uint8x8x3_t b) {
//   vst3_u8(a, b);
// }

// NYI-LABEL: @test_vst3_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x4x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x4x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_u16(uint16_t *a, uint16x4x3_t b) {
//   vst3_u16(a, b);
// }

// NYI-LABEL: @test_vst3_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x2x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x2x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_u32(uint32_t *a, uint32x2x3_t b) {
//   vst3_u32(a, b);
// }

// NYI-LABEL: @test_vst3_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x1x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x1x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_u64(uint64_t *a, uint64x1x3_t b) {
//   vst3_u64(a, b);
// }

// NYI-LABEL: @test_vst3_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int8x8x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3_s8(int8_t *a, int8x8x3_t b) {
//   vst3_s8(a, b);
// }

// NYI-LABEL: @test_vst3_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x4x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int16x4x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_s16(int16_t *a, int16x4x3_t b) {
//   vst3_s16(a, b);
// }

// NYI-LABEL: @test_vst3_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x2x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int32x2x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st3.v2i32.p0(<2 x i32> [[TMP9]], <2 x i32> [[TMP10]], <2 x i32> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_s32(int32_t *a, int32x2x3_t b) {
//   vst3_s32(a, b);
// }

// NYI-LABEL: @test_vst3_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x1x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int64x1x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_s64(int64_t *a, int64x1x3_t b) {
//   vst3_s64(a, b);
// }

// NYI-LABEL: @test_vst3_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x4x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float16x4x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x half>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// NYI:   call void @llvm.aarch64.neon.st3.v4f16.p0(<4 x half> [[TMP9]], <4 x half> [[TMP10]], <4 x half> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_f16(float16_t *a, float16x4x3_t b) {
//   vst3_f16(a, b);
// }

// NYI-LABEL: @test_vst3_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x2x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float32x2x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x float>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// NYI:   call void @llvm.aarch64.neon.st3.v2f32.p0(<2 x float> [[TMP9]], <2 x float> [[TMP10]], <2 x float> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_f32(float32_t *a, float32x2x3_t b) {
//   vst3_f32(a, b);
// }

// NYI-LABEL: @test_vst3_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st3.v1f64.p0(<1 x double> [[TMP9]], <1 x double> [[TMP10]], <1 x double> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_f64(float64_t *a, float64x1x3_t b) {
//   vst3_f64(a, b);
// }

// NYI-LABEL: @test_vst3_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x8x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   call void @llvm.aarch64.neon.st3.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], ptr %a)
// NYI:   ret void
// void test_vst3_p8(poly8_t *a, poly8x8x3_t b) {
//   vst3_p8(a, b);
// }

// NYI-LABEL: @test_vst3_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x4x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x4x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st3.v4i16.p0(<4 x i16> [[TMP9]], <4 x i16> [[TMP10]], <4 x i16> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst3_p16(poly16_t *a, poly16x4x3_t b) {
//   vst3_p16(a, b);
// }

// NYI-LABEL: @test_vst4q_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x16x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// NYI:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4q_u8(uint8_t *a, uint8x16x4_t b) {
//   vst4q_u8(a, b);
// }

// NYI-LABEL: @test_vst4q_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x8x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x8x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_u16(uint16_t *a, uint16x8x4_t b) {
//   vst4q_u16(a, b);
// }

// NYI-LABEL: @test_vst4q_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x4x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x4x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_u32(uint32_t *a, uint32x4x4_t b) {
//   vst4q_u32(a, b);
// }

// NYI-LABEL: @test_vst4q_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x2x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x2x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_u64(uint64_t *a, uint64x2x4_t b) {
//   vst4q_u64(a, b);
// }

// NYI-LABEL: @test_vst4q_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int8x16x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// NYI:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4q_s8(int8_t *a, int8x16x4_t b) {
//   vst4q_s8(a, b);
// }

// NYI-LABEL: @test_vst4q_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x8x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int16x8x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_s16(int16_t *a, int16x8x4_t b) {
//   vst4q_s16(a, b);
// }

// NYI-LABEL: @test_vst4q_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x4x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int32x4x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i32>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x i32> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i32>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x i32> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i32>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x i32> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i32>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x i32>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <4 x i32> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x i32>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x i32>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x i32>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x i32>
// NYI:   call void @llvm.aarch64.neon.st4.v4i32.p0(<4 x i32> [[TMP11]], <4 x i32> [[TMP12]], <4 x i32> [[TMP13]], <4 x i32> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_s32(int32_t *a, int32x4x4_t b) {
//   vst4q_s32(a, b);
// }

// NYI-LABEL: @test_vst4q_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x2x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.int64x2x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_s64(int64_t *a, int64x2x4_t b) {
//   vst4q_s64(a, b);
// }

// NYI-LABEL: @test_vst4q_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x8x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float16x8x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x half>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x half> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x half>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x half> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x half>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x half> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x half>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <8 x half>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <8 x half> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x half>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x half>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x half>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x half>
// NYI:   call void @llvm.aarch64.neon.st4.v8f16.p0(<8 x half> [[TMP11]], <8 x half> [[TMP12]], <8 x half> [[TMP13]], <8 x half> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_f16(float16_t *a, float16x8x4_t b) {
//   vst4q_f16(a, b);
// }

// NYI-LABEL: @test_vst4q_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x4x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float32x4x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x float>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <4 x float> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x float>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <4 x float> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x float>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <4 x float> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x float>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x float>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <4 x float> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <4 x float>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <4 x float>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <4 x float>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <4 x float>
// NYI:   call void @llvm.aarch64.neon.st4.v4f32.p0(<4 x float> [[TMP11]], <4 x float> [[TMP12]], <4 x float> [[TMP13]], <4 x float> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_f32(float32_t *a, float32x4x4_t b) {
//   vst4q_f32(a, b);
// }

// NYI-LABEL: @test_vst4q_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x double>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <2 x double> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st4.v2f64.p0(<2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_f64(float64_t *a, float64x2x4_t b) {
//   vst4q_f64(a, b);
// }

// NYI-LABEL: @test_vst4q_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x16x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <16 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <16 x i8>, ptr [[ARRAYIDX]], align 16
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <16 x i8>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <16 x i8>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <16 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <16 x i8>, ptr [[ARRAYIDX6]], align 16
// NYI:   call void @llvm.aarch64.neon.st4.v16i8.p0(<16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4q_p8(poly8_t *a, poly8x16x4_t b) {
//   vst4q_p8(a, b);
// }

// NYI-LABEL: @test_vst4q_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x8x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x8x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <8 x i16>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <8 x i16> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <8 x i16>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <8 x i16> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <8 x i16>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <8 x i16> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <8 x i16>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <8 x i16> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <8 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <8 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <8 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <8 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v8i16.p0(<8 x i16> [[TMP11]], <8 x i16> [[TMP12]], <8 x i16> [[TMP13]], <8 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4q_p16(poly16_t *a, poly16x8x4_t b) {
//   vst4q_p16(a, b);
// }

// NYI-LABEL: @test_vst4_u8(
// NYI:   [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint8x8x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// NYI:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4_u8(uint8_t *a, uint8x8x4_t b) {
//   vst4_u8(a, b);
// }

// NYI-LABEL: @test_vst4_u16(
// NYI:   [[B:%.*]] = alloca %struct.uint16x4x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint16x4x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_u16(uint16_t *a, uint16x4x4_t b) {
//   vst4_u16(a, b);
// }

// NYI-LABEL: @test_vst4_u32(
// NYI:   [[B:%.*]] = alloca %struct.uint32x2x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint32x2x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_u32(uint32_t *a, uint32x2x4_t b) {
//   vst4_u32(a, b);
// }

// NYI-LABEL: @test_vst4_u64(
// NYI:   [[B:%.*]] = alloca %struct.uint64x1x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.uint64x1x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.uint64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_u64(uint64_t *a, uint64x1x4_t b) {
//   vst4_u64(a, b);
// }

// NYI-LABEL: @test_vst4_s8(
// NYI:   [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int8x8x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// NYI:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4_s8(int8_t *a, int8x8x4_t b) {
//   vst4_s8(a, b);
// }

// NYI-LABEL: @test_vst4_s16(
// NYI:   [[B:%.*]] = alloca %struct.int16x4x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int16x4x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int16x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_s16(int16_t *a, int16x4x4_t b) {
//   vst4_s16(a, b);
// }

// NYI-LABEL: @test_vst4_s32(
// NYI:   [[B:%.*]] = alloca %struct.int32x2x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int32x2x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int32x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x i32>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i32>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x i32> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i32>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x i32> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i32>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x i32> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i32>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x i32>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <2 x i32> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x i32>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x i32>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x i32>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x i32>
// NYI:   call void @llvm.aarch64.neon.st4.v2i32.p0(<2 x i32> [[TMP11]], <2 x i32> [[TMP12]], <2 x i32> [[TMP13]], <2 x i32> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_s32(int32_t *a, int32x2x4_t b) {
//   vst4_s32(a, b);
// }

// NYI-LABEL: @test_vst4_s64(
// NYI:   [[B:%.*]] = alloca %struct.int64x1x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.int64x1x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int64x1x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.int64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_s64(int64_t *a, int64x1x4_t b) {
//   vst4_s64(a, b);
// }

// NYI-LABEL: @test_vst4_f16(
// NYI:   [[B:%.*]] = alloca %struct.float16x4x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float16x4x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float16x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x half>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x half>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x half> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x half>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x half> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x half>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x half> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x half>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x half>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <4 x half> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x half>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x half>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x half>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x half>
// NYI:   call void @llvm.aarch64.neon.st4.v4f16.p0(<4 x half> [[TMP11]], <4 x half> [[TMP12]], <4 x half> [[TMP13]], <4 x half> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_f16(float16_t *a, float16x4x4_t b) {
//   vst4_f16(a, b);
// }

// NYI-LABEL: @test_vst4_f32(
// NYI:   [[B:%.*]] = alloca %struct.float32x2x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float32x2x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float32x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x float>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x float>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <2 x float> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x float>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <2 x float> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x float>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <2 x float> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float32x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x float>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x float>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <2 x float> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <2 x float>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <2 x float>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <2 x float>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <2 x float>
// NYI:   call void @llvm.aarch64.neon.st4.v2f32.p0(<2 x float> [[TMP11]], <2 x float> [[TMP12]], <2 x float> [[TMP13]], <2 x float> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_f32(float32_t *a, float32x2x4_t b) {
//   vst4_f32(a, b);
// }

// NYI-LABEL: @test_vst4_f64(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <1 x double>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <1 x double> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st4.v1f64.p0(<1 x double> [[TMP11]], <1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_f64(float64_t *a, float64x1x4_t b) {
//   vst4_f64(a, b);
// }

// NYI-LABEL: @test_vst4_p8(
// NYI:   [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly8x8x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <8 x i8>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP2:%.*]] = load <8 x i8>, ptr [[ARRAYIDX]], align 8
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP3:%.*]] = load <8 x i8>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP4:%.*]] = load <8 x i8>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <8 x i8>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP5:%.*]] = load <8 x i8>, ptr [[ARRAYIDX6]], align 8
// NYI:   call void @llvm.aarch64.neon.st4.v8i8.p0(<8 x i8> [[TMP2]], <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <8 x i8> [[TMP5]], ptr %a)
// NYI:   ret void
// void test_vst4_p8(poly8_t *a, poly8x8x4_t b) {
//   vst4_p8(a, b);
// }

// NYI-LABEL: @test_vst4_p16(
// NYI:   [[B:%.*]] = alloca %struct.poly16x4x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly16x4x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <4 x i16>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <4 x i16>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <4 x i16> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <4 x i16>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <4 x i16> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <4 x i16>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <4 x i16> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly16x4x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <4 x i16>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <4 x i16>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <4 x i16> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <4 x i16>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <4 x i16>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <4 x i16>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <4 x i16>
// NYI:   call void @llvm.aarch64.neon.st4.v4i16.p0(<4 x i16> [[TMP11]], <4 x i16> [[TMP12]], <4 x i16> [[TMP13]], <4 x i16> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst4_p16(poly16_t *a, poly16x4x4_t b) {
//   vst4_p16(a, b);
// }

// NYI-LABEL: @test_vld1q_f64_x2(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x2.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x2_t [[TMP6]]
// float64x2x2_t test_vld1q_f64_x2(float64_t const *a) {
//   return vld1q_f64_x2(a);
// }

// NYI-LABEL: @test_vld1q_p64_x2(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x2x2_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x2.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x2x2_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly64x2x2_t [[TMP6]]
// poly64x2x2_t test_vld1q_p64_x2(poly64_t const *a) {
//   return vld1q_p64_x2(a);
// }

// NYI-LABEL: @test_vld1_f64_x2(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x2.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x2_t [[TMP6]]
// float64x1x2_t test_vld1_f64_x2(float64_t const *a) {
//   return vld1_f64_x2(a);
// }

// NYI-LABEL: @test_vld1_p64_x2(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x1x2_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x2.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x1x2_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly64x1x2_t [[TMP6]]
// poly64x1x2_t test_vld1_p64_x2(poly64_t const *a) {
//   return vld1_p64_x2(a);
// }

// NYI-LABEL: @test_vld1q_f64_x3(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x3.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x3_t [[TMP6]]
// float64x2x3_t test_vld1q_f64_x3(float64_t const *a) {
//   return vld1q_f64_x3(a);
// }

// NYI-LABEL: @test_vld1q_p64_x3(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x2x3_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x3.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x2x3_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly64x2x3_t [[TMP6]]
// poly64x2x3_t test_vld1q_p64_x3(poly64_t const *a) {
//   return vld1q_p64_x3(a);
// }

// NYI-LABEL: @test_vld1_f64_x3(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x3.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x3_t [[TMP6]]
// float64x1x3_t test_vld1_f64_x3(float64_t const *a) {
//   return vld1_f64_x3(a);
// }

// NYI-LABEL: @test_vld1_p64_x3(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x1x3_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x3.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x1x3_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly64x1x3_t [[TMP6]]
// poly64x1x3_t test_vld1_p64_x3(poly64_t const *a) {
//   return vld1_p64_x3(a);
// }

// NYI-LABEL: @test_vld1q_f64_x4(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x double>, <2 x double>, <2 x double>, <2 x double> } @llvm.aarch64.neon.ld1x4.v2f64.p0(ptr %a)
// NYI:   store { <2 x double>, <2 x double>, <2 x double>, <2 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x2x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.float64x2x4_t [[TMP6]]
// float64x2x4_t test_vld1q_f64_x4(float64_t const *a) {
//   return vld1q_f64_x4(a);
// }

// NYI-LABEL: @test_vld1q_p64_x4(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x2x4_t, align 16
// NYI:   [[VLD1XN:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld1x4.v2i64.p0(ptr %a)
// NYI:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x2x4_t, ptr [[RETVAL]], align 16
// NYI:   ret %struct.poly64x2x4_t [[TMP6]]
// poly64x2x4_t test_vld1q_p64_x4(poly64_t const *a) {
//   return vld1q_p64_x4(a);
// }

// NYI-LABEL: @test_vld1_f64_x4(
// NYI:   [[RETVAL:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x double>, <1 x double>, <1 x double>, <1 x double> } @llvm.aarch64.neon.ld1x4.v1f64.p0(ptr %a)
// NYI:   store { <1 x double>, <1 x double>, <1 x double>, <1 x double> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.float64x1x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.float64x1x4_t [[TMP6]]
// float64x1x4_t test_vld1_f64_x4(float64_t const *a) {
//   return vld1_f64_x4(a);
// }

// NYI-LABEL: @test_vld1_p64_x4(
// NYI:   [[RETVAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// NYI:   [[__RET:%.*]] = alloca %struct.poly64x1x4_t, align 8
// NYI:   [[VLD1XN:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld1x4.v1i64.p0(ptr %a)
// NYI:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD1XN]], ptr [[__RET]]
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// NYI:   [[TMP6:%.*]] = load %struct.poly64x1x4_t, ptr [[RETVAL]], align 8
// NYI:   ret %struct.poly64x1x4_t [[TMP6]]
// poly64x1x4_t test_vld1_p64_x4(poly64_t const *a) {
//   return vld1_p64_x4(a);
// }

// NYI-LABEL: @test_vst1q_f64_x2(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st1x2.v2f64.p0(<2 x double> [[TMP7]], <2 x double> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst1q_f64_x2(float64_t *a, float64x2x2_t b) {
//   vst1q_f64_x2(a, b);
// }

// NYI-LABEL: @test_vst1q_p64_x2(
// NYI:   [[B:%.*]] = alloca %struct.poly64x2x2_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x2x2_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst1q_p64_x2(poly64_t *a, poly64x2x2_t b) {
//   vst1q_p64_x2(a, b);
// }

// NYI-LABEL: @test_vst1_f64_x2(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st1x2.v1f64.p0(<1 x double> [[TMP7]], <1 x double> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst1_f64_x2(float64_t *a, float64x1x2_t b) {
//   vst1_f64_x2(a, b);
// }

// NYI-LABEL: @test_vst1_p64_x2(
// NYI:   [[B:%.*]] = alloca %struct.poly64x1x2_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x1x2_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[B]], i32 0, i32 0
// NYI:   store [2 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 16, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %a)
// NYI:   ret void
// void test_vst1_p64_x2(poly64_t *a, poly64x1x2_t b) {
//   vst1_p64_x2(a, b);
// }

// NYI-LABEL: @test_vst1q_f64_x3(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st1x3.v2f64.p0(<2 x double> [[TMP9]], <2 x double> [[TMP10]], <2 x double> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst1q_f64_x3(float64_t *a, float64x2x3_t b) {
//   vst1q_f64_x3(a, b);
// }

// NYI-LABEL: @test_vst1q_p64_x3(
// NYI:   [[B:%.*]] = alloca %struct.poly64x2x3_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x2x3_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 48, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst1q_p64_x3(poly64_t *a, poly64x2x3_t b) {
//   vst1q_p64_x3(a, b);
// }

// NYI-LABEL: @test_vst1_f64_x3(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st1x3.v1f64.p0(<1 x double> [[TMP9]], <1 x double> [[TMP10]], <1 x double> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst1_f64_x3(float64_t *a, float64x1x3_t b) {
//   vst1_f64_x3(a, b);
// }

// NYI-LABEL: @test_vst1_p64_x3(
// NYI:   [[B:%.*]] = alloca %struct.poly64x1x3_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x1x3_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[B]], i32 0, i32 0
// NYI:   store [3 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 24, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %a)
// NYI:   ret void
// void test_vst1_p64_x3(poly64_t *a, poly64x1x3_t b) {
//   vst1_p64_x3(a, b);
// }

// NYI-LABEL: @test_vst1q_f64_x4(
// NYI:   [[B:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.float64x2x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x double>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x double> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x double>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x double> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x double>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x double> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x double>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x double>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <2 x double> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x double>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x double>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x double>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x double>
// NYI:   call void @llvm.aarch64.neon.st1x4.v2f64.p0(<2 x double> [[TMP11]], <2 x double> [[TMP12]], <2 x double> [[TMP13]], <2 x double> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst1q_f64_x4(float64_t *a, float64x2x4_t b) {
//   vst1q_f64_x4(a, b);
// }

// NYI-LABEL: @test_vst1q_p64_x4(
// NYI:   [[B:%.*]] = alloca %struct.poly64x2x4_t, align 16
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x2x4_t, align 16
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <2 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 16
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[B]], i64 64, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// NYI:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX2]], align 16
// NYI:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX4]], align 16
// NYI:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX6]], align 16
// NYI:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst1q_p64_x4(poly64_t *a, poly64x2x4_t b) {
//   vst1q_p64_x4(a, b);
// }

// NYI-LABEL: @test_vst1_f64_x4(
// NYI:   [[B:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.float64x1x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <1 x double>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x double>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x double> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x double>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x double> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x double>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x double> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.float64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x double>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <1 x double>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <1 x double> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x double>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x double>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x double>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x double>
// NYI:   call void @llvm.aarch64.neon.st1x4.v1f64.p0(<1 x double> [[TMP11]], <1 x double> [[TMP12]], <1 x double> [[TMP13]], <1 x double> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst1_f64_x4(float64_t *a, float64x1x4_t b) {
//   vst1_f64_x4(a, b);
// }

// NYI-LABEL: @test_vst1_p64_x4(
// NYI:   [[B:%.*]] = alloca %struct.poly64x1x4_t, align 8
// NYI:   [[__S1:%.*]] = alloca %struct.poly64x1x4_t, align 8
// NYI:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[B]], i32 0, i32 0
// NYI:   store [4 x <1 x i64>] [[B]].coerce, ptr [[COERCE_DIVE]], align 8
// NYI:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[B]], i64 32, i1 false)
// NYI:   [[VAL:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL]], i64 0, i64 0
// NYI:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// NYI:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// NYI:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX2:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 1
// NYI:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX2]], align 8
// NYI:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// NYI:   [[VAL3:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX4:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL3]], i64 0, i64 2
// NYI:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX4]], align 8
// NYI:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// NYI:   [[VAL5:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// NYI:   [[ARRAYIDX6:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL5]], i64 0, i64 3
// NYI:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX6]], align 8
// NYI:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// NYI:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// NYI:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// NYI:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// NYI:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// NYI:   call void @llvm.aarch64.neon.st1x4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %a)
// NYI:   ret void
// void test_vst1_p64_x4(poly64_t *a, poly64x1x4_t b) {
//   vst1_p64_x4(a, b);
// }

// NYI-LABEL: @test_vceqd_s64(
// NYI:   [[TMP0:%.*]] = icmp eq i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vceqd_s64(int64_t a, int64_t b) {
//   return (uint64_t)vceqd_s64(a, b);
// }

// NYI-LABEL: @test_vceqd_u64(
// NYI:   [[TMP0:%.*]] = icmp eq i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vceqd_u64(uint64_t a, uint64_t b) {
//   return (int64_t)vceqd_u64(a, b);
// }

// NYI-LABEL: @test_vceqzd_s64(
// NYI:   [[TMP0:%.*]] = icmp eq i64 %a, 0
// NYI:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQZ_I]]
// uint64_t test_vceqzd_s64(int64_t a) {
//   return (uint64_t)vceqzd_s64(a);
// }

// NYI-LABEL: @test_vceqzd_u64(
// NYI:   [[TMP0:%.*]] = icmp eq i64 %a, 0
// NYI:   [[VCEQZD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQZD_I]]
// int64_t test_vceqzd_u64(int64_t a) {
//   return (int64_t)vceqzd_u64(a);
// }

// NYI-LABEL: @test_vcged_s64(
// NYI:   [[TMP0:%.*]] = icmp sge i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcged_s64(int64_t a, int64_t b) {
//   return (uint64_t)vcged_s64(a, b);
// }

// NYI-LABEL: @test_vcged_u64(
// NYI:   [[TMP0:%.*]] = icmp uge i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcged_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vcged_u64(a, b);
// }

// NYI-LABEL: @test_vcgezd_s64(
// NYI:   [[TMP0:%.*]] = icmp sge i64 %a, 0
// NYI:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCGEZ_I]]
// uint64_t test_vcgezd_s64(int64_t a) {
//   return (uint64_t)vcgezd_s64(a);
// }

// NYI-LABEL: @test_vcgtd_s64(
// NYI:   [[TMP0:%.*]] = icmp sgt i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcgtd_s64(int64_t a, int64_t b) {
//   return (uint64_t)vcgtd_s64(a, b);
// }

// NYI-LABEL: @test_vcgtd_u64(
// NYI:   [[TMP0:%.*]] = icmp ugt i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcgtd_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vcgtd_u64(a, b);
// }

// NYI-LABEL: @test_vcgtzd_s64(
// NYI:   [[TMP0:%.*]] = icmp sgt i64 %a, 0
// NYI:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCGTZ_I]]
// uint64_t test_vcgtzd_s64(int64_t a) {
//   return (uint64_t)vcgtzd_s64(a);
// }

// NYI-LABEL: @test_vcled_s64(
// NYI:   [[TMP0:%.*]] = icmp sle i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcled_s64(int64_t a, int64_t b) {
//   return (uint64_t)vcled_s64(a, b);
// }

// NYI-LABEL: @test_vcled_u64(
// NYI:   [[TMP0:%.*]] = icmp ule i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcled_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vcled_u64(a, b);
// }

// NYI-LABEL: @test_vclezd_s64(
// NYI:   [[TMP0:%.*]] = icmp sle i64 %a, 0
// NYI:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCLEZ_I]]
// uint64_t test_vclezd_s64(int64_t a) {
//   return (uint64_t)vclezd_s64(a);
// }

// NYI-LABEL: @test_vcltd_s64(
// NYI:   [[TMP0:%.*]] = icmp slt i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcltd_s64(int64_t a, int64_t b) {
//   return (uint64_t)vcltd_s64(a, b);
// }

// NYI-LABEL: @test_vcltd_u64(
// NYI:   [[TMP0:%.*]] = icmp ult i64 %a, %b
// NYI:   [[VCEQD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQD_I]]
// uint64_t test_vcltd_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vcltd_u64(a, b);
// }

// NYI-LABEL: @test_vcltzd_s64(
// NYI:   [[TMP0:%.*]] = icmp slt i64 %a, 0
// NYI:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCLTZ_I]]
// uint64_t test_vcltzd_s64(int64_t a) {
//   return (uint64_t)vcltzd_s64(a);
// }

// NYI-LABEL: @test_vtstd_s64(
// NYI:   [[TMP0:%.*]] = and i64 %a, %b
// NYI:   [[TMP1:%.*]] = icmp ne i64 [[TMP0]], 0
// NYI:   [[VTSTD_I:%.*]] = sext i1 [[TMP1]] to i64
// NYI:   ret i64 [[VTSTD_I]]
// uint64_t test_vtstd_s64(int64_t a, int64_t b) {
//   return (uint64_t)vtstd_s64(a, b);
// }

// NYI-LABEL: @test_vtstd_u64(
// NYI:   [[TMP0:%.*]] = and i64 %a, %b
// NYI:   [[TMP1:%.*]] = icmp ne i64 [[TMP0]], 0
// NYI:   [[VTSTD_I:%.*]] = sext i1 [[TMP1]] to i64
// NYI:   ret i64 [[VTSTD_I]]
// uint64_t test_vtstd_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vtstd_u64(a, b);
// }

int64_t test_vabsd_s64(int64_t a) {
  return (int64_t)vabsd_s64(a);

  // CIR-LABEL: vabsd_s64
  // CIR: cir.llvm.intrinsic "aarch64.neon.abs" {{%.*}} : (!s64i) -> !s64i

  // LLVM-LABEL: @test_vabsd_s64
  // LLVM-SAME: (i64 [[a:%.*]])
  // LLVM:   [[VABSD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.abs.i64(i64 [[a]])
  // LLVM:   ret i64 [[VABSD_S64_I]]
}

// NYI-LABEL: @test_vqabsb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[VQABSB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqabs.v8i8(<8 x i8> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQABSB_S8_I]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqabsb_s8(int8_t a) {
//   return (int8_t)vqabsb_s8(a);
// }

// NYI-LABEL: @test_vqabsh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[VQABSH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqabs.v4i16(<4 x i16> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQABSH_S16_I]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqabsh_s16(int16_t a) {
//   return (int16_t)vqabsh_s16(a);
// }

// NYI-LABEL: @test_vqabss_s32(
// NYI:   [[VQABSS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqabs.i32(i32 %a)
// NYI:   ret i32 [[VQABSS_S32_I]]
// int32_t test_vqabss_s32(int32_t a) {
//   return (int32_t)vqabss_s32(a);
// }

// NYI-LABEL: @test_vqabsd_s64(
// NYI:   [[VQABSD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqabs.i64(i64 %a)
// NYI:   ret i64 [[VQABSD_S64_I]]
// int64_t test_vqabsd_s64(int64_t a) {
//   return (int64_t)vqabsd_s64(a);
// }

// NYI-LABEL: @test_vnegd_s64(
// NYI:   [[VNEGD_I:%.*]] = sub i64 0, %a
// NYI:   ret i64 [[VNEGD_I]]
// int64_t test_vnegd_s64(int64_t a) {
//   return (int64_t)vnegd_s64(a);
// }

// NYI-LABEL: @test_vqnegb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[VQNEGB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqneg.v8i8(<8 x i8> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQNEGB_S8_I]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqnegb_s8(int8_t a) {
//   return (int8_t)vqnegb_s8(a);
// }

// NYI-LABEL: @test_vqnegh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[VQNEGH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqneg.v4i16(<4 x i16> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQNEGH_S16_I]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqnegh_s16(int16_t a) {
//   return (int16_t)vqnegh_s16(a);
// }

// NYI-LABEL: @test_vqnegs_s32(
// NYI:   [[VQNEGS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sqneg.i32(i32 %a)
// NYI:   ret i32 [[VQNEGS_S32_I]]
// int32_t test_vqnegs_s32(int32_t a) {
//   return (int32_t)vqnegs_s32(a);
// }

// NYI-LABEL: @test_vqnegd_s64(
// NYI:   [[VQNEGD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.sqneg.i64(i64 %a)
// NYI:   ret i64 [[VQNEGD_S64_I]]
// int64_t test_vqnegd_s64(int64_t a) {
//   return (int64_t)vqnegd_s64(a);
// }

// NYI-LABEL: @test_vuqaddb_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VUQADDB_S8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VUQADDB_S8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// int8_t test_vuqaddb_s8(int8_t a, uint8_t b) {
//   return (int8_t)vuqaddb_s8(a, b);
// }

// NYI-LABEL: @test_vuqaddh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VUQADDH_S16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VUQADDH_S16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// int16_t test_vuqaddh_s16(int16_t a, uint16_t b) {
//   return (int16_t)vuqaddh_s16(a, b);
// }

// NYI-LABEL: @test_vuqadds_s32(
// NYI:   [[VUQADDS_S32_I:%.*]] = call i32 @llvm.aarch64.neon.suqadd.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VUQADDS_S32_I]]
// int32_t test_vuqadds_s32(int32_t a, uint32_t b) {
//   return (int32_t)vuqadds_s32(a, b);
// }

// NYI-LABEL: @test_vuqaddd_s64(
// NYI:   [[VUQADDD_S64_I:%.*]] = call i64 @llvm.aarch64.neon.suqadd.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VUQADDD_S64_I]]
// int64_t test_vuqaddd_s64(int64_t a, uint64_t b) {
//   return (int64_t)vuqaddd_s64(a, b);
// }

// NYI-LABEL: @test_vsqaddb_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <8 x i8> poison, i8 %b, i64 0
// NYI:   [[VSQADDB_U8_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> [[TMP0]], <8 x i8> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <8 x i8> [[VSQADDB_U8_I]], i64 0
// NYI:   ret i8 [[TMP2]]
// uint8_t test_vsqaddb_u8(uint8_t a, int8_t b) {
//   return (uint8_t)vsqaddb_u8(a, b);
// }

// NYI-LABEL: @test_vsqaddh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VSQADDH_U16_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i16> [[VSQADDH_U16_I]], i64 0
// NYI:   ret i16 [[TMP2]]
// uint16_t test_vsqaddh_u16(uint16_t a, int16_t b) {
//   return (uint16_t)vsqaddh_u16(a, b);
// }

// NYI-LABEL: @test_vsqadds_u32(
// NYI:   [[VSQADDS_U32_I:%.*]] = call i32 @llvm.aarch64.neon.usqadd.i32(i32 %a, i32 %b)
// NYI:   ret i32 [[VSQADDS_U32_I]]
// uint32_t test_vsqadds_u32(uint32_t a, int32_t b) {
//   return (uint32_t)vsqadds_u32(a, b);
// }

// NYI-LABEL: @test_vsqaddd_u64(
// NYI:   [[VSQADDD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.usqadd.i64(i64 %a, i64 %b)
// NYI:   ret i64 [[VSQADDD_U64_I]]
// uint64_t test_vsqaddd_u64(uint64_t a, int64_t b) {
//   return (uint64_t)vsqaddd_u64(a, b);
// }

// NYI-LABEL: @test_vqdmlalh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %c, i64 0
// NYI:   [[VQDMLXL_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[LANE0_I:%.*]] = extractelement <4 x i32> [[VQDMLXL_I]], i64 0
// NYI:   [[VQDMLXL1_I:%.*]] = call i32 @llvm.aarch64.neon.sqadd.i32(i32 %a, i32 [[LANE0_I]])
// NYI:   ret i32 [[VQDMLXL1_I]]
// int32_t test_vqdmlalh_s16(int32_t a, int16_t b, int16_t c) {
//   return (int32_t)vqdmlalh_s16(a, b, c);
// }

// NYI-LABEL: @test_vqdmlals_s32(
// NYI:   [[VQDMLXL_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 %c)
// NYI:   [[VQDMLXL1_I:%.*]] = call i64 @llvm.aarch64.neon.sqadd.i64(i64 %a, i64 [[VQDMLXL_I]])
// NYI:   ret i64 [[VQDMLXL1_I]]
// int64_t test_vqdmlals_s32(int64_t a, int32_t b, int32_t c) {
//   return (int64_t)vqdmlals_s32(a, b, c);
// }

// NYI-LABEL: @test_vqdmlslh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %c, i64 0
// NYI:   [[VQDMLXL_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[LANE0_I:%.*]] = extractelement <4 x i32> [[VQDMLXL_I]], i64 0
// NYI:   [[VQDMLXL1_I:%.*]] = call i32 @llvm.aarch64.neon.sqsub.i32(i32 %a, i32 [[LANE0_I]])
// NYI:   ret i32 [[VQDMLXL1_I]]
// int32_t test_vqdmlslh_s16(int32_t a, int16_t b, int16_t c) {
//   return (int32_t)vqdmlslh_s16(a, b, c);
// }

// NYI-LABEL: @test_vqdmlsls_s32(
// NYI:   [[VQDMLXL_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %b, i32 %c)
// NYI:   [[VQDMLXL1_I:%.*]] = call i64 @llvm.aarch64.neon.sqsub.i64(i64 %a, i64 [[VQDMLXL_I]])
// NYI:   ret i64 [[VQDMLXL1_I]]
// int64_t test_vqdmlsls_s32(int64_t a, int32_t b, int32_t c) {
//   return (int64_t)vqdmlsls_s32(a, b, c);
// }

// NYI-LABEL: @test_vqdmullh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[TMP1:%.*]] = insertelement <4 x i16> poison, i16 %b, i64 0
// NYI:   [[VQDMULLH_S16_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqdmull.v4i32(<4 x i16> [[TMP0]], <4 x i16> [[TMP1]])
// NYI:   [[TMP2:%.*]] = extractelement <4 x i32> [[VQDMULLH_S16_I]], i64 0
// NYI:   ret i32 [[TMP2]]
// int32_t test_vqdmullh_s16(int16_t a, int16_t b) {
//   return (int32_t)vqdmullh_s16(a, b);
// }

// NYI-LABEL: @test_vqdmulls_s32(
// NYI:   [[VQDMULLS_S32_I:%.*]] = call i64 @llvm.aarch64.neon.sqdmulls.scalar(i32 %a, i32 %b)
// NYI:   ret i64 [[VQDMULLS_S32_I]]
// int64_t test_vqdmulls_s32(int32_t a, int32_t b) {
//   return (int64_t)vqdmulls_s32(a, b);
// }

// NYI-LABEL: @test_vqmovunh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQMOVUNH_S16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtun.v8i8(<8 x i16> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVUNH_S16_I]], i64 0
// NYI:   ret i8 [[TMP1]]
// uint8_t test_vqmovunh_s16(int16_t a) {
//   return (uint8_t)vqmovunh_s16(a);
// }

// NYI-LABEL: @test_vqmovuns_s32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQMOVUNS_S32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtun.v4i16(<4 x i32> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVUNS_S32_I]], i64 0
// NYI:   ret i16 [[TMP1]]
// uint16_t test_vqmovuns_s32(int32_t a) {
//   return (uint16_t)vqmovuns_s32(a);
// }

// NYI-LABEL: @test_vqmovund_s64(
// NYI:   [[VQMOVUND_S64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.sqxtun.i32.i64(i64 %a)
// NYI:   ret i32 [[VQMOVUND_S64_I]]
// uint32_t test_vqmovund_s64(int64_t a) {
//   return (uint32_t)vqmovund_s64(a);
// }

// NYI-LABEL: @test_vqmovnh_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQMOVNH_S16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqxtn.v8i8(<8 x i16> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVNH_S16_I]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqmovnh_s16(int16_t a) {
//   return (int8_t)vqmovnh_s16(a);
// }

int16_t test_vqmovns_s32(int32_t a) {
  return (int16_t)vqmovns_s32(a);

  // CIR-LABEL: vqmovns_s32
  // CIR: [[A:%.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
  // CIR: [[VQMOVNS_S32_ZERO1:%.*]] = cir.const #cir.int<0> : !u64i
  // CIR: [[POISON:%.*]] = cir.const #cir.poison : !s32i
  // CIR: [[POISON_VEC:%.*]] = cir.vec.splat [[POISON]] : !s32i, !cir.vector<!s32i x 4>
  // CIR: [[TMP0:%.*]] = cir.vec.insert [[A]], [[POISON_VEC]][[[VQMOVNS_S32_ZERO1]] : !u64i] : !cir.vector<!s32i x 4>
  // CIR: [[VQMOVNS_S32_I:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqxtn" [[TMP0]] : (!cir.vector<!s32i x 4>) -> !cir.vector<!s16i x 4>
  // CIR: [[VQMOVNS_S32_ZERO2:%.*]] = cir.const #cir.int<0> : !u64i
  // CIR: [[TMP1:%.*]] = cir.vec.extract [[VQMOVNS_S32_I]][[[VQMOVNS_S32_ZERO2]] : !u64i] : !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vqmovns_s32(i32{{.*}}[[a:%.*]])
  // LLVM:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 [[a]], i64 0
  // LLVM:   [[VQMOVNS_S32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqxtn.v4i16(<4 x i32> [[TMP0]])
  // LLVM:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVNS_S32_I]], i64 0
  // LLVM:   ret i16 [[TMP1]]
}

// NYI-LABEL: @test_vqmovnd_s64(
// NYI:   [[VQMOVND_S64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.sqxtn.i32.i64(i64 %a)
// NYI:   ret i32 [[VQMOVND_S64_I]]
// int32_t test_vqmovnd_s64(int64_t a) {
//   return (int32_t)vqmovnd_s64(a);
// }

// NYI-LABEL: @test_vqmovnh_u16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQMOVNH_U16_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqxtn.v8i8(<8 x i16> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQMOVNH_U16_I]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqmovnh_u16(int16_t a) {
//   return (int8_t)vqmovnh_u16(a);
// }

// NYI-LABEL: @test_vqmovns_u32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQMOVNS_U32_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqxtn.v4i16(<4 x i32> [[TMP0]])
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQMOVNS_U32_I]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqmovns_u32(int32_t a) {
//   return (int16_t)vqmovns_u32(a);
// }

// NYI-LABEL: @test_vqmovnd_u64(
// NYI:   [[VQMOVND_U64_I:%.*]] = call i32 @llvm.aarch64.neon.scalar.uqxtn.i32.i64(i64 %a)
// NYI:   ret i32 [[VQMOVND_U64_I]]
// int32_t test_vqmovnd_u64(int64_t a) {
//   return (int32_t)vqmovnd_u64(a);
// }

// NYI-LABEL: @test_vceqs_f32(
// NYI:   [[TMP0:%.*]] = fcmp oeq float %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCMPD_I]]
// uint32_t test_vceqs_f32(float32_t a, float32_t b) {
//   return (uint32_t)vceqs_f32(a, b);
// }

// NYI-LABEL: @test_vceqd_f64(
// NYI:   [[TMP0:%.*]] = fcmp oeq double %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCMPD_I]]
// uint64_t test_vceqd_f64(float64_t a, float64_t b) {
//   return (uint64_t)vceqd_f64(a, b);
// }

// NYI-LABEL: @test_vceqzs_f32(
// NYI:   [[TMP0:%.*]] = fcmp oeq float %a, 0.000000e+00
// NYI:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCEQZ_I]]
// uint32_t test_vceqzs_f32(float32_t a) {
//   return (uint32_t)vceqzs_f32(a);
// }

// NYI-LABEL: @test_vceqzd_f64(
// NYI:   [[TMP0:%.*]] = fcmp oeq double %a, 0.000000e+00
// NYI:   [[VCEQZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCEQZ_I]]
// uint64_t test_vceqzd_f64(float64_t a) {
//   return (uint64_t)vceqzd_f64(a);
// }

// NYI-LABEL: @test_vcges_f32(
// NYI:   [[TMP0:%.*]] = fcmp oge float %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCMPD_I]]
// uint32_t test_vcges_f32(float32_t a, float32_t b) {
//   return (uint32_t)vcges_f32(a, b);
// }

// NYI-LABEL: @test_vcged_f64(
// NYI:   [[TMP0:%.*]] = fcmp oge double %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCMPD_I]]
// uint64_t test_vcged_f64(float64_t a, float64_t b) {
//   return (uint64_t)vcged_f64(a, b);
// }

// NYI-LABEL: @test_vcgezs_f32(
// NYI:   [[TMP0:%.*]] = fcmp oge float %a, 0.000000e+00
// NYI:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCGEZ_I]]
// uint32_t test_vcgezs_f32(float32_t a) {
//   return (uint32_t)vcgezs_f32(a);
// }

// NYI-LABEL: @test_vcgezd_f64(
// NYI:   [[TMP0:%.*]] = fcmp oge double %a, 0.000000e+00
// NYI:   [[VCGEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCGEZ_I]]
// uint64_t test_vcgezd_f64(float64_t a) {
//   return (uint64_t)vcgezd_f64(a);
// }

// NYI-LABEL: @test_vcgts_f32(
// NYI:   [[TMP0:%.*]] = fcmp ogt float %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCMPD_I]]
// uint32_t test_vcgts_f32(float32_t a, float32_t b) {
//   return (uint32_t)vcgts_f32(a, b);
// }

// NYI-LABEL: @test_vcgtd_f64(
// NYI:   [[TMP0:%.*]] = fcmp ogt double %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCMPD_I]]
// uint64_t test_vcgtd_f64(float64_t a, float64_t b) {
//   return (uint64_t)vcgtd_f64(a, b);
// }

// NYI-LABEL: @test_vcgtzs_f32(
// NYI:   [[TMP0:%.*]] = fcmp ogt float %a, 0.000000e+00
// NYI:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCGTZ_I]]
// uint32_t test_vcgtzs_f32(float32_t a) {
//   return (uint32_t)vcgtzs_f32(a);
// }

// NYI-LABEL: @test_vcgtzd_f64(
// NYI:   [[TMP0:%.*]] = fcmp ogt double %a, 0.000000e+00
// NYI:   [[VCGTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCGTZ_I]]
// uint64_t test_vcgtzd_f64(float64_t a) {
//   return (uint64_t)vcgtzd_f64(a);
// }

// NYI-LABEL: @test_vcles_f32(
// NYI:   [[TMP0:%.*]] = fcmp ole float %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCMPD_I]]
// uint32_t test_vcles_f32(float32_t a, float32_t b) {
//   return (uint32_t)vcles_f32(a, b);
// }

// NYI-LABEL: @test_vcled_f64(
// NYI:   [[TMP0:%.*]] = fcmp ole double %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCMPD_I]]
// uint64_t test_vcled_f64(float64_t a, float64_t b) {
//   return (uint64_t)vcled_f64(a, b);
// }

// NYI-LABEL: @test_vclezs_f32(
// NYI:   [[TMP0:%.*]] = fcmp ole float %a, 0.000000e+00
// NYI:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCLEZ_I]]
// uint32_t test_vclezs_f32(float32_t a) {
//   return (uint32_t)vclezs_f32(a);
// }

// NYI-LABEL: @test_vclezd_f64(
// NYI:   [[TMP0:%.*]] = fcmp ole double %a, 0.000000e+00
// NYI:   [[VCLEZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCLEZ_I]]
// uint64_t test_vclezd_f64(float64_t a) {
//   return (uint64_t)vclezd_f64(a);
// }

// NYI-LABEL: @test_vclts_f32(
// NYI:   [[TMP0:%.*]] = fcmp olt float %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCMPD_I]]
// uint32_t test_vclts_f32(float32_t a, float32_t b) {
//   return (uint32_t)vclts_f32(a, b);
// }

// NYI-LABEL: @test_vcltd_f64(
// NYI:   [[TMP0:%.*]] = fcmp olt double %a, %b
// NYI:   [[VCMPD_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCMPD_I]]
// uint64_t test_vcltd_f64(float64_t a, float64_t b) {
//   return (uint64_t)vcltd_f64(a, b);
// }

// NYI-LABEL: @test_vcltzs_f32(
// NYI:   [[TMP0:%.*]] = fcmp olt float %a, 0.000000e+00
// NYI:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i32
// NYI:   ret i32 [[VCLTZ_I]]
// uint32_t test_vcltzs_f32(float32_t a) {
//   return (uint32_t)vcltzs_f32(a);
// }

// NYI-LABEL: @test_vcltzd_f64(
// NYI:   [[TMP0:%.*]] = fcmp olt double %a, 0.000000e+00
// NYI:   [[VCLTZ_I:%.*]] = sext i1 [[TMP0]] to i64
// NYI:   ret i64 [[VCLTZ_I]]
// uint64_t test_vcltzd_f64(float64_t a) {
//   return (uint64_t)vcltzd_f64(a);
// }

uint32_t test_vcages_f32(float32_t a, float32_t b) {
  return (uint32_t)vcages_f32(a, b);

  // CIR-LABEL: vcages_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facge" {{.*}}, {{.*}} : (!cir.float, !cir.float) -> !u32i

  // LLVM-LABEL: @test_vcages_f32(
  // LLVM:   [[VCAGED_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f32(float %0, float %1)
  // LLVM:   ret i32 [[VCAGED_F32_I]]
}

uint64_t test_vcaged_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaged_f64(a, b);

  // CIR-LABEL: vcaged_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facge" {{.*}}, {{.*}} : (!cir.double, !cir.double) -> !u64i

  // LLVM-LABEL: @test_vcaged_f64(
  // LLVM:   [[VCAGED_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facge.i64.f64(double %0, double %1)
  // LLVM:   ret i64 [[VCAGED_F64_I]]
}

uint32_t test_vcagts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcagts_f32(a, b);

  // CIR-LABEL: vcagts_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facgt" {{.*}}, {{.*}} : (!cir.float, !cir.float) -> !u32i

  // LLVM-LABEL: @test_vcagts_f32(
  // LLVM:   [[VCAGED_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %0, float %1)
  // LLVM:   ret i32 [[VCAGED_F32_I]]
}

uint64_t test_vcagtd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcagtd_f64(a, b);

  // CIR-LABEL: vcagtd_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facgt" {{.*}}, {{.*}} : (!cir.double, !cir.double) -> !u64i

  // LLVM-LABEL: @test_vcagtd_f64(
  // LLVM:   [[VCAGED_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facgt.i64.f64(double %0, double %1)
  // LLVM:   ret i64 [[VCAGED_F64_I]]
}

uint32_t test_vcales_f32(float32_t a, float32_t b) {
  return (uint32_t)vcales_f32(a, b);

  // CIR-LABEL: vcales_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facge" {{.*}}, {{.*}} : (!cir.float, !cir.float) -> !u32i

  // LLVM-LABEL: @test_vcales_f32(
  // LLVM:   [[VCALES_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facge.i32.f32(float %0, float %1)
  // LLVM:   ret i32 [[VCALES_F32_I]]
}

uint64_t test_vcaled_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaled_f64(a, b);

  // CIR-LABEL: vcaled_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facge" {{.*}}, {{.*}} : (!cir.double, !cir.double) -> !u64i

  // LLVM-LABEL: @test_vcaled_f64(
  // LLVM:   [[VCALED_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facge.i64.f64(double %0, double %1)
  // LLVM:   ret i64 [[VCALED_F64_I]]
}

// NYI-LABEL: @test_vcalts_f32(
// NYI:   [[VCALTS_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %b, float %a)
// NYI:   ret i32 [[VCALTS_F32_I]]
uint32_t test_vcalts_f32(float32_t a, float32_t b) {
  return (uint32_t)vcalts_f32(a, b);

  // CIR-LABEL: vcalts_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facgt" {{.*}}, {{.*}} : (!cir.float, !cir.float) -> !u32i

  // LLVM-LABEL: @test_vcalts_f32(
  // LLVM:   [[VCALTS_F32_I:%.*]] = call i32 @llvm.aarch64.neon.facgt.i32.f32(float %0, float %1)
  // LLVM:   ret i32 [[VCALTS_F32_I]]
}

uint64_t test_vcaltd_f64(float64_t a, float64_t b) {
  return (uint64_t)vcaltd_f64(a, b);

  // CIR-LABEL: vcaltd_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.facgt" {{.*}}, {{.*}} : (!cir.double, !cir.double) -> !u64i

  // LLVM-LABEL: @test_vcaltd_f64(
  // LLVM:   [[VCALTD_F64_I:%.*]] = call i64 @llvm.aarch64.neon.facgt.i64.f64(double %0, double %1)
  // LLVM:   ret i64 [[VCALTD_F64_I]]
}

int64_t test_vshrd_n_s64(int64_t a) {
  return (int64_t)vshrd_n_s64(a, 1);

  // CIR-LABEL: vshrd_n_s64
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !s64i, {{%.*}} : !s64i) -> !s64i

  // LLVM-LABEL: @test_vshrd_n_s64(
  // LLVM:   [[SHRD_N:%.*]] = ashr i64 %0, 1
  // LLVM:   ret i64 [[SHRD_N]]
}

uint64_t test_vshrd_n_u64(uint64_t a) {
   return (uint64_t)vshrd_n_u64(a, 64);

  // CIR-LABEL: vshrd_n_u64
  // CIR: {{.*}} = cir.const #cir.int<0> : !u64i
  // CIR: cir.return {{.*}} : !u64i

  // LLVM-LABEL: @test_vshrd_n_u64(
  // LLVM:   ret i64 0
}

uint64_t test_vshrd_n_u64_2() {
  uint64_t a = UINT64_C(0xf000000000000000);
  return vshrd_n_u64(a, 64);

  // CIR-LABEL: vshrd_n_u64
  // CIR: {{.*}} = cir.const #cir.int<0> : !u64i
  // CIR: cir.return {{.*}} : !u64i

  // LLVM-LABEL: @test_vshrd_n_u64_2(
  // LLVM:   ret i64 0

}

uint64_t test_vshrd_n_u64_3(uint64_t a) {
  return vshrd_n_u64(a, 1);

  // CIR-LABEL: vshrd_n_u64
  // CIR: {{%.*}} = cir.shift(right, {{%.*}} : !u64i, {{%.*}} : !u64i) -> !u64i

  // LLVM-LABEL: @test_vshrd_n_u64_3(
  // LLVM:   [[SHRD_N:%.*]] = lshr i64 %0, 1
  // LLVM:   ret i64 [[SHRD_N]]
}

int64_t test_vrshrd_n_s64(int64_t a) {
  return (int64_t)vrshrd_n_s64(a, 63);

  // CIR-LABEL: vrshrd_n_s64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" {{.*}}, {{.*}} : (!s64i, !s64i) -> !s64i

  // LLVM-LABEL: @test_vrshrd_n_s64(
  // LLVM:  [[VRSHR_N:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %0, i64 -63)
  // LLVM:  ret i64 [[VRSHR_N]]
}

// NYI-LABEL: @test_vrshr_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> <i64 -1>)
// NYI:   ret <1 x i64> [[VRSHR_N1]]
// int64x1_t test_vrshr_n_s64(int64x1_t a) {
//   return vrshr_n_s64(a, 1);
// }

uint64_t test_vrshrd_n_u64(uint64_t a) {
  return (uint64_t)vrshrd_n_u64(a, 63);

  // CIR-LABEL: vrshrd_n_u64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" {{.*}}, {{.*}} : (!u64i, !s64i) -> !u64i

  // LLVM-LABEL: @test_vrshrd_n_u64(
  // LLVM:   [[VRSHR_N:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %0, i64 -63)
  // LLVM:   ret i64 [[VRSHR_N]]
}

// NYI-LABEL: @test_vrshr_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> <i64 -1>)
// NYI:   ret <1 x i64> [[VRSHR_N1]]
// uint64x1_t test_vrshr_n_u64(uint64x1_t a) {
//   return vrshr_n_u64(a, 1);
// }


int64_t test_vsrad_n_s64(int64_t a, int64_t b) {
  return (int64_t)vsrad_n_s64(a, b, 63);

  // CIR-LABEL: vsrad_n_s64
  // CIR: [[ASHR:%.*]] = cir.shift(right, {{%.*}} : !s64i, {{%.*}} : !s64i) -> !s64i
  // CIR: {{.*}} = cir.binop(add, {{.*}}, [[ASHR]]) : !s64i

  // LLVM-LABEL: test_vsrad_n_s64(
  // LLVM: [[SHRD_N:%.*]] = ashr i64 %1, 63
  // LLVM: [[TMP0:%.*]] = add i64 %0, [[SHRD_N]]
  // LLVM: ret i64 [[TMP0]]
}

int64_t test_vsrad_n_s64_2(int64_t a, int64_t b) {
  return (int64_t)vsrad_n_s64(a, b, 64);

  // CIR-LABEL: vsrad_n_s64
  // CIR: [[ASHR:%.*]] = cir.shift(right, {{%.*}} : !s64i, {{%.*}} : !s64i) -> !s64i
  // CIR: {{.*}} = cir.binop(add, {{.*}}, [[ASHR]]) : !s64i

  // LLVM-LABEL: test_vsrad_n_s64_2(
  // LLVM: [[SHRD_N:%.*]] = ashr i64 %1, 63
  // LLVM: [[TMP0:%.*]] = add i64 %0, [[SHRD_N]]
  // LLVM: ret i64 [[TMP0]]
}

int64x1_t test_vsra_n_s64(int64x1_t a, int64x1_t b) {
  return vsra_n_s64(a, b, 1);

  // CIR-LABEL: vsra_n_s64
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!s64i x 1>

  // LLVM-LABEL: test_vsra_n_s64
  // LLVM: [[TMP0:%.*]] = bitcast <1 x i64> %0 to <8 x i8>
  // LLVM: [[TMP1:%.*]] = bitcast <1 x i64> %1 to <8 x i8>
  // LLVM: [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM: [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM: [[VSRA_N:%.*]] = ashr <1 x i64> [[TMP3]], splat (i64 1)
  // LLVM: [[TMP4:%.*]] = add <1 x i64> [[TMP2]], [[VSRA_N]]
  // LLVM: ret <1 x i64> [[TMP4]]
}

uint64_t test_vsrad_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vsrad_n_u64(a, b, 63);

  // CIR-LABEL:test_vsrad_n_u64
  // CIR: [[SHL:%.*]] = cir.shift(left, {{%.*}} : !u64i, {{%.*}} : !u64i) -> !u64i
  // CIR: {{.*}} = cir.binop(add, {{.*}}, [[SHL]]) : !u64i

  // LLVM-LABEL: test_vsrad_n_u64(
  // LLVM: [[SHRD_N:%.*]] = shl i64 %1, 63
  // LLVM: [[TMP0:%.*]] = add i64 %0, [[SHRD_N]]
  // LLVM: ret i64 [[TMP0]]
}

uint64_t test_vsrad_n_u64_2(uint64_t a, uint64_t b) {
  return (uint64_t)vsrad_n_u64(a, b, 64);

  // CIR-LABEL:test_vsrad_n_u64
  // CIR: cir.return {{.*}} : !u64i

  // LLVM-LABEL: test_vsrad_n_u64_2(
  // LLVM: ret i64 %0
}

uint64x1_t test_vsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vsra_n_u64(a, b, 1);

  // CIR-LABEL: vsra_n_u64
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VSRA_N:%.*]] = cir.shift(right, {{%.*}}, [[splat]] : !cir.vector<!u64i x 1>) -> !cir.vector<!u64i x 1>
  // CIR: cir.binop(add, {{%.*}}, [[VSRA_N]]) : !cir.vector<!u64i x 1>

  // LLVM-LABEL: test_vsra_n_u64
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> %1 to <8 x i8>
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM:   [[VSRA_N:%.*]] = lshr <1 x i64> [[TMP3]], splat (i64 1)
  // LLVM:   [[TMP4:%.*]] = add <1 x i64> [[TMP2]], [[VSRA_N]]
  // LLVM:   ret <1 x i64> [[TMP4]]
}

int64_t test_vrsrad_n_s64(int64_t a, int64_t b) {
  return (int64_t)vrsrad_n_s64(a, b, 63);

  // CIR-LABEL: vrsrad_n_s64
  // CIR: [[TMP0:%.*]] = cir.const #cir.int<63> : !s32i
  // CIR: [[TMP1:%.*]] = cir.unary(minus, [[TMP0]]) : !s32i, !s32i
  // CIR: [[TMP2:%.*]] = cir.cast(integral, [[TMP1]] : !s32i), !s64i
  // CIR: [[TMP3:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" {{.*}}, [[TMP2]] : (!s64i, !s64i) -> !s64i
  // CIR: [[TMP4:%.*]] = cir.binop(add, {{.*}}, [[TMP3]]) : !s64i

  // LLVM-LABEL: @test_vrsrad_n_s64(
  // LLVM: [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.srshl.i64(i64 %1, i64 -63)
  // LLVM: [[TMP1:%.*]] = add i64 %0, [[TMP0]]
  // LLVM: ret i64 [[TMP1]]
}

int64x1_t test_vrsra_n_s64(int64x1_t a, int64x1_t b) {
  return vrsra_n_s64(a, b, 1);

  // CIR-LABEL: vrsra_n_s64
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s64i x 1>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.srshl" {{%.*}}, [[splat]] : (!cir.vector<!s64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!s64i x 1>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!s64i x 1>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!s64i x 1>

  // LLVM-LABEL: test_vrsra_n_s64
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.srshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM:   [[TMP3:%.*]] = add <1 x i64> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <1 x i64> [[TMP3]]
}

uint64_t test_vrsrad_n_u64(uint64_t a, uint64_t b) {
  return (uint64_t)vrsrad_n_u64(a, b, 63);

  // CIR-LABEL:vrsrad_n_u64
  // CIR: [[TMP0:%.*]] = cir.const #cir.int<63> : !s32i
  // CIR: [[TMP1:%.*]] = cir.unary(minus, [[TMP0]]) : !s32i, !s32i
  // CIR: [[TMP2:%.*]] = cir.cast(integral, [[TMP1]] : !s32i), !u64i
  // CIR: [[TMP3:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" {{.*}}, [[TMP2]] : (!u64i, !u64i) -> !u64i
  // CIR: [[TMP4:%.*]] = cir.binop(add, {{.*}}, [[TMP3]]) : !u64i

  // LLVM-LABEL: @test_vrsrad_n_u64(
  // LLVM: [[TMP0:%.*]] = call i64 @llvm.aarch64.neon.urshl.i64(i64 %1, i64 -63)
  // LLVM: [[TMP1:%.*]] = add i64 %0, [[TMP0]]
  // LLVM: ret i64 [[TMP1]]
}

uint64x1_t test_vrsra_n_u64(uint64x1_t a, uint64x1_t b) {
  return vrsra_n_u64(a, b, 1);

  // CIR-LABEL: vrsra_n_u64
  // CIR: [[VRSHR_N:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u64i x 1>
  // CIR: [[splat:%.*]] = cir.const #cir.const_vector
  // CIR: [[VRSHR_N1:%.*]] = cir.llvm.intrinsic "aarch64.neon.urshl" [[VRSHR_N]], [[splat]] : (!cir.vector<!u64i x 1>, !cir.vector<!s64i x 1>) -> !cir.vector<!u64i x 1>
  // CIR: [[TMP2:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 8>), !cir.vector<!u64i x 1>
  // CIR: cir.binop(add, [[TMP2]], [[VRSHR_N1]]) : !cir.vector<!u64i x 1>

  // LLVM-LABEL: test_vrsra_n_u64
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x i64> %0 to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x i64> %1 to <8 x i8>
  // LLVM:   [[VRSHR_N:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
  // LLVM:   [[VRSHR_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.urshl.v1i64(<1 x i64> [[VRSHR_N]], <1 x i64> splat (i64 -1))
  // LLVM:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
  // LLVM:   [[TMP3:%.*]] = add <1 x i64> [[TMP2]], [[VRSHR_N1]]
  // LLVM:   ret <1 x i64> [[TMP3]]
}

int64_t test_vshld_n_s64(int64_t a) {
  return (int64_t)vshld_n_s64(a, 1);

  // CIR-LABEL: vshld_n_s64
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !s64i, {{%.*}} : !s64i) -> !s64i

  // LLVM-LABEL: @test_vshld_n_s64(
  // LLVM:   [[SHLD_N:%.*]] = shl i64 %0, 1
  // LLVM:   ret i64 [[SHLD_N]]
}

// NYI-LABEL: @test_vshl_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], <i64 1>
// NYI:   ret <1 x i64> [[VSHL_N]]
// int64x1_t test_vshl_n_s64(int64x1_t a) {
//   return vshl_n_s64(a, 1);
// }

uint64_t test_vshld_n_u64(uint64_t a) {
  return (uint64_t)vshld_n_u64(a, 63);

  // CIR-LABEL: vshld_n_u64
  // CIR: {{%.*}} = cir.shift(left, {{%.*}} : !u64i, {{%.*}} : !u64i) -> !u64i

  // LLVM-LABEL: @test_vshld_n_u64(
  // LLVM:   [[SHLD_N:%.*]] = shl i64 %0, 63
  // LLVM:   ret i64 [[SHLD_N]]
}

// NYI-LABEL: @test_vshl_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSHL_N:%.*]] = shl <1 x i64> [[TMP1]], <i64 1>
// NYI:   ret <1 x i64> [[VSHL_N]]
// uint64x1_t test_vshl_n_u64(uint64x1_t a) {
//   return vshl_n_u64(a, 1);
// }

// NYI-LABEL: @test_vqshlb_n_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[VQSHLB_N_S8:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLB_N_S8]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqshlb_n_s8(int8_t a) {
//   return (int8_t)vqshlb_n_s8(a, 7);
// }

// NYI-LABEL: @test_vqshlh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHLH_N_S16:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLH_N_S16]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqshlh_n_s16(int16_t a) {
//   return (int16_t)vqshlh_n_s16(a, 15);
// }

// NYI-LABEL: @test_vqshls_n_s32(
// NYI:   [[VQSHLS_N_S32:%.*]] = call i32 @llvm.aarch64.neon.sqshl.i32(i32 %a, i32 31)
// NYI:   ret i32 [[VQSHLS_N_S32]]
// int32_t test_vqshls_n_s32(int32_t a) {
//   return (int32_t)vqshls_n_s32(a, 31);
// }

int64_t test_vqshld_n_s64(int64_t a) {
 return (int64_t)vqshld_n_s64(a, 63);

 // CIR-LABEL: vqshld_n_s64
 // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqshl" {{.*}}, {{.*}} : (!s64i, !s64i) -> !s64i

 // LLVM-LABEL: @test_vqshld_n_s64(
 // LLVM: [[VQSHL_N:%.*]] = call i64 @llvm.aarch64.neon.sqshl.i64(i64 %0, i64 63)
 // LLVM: ret i64 [[VQSHL_N]]
}

// NYI-LABEL: @test_vqshl_n_s8(
// NYI:   [[VQSHL_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshl.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
// NYI:   ret <8 x i8> [[VQSHL_N]]
// int8x8_t test_vqshl_n_s8(int8x8_t a) {
//   return vqshl_n_s8(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_s8(
// NYI:   [[VQSHL_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.sqshl.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
// NYI:   ret <16 x i8> [[VQSHL_N]]
// int8x16_t test_vqshlq_n_s8(int8x16_t a) {
//   return vqshlq_n_s8(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VQSHL_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshl.v4i16(<4 x i16> [[VQSHL_N]], <4 x i16> zeroinitializer)
// NYI:   ret <4 x i16> [[VQSHL_N1]]
// int16x4_t test_vqshl_n_s16(int16x4_t a) {
//   return vqshl_n_s16(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHL_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.sqshl.v8i16(<8 x i16> [[VQSHL_N]], <8 x i16> zeroinitializer)
// NYI:   ret <8 x i16> [[VQSHL_N1]]
// int16x8_t test_vqshlq_n_s16(int16x8_t a) {
//   return vqshlq_n_s16(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VQSHL_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.sqshl.v2i32(<2 x i32> [[VQSHL_N]], <2 x i32> zeroinitializer)
// NYI:   ret <2 x i32> [[VQSHL_N1]]
// int32x2_t test_vqshl_n_s32(int32x2_t a) {
//   return vqshl_n_s32(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHL_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.sqshl.v4i32(<4 x i32> [[VQSHL_N]], <4 x i32> zeroinitializer)
// NYI:   ret <4 x i32> [[VQSHL_N1]]
// int32x4_t test_vqshlq_n_s32(int32x4_t a) {
//   return vqshlq_n_s32(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHL_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.sqshl.v2i64(<2 x i64> [[VQSHL_N]], <2 x i64> zeroinitializer)
// NYI:   ret <2 x i64> [[VQSHL_N1]]
// int64x2_t test_vqshlq_n_s64(int64x2_t a) {
//   return vqshlq_n_s64(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_u8(
// NYI:   [[VQSHL_N:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> %a, <8 x i8> zeroinitializer)
// NYI:   ret <8 x i8> [[VQSHL_N]]
// uint8x8_t test_vqshl_n_u8(uint8x8_t a) {
//   return vqshl_n_u8(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_u8(
// NYI:   [[VQSHL_N:%.*]] = call <16 x i8> @llvm.aarch64.neon.uqshl.v16i8(<16 x i8> %a, <16 x i8> zeroinitializer)
// NYI:   ret <16 x i8> [[VQSHL_N]]
// uint8x16_t test_vqshlq_n_u8(uint8x16_t a) {
//   return vqshlq_n_u8(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <4 x i16>
// NYI:   [[VQSHL_N1:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[VQSHL_N]], <4 x i16> zeroinitializer)
// NYI:   ret <4 x i16> [[VQSHL_N1]]
// uint16x4_t test_vqshl_n_u16(uint16x4_t a) {
//   return vqshl_n_u16(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <8 x i16>
// NYI:   [[VQSHL_N1:%.*]] = call <8 x i16> @llvm.aarch64.neon.uqshl.v8i16(<8 x i16> [[VQSHL_N]], <8 x i16> zeroinitializer)
// NYI:   ret <8 x i16> [[VQSHL_N1]]
// uint16x8_t test_vqshlq_n_u16(uint16x8_t a) {
//   return vqshlq_n_u16(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <2 x i32>
// NYI:   [[VQSHL_N1:%.*]] = call <2 x i32> @llvm.aarch64.neon.uqshl.v2i32(<2 x i32> [[VQSHL_N]], <2 x i32> zeroinitializer)
// NYI:   ret <2 x i32> [[VQSHL_N1]]
// uint32x2_t test_vqshl_n_u32(uint32x2_t a) {
//   return vqshl_n_u32(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <4 x i32>
// NYI:   [[VQSHL_N1:%.*]] = call <4 x i32> @llvm.aarch64.neon.uqshl.v4i32(<4 x i32> [[VQSHL_N]], <4 x i32> zeroinitializer)
// NYI:   ret <4 x i32> [[VQSHL_N1]]
// uint32x4_t test_vqshlq_n_u32(uint32x4_t a) {
//   return vqshlq_n_u32(a, 0);
// }

// NYI-LABEL: @test_vqshlq_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// NYI:   [[VQSHL_N1:%.*]] = call <2 x i64> @llvm.aarch64.neon.uqshl.v2i64(<2 x i64> [[VQSHL_N]], <2 x i64> zeroinitializer)
// NYI:   ret <2 x i64> [[VQSHL_N1]]
// uint64x2_t test_vqshlq_n_u64(uint64x2_t a) {
//   return vqshlq_n_u64(a, 0);
// }

// NYI-LABEL: @test_vqshl_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VQSHL_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshl.v1i64(<1 x i64> [[VQSHL_N]], <1 x i64> <i64 1>)
// NYI:   ret <1 x i64> [[VQSHL_N1]]
// int64x1_t test_vqshl_n_s64(int64x1_t a) {
//   return vqshl_n_s64(a, 1);
// }

// NYI-LABEL: @test_vqshlb_n_u8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[VQSHLB_N_U8:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshl.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLB_N_U8]], i64 0
// NYI:   ret i8 [[TMP1]]
// uint8_t test_vqshlb_n_u8(uint8_t a) {
//   return (uint8_t)vqshlb_n_u8(a, 7);
// }

// NYI-LABEL: @test_vqshlh_n_u16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHLH_N_U16:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshl.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLH_N_U16]], i64 0
// NYI:   ret i16 [[TMP1]]
// uint16_t test_vqshlh_n_u16(uint16_t a) {
//   return (uint16_t)vqshlh_n_u16(a, 15);
// }

// NYI-LABEL: @test_vqshls_n_u32(
// NYI:   [[VQSHLS_N_U32:%.*]] = call i32 @llvm.aarch64.neon.uqshl.i32(i32 %a, i32 31)
// NYI:   ret i32 [[VQSHLS_N_U32]]
// uint32_t test_vqshls_n_u32(uint32_t a) {
//   return (uint32_t)vqshls_n_u32(a, 31);
// }

uint64_t test_vqshld_n_u64(uint64_t a) {
 return (uint64_t)vqshld_n_u64(a, 63);

 // CIR-LABEL: vqshld_n_u64
 // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.uqshl" {{.*}}, {{.*}} : (!u64i, !u64i) -> !u64i

 // LLVM-LABEL: @test_vqshld_n_u64(
 // LLVM: [[VQSHL_N:%.*]] = call i64 @llvm.aarch64.neon.uqshl.i64(i64 %0, i64 63)
 // LLVM: ret i64 [[VQSHL_N]]
}

// NYI-LABEL: @test_vqshl_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VQSHL_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VQSHL_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.uqshl.v1i64(<1 x i64> [[VQSHL_N]], <1 x i64> <i64 1>)
// NYI:   ret <1 x i64> [[VQSHL_N1]]
// uint64x1_t test_vqshl_n_u64(uint64x1_t a) {
//   return vqshl_n_u64(a, 1);
// }

// NYI-LABEL: @test_vqshlub_n_s8(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i8> poison, i8 %a, i64 0
// NYI:   [[VQSHLUB_N_S8:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshlu.v8i8(<8 x i8> [[TMP0]], <8 x i8> <i8 7, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison, i8 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHLUB_N_S8]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqshlub_n_s8(int8_t a) {
//   return (int8_t)vqshlub_n_s8(a, 7);
// }

// NYI-LABEL: @test_vqshluh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHLUH_N_S16:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshlu.v4i16(<4 x i16> [[TMP0]], <4 x i16> <i16 15, i16 poison, i16 poison, i16 poison>)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHLUH_N_S16]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqshluh_n_s16(int16_t a) {
//   return (int16_t)vqshluh_n_s16(a, 15);
// }

// NYI-LABEL: @test_vqshlus_n_s32(
// NYI:   [[VQSHLUS_N_S32:%.*]] = call i32 @llvm.aarch64.neon.sqshlu.i32(i32 %a, i32 31)
// NYI:   ret i32 [[VQSHLUS_N_S32]]
// int32_t test_vqshlus_n_s32(int32_t a) {
//   return (int32_t)vqshlus_n_s32(a, 31);
// }

int64_t test_vqshlud_n_s64(int64_t a) {
  return (int64_t)vqshlud_n_s64(a, 63);

  // CIR-LABEL: vqshlud_n_s64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.neon.sqshlu" {{.*}}, {{.*}} : (!s64i, !s64i) -> !s64i

  // LLVM-LABEL: @test_vqshlud_n_s64(
  // LLVM: [[VQSHLU_N:%.*]] = call i64 @llvm.aarch64.neon.sqshlu.i64(i64 %0, i64 63)
  // LLVM: ret i64 [[VQSHLU_N]]
}

// NYI-LABEL: @test_vqshlu_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VQSHLU_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VQSHLU_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqshlu.v1i64(<1 x i64> [[VQSHLU_N]], <1 x i64> <i64 1>)
// NYI:   ret <1 x i64> [[VQSHLU_N1]]
// uint64x1_t test_vqshlu_n_s64(int64x1_t a) {
//   return vqshlu_n_s64(a, 1);
// }

// NYI-LABEL: @test_vsrid_n_s64(
// NYI:   [[VSRID_N_S64:%.*]] = bitcast i64 %a to <1 x i64>
// NYI:   [[VSRID_N_S641:%.*]] = bitcast i64 %b to <1 x i64>
// NYI:   [[VSRID_N_S642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRID_N_S64]], <1 x i64> [[VSRID_N_S641]], i32 63)
// NYI:   [[VSRID_N_S643:%.*]] = bitcast <1 x i64> [[VSRID_N_S642]] to i64
// NYI:   ret i64 [[VSRID_N_S643]]
// int64_t test_vsrid_n_s64(int64_t a, int64_t b) {
//   return (int64_t)vsrid_n_s64(a, b, 63);
// }

// NYI-LABEL: @test_vsri_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   [[VSRI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRI_N]], <1 x i64> [[VSRI_N1]], i32 1)
// NYI:   ret <1 x i64> [[VSRI_N2]]
// int64x1_t test_vsri_n_s64(int64x1_t a, int64x1_t b) {
//   return vsri_n_s64(a, b, 1);
// }

// NYI-LABEL: @test_vsrid_n_u64(
// NYI:   [[VSRID_N_U64:%.*]] = bitcast i64 %a to <1 x i64>
// NYI:   [[VSRID_N_U641:%.*]] = bitcast i64 %b to <1 x i64>
// NYI:   [[VSRID_N_U642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRID_N_U64]], <1 x i64> [[VSRID_N_U641]], i32 63)
// NYI:   [[VSRID_N_U643:%.*]] = bitcast <1 x i64> [[VSRID_N_U642]] to i64
// NYI:   ret i64 [[VSRID_N_U643]]
// uint64_t test_vsrid_n_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vsrid_n_u64(a, b, 63);
// }

// NYI-LABEL: @test_vsri_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   [[VSRI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRI_N]], <1 x i64> [[VSRI_N1]], i32 1)
// NYI:   ret <1 x i64> [[VSRI_N2]]
// uint64x1_t test_vsri_n_u64(uint64x1_t a, uint64x1_t b) {
//   return vsri_n_u64(a, b, 1);
// }

// NYI-LABEL: @test_vslid_n_s64(
// NYI:   [[VSLID_N_S64:%.*]] = bitcast i64 %a to <1 x i64>
// NYI:   [[VSLID_N_S641:%.*]] = bitcast i64 %b to <1 x i64>
// NYI:   [[VSLID_N_S642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLID_N_S64]], <1 x i64> [[VSLID_N_S641]], i32 63)
// NYI:   [[VSLID_N_S643:%.*]] = bitcast <1 x i64> [[VSLID_N_S642]] to i64
// NYI:   ret i64 [[VSLID_N_S643]]
// int64_t test_vslid_n_s64(int64_t a, int64_t b) {
//   return (int64_t)vslid_n_s64(a, b, 63);
// }

// NYI-LABEL: @test_vsli_n_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 1)
// NYI:   ret <1 x i64> [[VSLI_N2]]
// int64x1_t test_vsli_n_s64(int64x1_t a, int64x1_t b) {
//   return vsli_n_s64(a, b, 1);
// }

// NYI-LABEL: @test_vslid_n_u64(
// NYI:   [[VSLID_N_U64:%.*]] = bitcast i64 %a to <1 x i64>
// NYI:   [[VSLID_N_U641:%.*]] = bitcast i64 %b to <1 x i64>
// NYI:   [[VSLID_N_U642:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLID_N_U64]], <1 x i64> [[VSLID_N_U641]], i32 63)
// NYI:   [[VSLID_N_U643:%.*]] = bitcast <1 x i64> [[VSLID_N_U642]] to i64
// NYI:   ret i64 [[VSLID_N_U643]]
// uint64_t test_vslid_n_u64(uint64_t a, uint64_t b) {
//   return (uint64_t)vslid_n_u64(a, b, 63);
// }

// NYI-LABEL: @test_vsli_n_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSLI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VSLI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// NYI:   [[VSLI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsli.v1i64(<1 x i64> [[VSLI_N]], <1 x i64> [[VSLI_N1]], i32 1)
// NYI:   ret <1 x i64> [[VSLI_N2]]
// uint64x1_t test_vsli_n_u64(uint64x1_t a, uint64x1_t b) {
//   return vsli_n_u64(a, b, 1);
// }

// NYI-LABEL: @test_vqshrnh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHRNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRNH_N_S16]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqshrnh_n_s16(int16_t a) {
//   return (int8_t)vqshrnh_n_s16(a, 8);
// }

// NYI-LABEL: @test_vqshrns_n_s32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQSHRNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRNS_N_S32]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqshrns_n_s32(int32_t a) {
//   return (int16_t)vqshrns_n_s32(a, 16);
// }

// NYI-LABEL: @test_vqshrnd_n_s64(
// NYI:   [[VQSHRND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqshrn.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQSHRND_N_S64]]
// int32_t test_vqshrnd_n_s64(int64_t a) {
//   return (int32_t)vqshrnd_n_s64(a, 32);
// }

// NYI-LABEL: @test_vqshrnh_n_u16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHRNH_N_U16:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRNH_N_U16]], i64 0
// NYI:   ret i8 [[TMP1]]
// uint8_t test_vqshrnh_n_u16(uint16_t a) {
//   return (uint8_t)vqshrnh_n_u16(a, 8);
// }

// NYI-LABEL: @test_vqshrns_n_u32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQSHRNS_N_U32:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRNS_N_U32]], i64 0
// NYI:   ret i16 [[TMP1]]
// uint16_t test_vqshrns_n_u32(uint32_t a) {
//   return (uint16_t)vqshrns_n_u32(a, 16);
// }

// NYI-LABEL: @test_vqshrnd_n_u64(
// NYI:   [[VQSHRND_N_U64:%.*]] = call i32 @llvm.aarch64.neon.uqshrn.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQSHRND_N_U64]]
// uint32_t test_vqshrnd_n_u64(uint64_t a) {
//   return (uint32_t)vqshrnd_n_u64(a, 32);
// }

// NYI-LABEL: @test_vqrshrnh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQRSHRNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRNH_N_S16]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqrshrnh_n_s16(int16_t a) {
//   return (int8_t)vqrshrnh_n_s16(a, 8);
// }

// NYI-LABEL: @test_vqrshrns_n_s32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQRSHRNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRNS_N_S32]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqrshrns_n_s32(int32_t a) {
//   return (int16_t)vqrshrns_n_s32(a, 16);
// }

// NYI-LABEL: @test_vqrshrnd_n_s64(
// NYI:   [[VQRSHRND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqrshrn.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQRSHRND_N_S64]]
// int32_t test_vqrshrnd_n_s64(int64_t a) {
//   return (int32_t)vqrshrnd_n_s64(a, 32);
// }

// NYI-LABEL: @test_vqrshrnh_n_u16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQRSHRNH_N_U16:%.*]] = call <8 x i8> @llvm.aarch64.neon.uqrshrn.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRNH_N_U16]], i64 0
// NYI:   ret i8 [[TMP1]]
// uint8_t test_vqrshrnh_n_u16(uint16_t a) {
//   return (uint8_t)vqrshrnh_n_u16(a, 8);
// }

// NYI-LABEL: @test_vqrshrns_n_u32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQRSHRNS_N_U32:%.*]] = call <4 x i16> @llvm.aarch64.neon.uqrshrn.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRNS_N_U32]], i64 0
// NYI:   ret i16 [[TMP1]]
// uint16_t test_vqrshrns_n_u32(uint32_t a) {
//   return (uint16_t)vqrshrns_n_u32(a, 16);
// }

// NYI-LABEL: @test_vqrshrnd_n_u64(
// NYI:   [[VQRSHRND_N_U64:%.*]] = call i32 @llvm.aarch64.neon.uqrshrn.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQRSHRND_N_U64]]
// uint32_t test_vqrshrnd_n_u64(uint64_t a) {
//   return (uint32_t)vqrshrnd_n_u64(a, 32);
// }

// NYI-LABEL: @test_vqshrunh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQSHRUNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqshrun.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQSHRUNH_N_S16]], i64 0
// NYI:   ret i8 [[TMP1]]
// int8_t test_vqshrunh_n_s16(int16_t a) {
//   return (int8_t)vqshrunh_n_s16(a, 8);
// }

// NYI-LABEL: @test_vqshruns_n_s32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQSHRUNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqshrun.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQSHRUNS_N_S32]], i64 0
// NYI:   ret i16 [[TMP1]]
// int16_t test_vqshruns_n_s32(int32_t a) {
//   return (int16_t)vqshruns_n_s32(a, 16);
// }

// NYI-LABEL: @test_vqshrund_n_s64(
// NYI:   [[VQSHRUND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqshrun.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQSHRUND_N_S64]]
// int32_t test_vqshrund_n_s64(int64_t a) {
//   return (int32_t)vqshrund_n_s64(a, 32);
// }

// NYI-LABEL: @test_vqrshrunh_n_s16(
// NYI:   [[TMP0:%.*]] = insertelement <8 x i16> poison, i16 %a, i64 0
// NYI:   [[VQRSHRUNH_N_S16:%.*]] = call <8 x i8> @llvm.aarch64.neon.sqrshrun.v8i8(<8 x i16> [[TMP0]], i32 8)
// NYI:   [[TMP1:%.*]] = extractelement <8 x i8> [[VQRSHRUNH_N_S16]], i64 0
// NYI:   ret i8 [[TMP1]]
// uint8_t test_vqrshrunh_n_s16(int16_t a) {
//   return (uint8_t)vqrshrunh_n_s16(a, 8);
// }

// NYI-LABEL: @test_vqrshruns_n_s32(
// NYI:   [[TMP0:%.*]] = insertelement <4 x i32> poison, i32 %a, i64 0
// NYI:   [[VQRSHRUNS_N_S32:%.*]] = call <4 x i16> @llvm.aarch64.neon.sqrshrun.v4i16(<4 x i32> [[TMP0]], i32 16)
// NYI:   [[TMP1:%.*]] = extractelement <4 x i16> [[VQRSHRUNS_N_S32]], i64 0
// NYI:   ret i16 [[TMP1]]
// uint16_t test_vqrshruns_n_s32(int32_t a) {
//   return (uint16_t)vqrshruns_n_s32(a, 16);
// }

// NYI-LABEL: @test_vqrshrund_n_s64(
// NYI:   [[VQRSHRUND_N_S64:%.*]] = call i32 @llvm.aarch64.neon.sqrshrun.i32(i64 %a, i32 32)
// NYI:   ret i32 [[VQRSHRUND_N_S64]]
// uint32_t test_vqrshrund_n_s64(int64_t a) {
//   return (uint32_t)vqrshrund_n_s64(a, 32);
// }

// NYI-LABEL: @test_vcvts_n_f32_s32(
// NYI:   [[VCVTS_N_F32_S32:%.*]] = call float @llvm.aarch64.neon.vcvtfxs2fp.f32.i32(i32 %a, i32 1)
// NYI:   ret float [[VCVTS_N_F32_S32]]
// float32_t test_vcvts_n_f32_s32(int32_t a) {
//   return vcvts_n_f32_s32(a, 1);
// }

// NYI-LABEL: @test_vcvtd_n_f64_s64(
// NYI:   [[VCVTD_N_F64_S64:%.*]] = call double @llvm.aarch64.neon.vcvtfxs2fp.f64.i64(i64 %a, i32 1)
// NYI:   ret double [[VCVTD_N_F64_S64]]
// float64_t test_vcvtd_n_f64_s64(int64_t a) {
//   return vcvtd_n_f64_s64(a, 1);
// }

// NYI-LABEL: @test_vcvts_n_f32_u32(
// NYI:   [[VCVTS_N_F32_U32:%.*]] = call float @llvm.aarch64.neon.vcvtfxu2fp.f32.i32(i32 %a, i32 32)
// NYI:   ret float [[VCVTS_N_F32_U32]]
// float32_t test_vcvts_n_f32_u32(uint32_t a) {
//   return vcvts_n_f32_u32(a, 32);
// }

// NYI-LABEL: @test_vcvtd_n_f64_u64(
// NYI:   [[VCVTD_N_F64_U64:%.*]] = call double @llvm.aarch64.neon.vcvtfxu2fp.f64.i64(i64 %a, i32 64)
// NYI:   ret double [[VCVTD_N_F64_U64]]
// float64_t test_vcvtd_n_f64_u64(uint64_t a) {
//   return vcvtd_n_f64_u64(a, 64);
// }

// NYI-LABEL: @test_vcvts_n_s32_f32(
// NYI:   [[VCVTS_N_S32_F32:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxs.i32.f32(float %a, i32 1)
// NYI:   ret i32 [[VCVTS_N_S32_F32]]
// int32_t test_vcvts_n_s32_f32(float32_t a) {
//   return (int32_t)vcvts_n_s32_f32(a, 1);
// }

// NYI-LABEL: @test_vcvtd_n_s64_f64(
// NYI:   [[VCVTD_N_S64_F64:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxs.i64.f64(double %a, i32 1)
// NYI:   ret i64 [[VCVTD_N_S64_F64]]
// int64_t test_vcvtd_n_s64_f64(float64_t a) {
//   return (int64_t)vcvtd_n_s64_f64(a, 1);
// }

// NYI-LABEL: @test_vcvts_n_u32_f32(
// NYI:   [[VCVTS_N_U32_F32:%.*]] = call i32 @llvm.aarch64.neon.vcvtfp2fxu.i32.f32(float %a, i32 32)
// NYI:   ret i32 [[VCVTS_N_U32_F32]]
// uint32_t test_vcvts_n_u32_f32(float32_t a) {
//   return (uint32_t)vcvts_n_u32_f32(a, 32);
// }

// NYI-LABEL: @test_vcvtd_n_u64_f64(
// NYI:   [[VCVTD_N_U64_F64:%.*]] = call i64 @llvm.aarch64.neon.vcvtfp2fxu.i64.f64(double %a, i32 64)
// NYI:   ret i64 [[VCVTD_N_U64_F64]]
// uint64_t test_vcvtd_n_u64_f64(float64_t a) {
//   return (uint64_t)vcvtd_n_u64_f64(a, 64);
// }

// NYI-LABEL: @test_vreinterpret_s8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_s16(int16x4_t a) {
//   return vreinterpret_s8_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_s32(int32x2_t a) {
//   return vreinterpret_s8_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_s64(int64x1_t a) {
//   return vreinterpret_s8_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_u8(
// NYI:   ret <8 x i8> %a
// int8x8_t test_vreinterpret_s8_u8(uint8x8_t a) {
//   return vreinterpret_s8_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_u16(uint16x4_t a) {
//   return vreinterpret_s8_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_u32(uint32x2_t a) {
//   return vreinterpret_s8_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_u64(uint64x1_t a) {
//   return vreinterpret_s8_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_f16(float16x4_t a) {
//   return vreinterpret_s8_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_f32(float32x2_t a) {
//   return vreinterpret_s8_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_f64(float64x1_t a) {
//   return vreinterpret_s8_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_p8(
// NYI:   ret <8 x i8> %a
// int8x8_t test_vreinterpret_s8_p8(poly8x8_t a) {
//   return vreinterpret_s8_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_p16(poly16x4_t a) {
//   return vreinterpret_s8_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_s8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// int8x8_t test_vreinterpret_s8_p64(poly64x1_t a) {
//   return vreinterpret_s8_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_s8(int8x8_t a) {
//   return vreinterpret_s16_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_s32(int32x2_t a) {
//   return vreinterpret_s16_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_s64(int64x1_t a) {
//   return vreinterpret_s16_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_u8(uint8x8_t a) {
//   return vreinterpret_s16_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_u16(
// NYI:   ret <4 x i16> %a
// int16x4_t test_vreinterpret_s16_u16(uint16x4_t a) {
//   return vreinterpret_s16_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_u32(uint32x2_t a) {
//   return vreinterpret_s16_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_u64(uint64x1_t a) {
//   return vreinterpret_s16_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_f16(float16x4_t a) {
//   return vreinterpret_s16_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_f32(float32x2_t a) {
//   return vreinterpret_s16_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_f64(float64x1_t a) {
//   return vreinterpret_s16_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_p8(poly8x8_t a) {
//   return vreinterpret_s16_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_p16(
// NYI:   ret <4 x i16> %a
// int16x4_t test_vreinterpret_s16_p16(poly16x4_t a) {
//   return vreinterpret_s16_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_s16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// int16x4_t test_vreinterpret_s16_p64(poly64x1_t a) {
//   return vreinterpret_s16_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_s8(int8x8_t a) {
//   return vreinterpret_s32_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_s16(int16x4_t a) {
//   return vreinterpret_s32_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_s64(int64x1_t a) {
//   return vreinterpret_s32_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_u8(uint8x8_t a) {
//   return vreinterpret_s32_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_u16(uint16x4_t a) {
//   return vreinterpret_s32_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_u32(
// NYI:   ret <2 x i32> %a
// int32x2_t test_vreinterpret_s32_u32(uint32x2_t a) {
//   return vreinterpret_s32_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_u64(uint64x1_t a) {
//   return vreinterpret_s32_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_f16(float16x4_t a) {
//   return vreinterpret_s32_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_f32(float32x2_t a) {
//   return vreinterpret_s32_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_f64(float64x1_t a) {
//   return vreinterpret_s32_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_p8(poly8x8_t a) {
//   return vreinterpret_s32_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_p16(poly16x4_t a) {
//   return vreinterpret_s32_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_s32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// int32x2_t test_vreinterpret_s32_p64(poly64x1_t a) {
//   return vreinterpret_s32_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_s8(int8x8_t a) {
//   return vreinterpret_s64_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_s16(int16x4_t a) {
//   return vreinterpret_s64_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_s32(int32x2_t a) {
//   return vreinterpret_s64_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_u8(uint8x8_t a) {
//   return vreinterpret_s64_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_u16(uint16x4_t a) {
//   return vreinterpret_s64_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_u32(uint32x2_t a) {
//   return vreinterpret_s64_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_u64(
// NYI:   ret <1 x i64> %a
// int64x1_t test_vreinterpret_s64_u64(uint64x1_t a) {
//   return vreinterpret_s64_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_f16(float16x4_t a) {
//   return vreinterpret_s64_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_f32(float32x2_t a) {
//   return vreinterpret_s64_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_f64(float64x1_t a) {
//   return vreinterpret_s64_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_p8(poly8x8_t a) {
//   return vreinterpret_s64_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// int64x1_t test_vreinterpret_s64_p16(poly16x4_t a) {
//   return vreinterpret_s64_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_s64_p64(
// NYI:   ret <1 x i64> %a
// int64x1_t test_vreinterpret_s64_p64(poly64x1_t a) {
//   return vreinterpret_s64_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_s8(
// NYI:   ret <8 x i8> %a
// uint8x8_t test_vreinterpret_u8_s8(int8x8_t a) {
//   return vreinterpret_u8_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_s16(int16x4_t a) {
//   return vreinterpret_u8_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_s32(int32x2_t a) {
//   return vreinterpret_u8_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_s64(int64x1_t a) {
//   return vreinterpret_u8_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_u16(uint16x4_t a) {
//   return vreinterpret_u8_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_u32(uint32x2_t a) {
//   return vreinterpret_u8_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_u64(uint64x1_t a) {
//   return vreinterpret_u8_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_f16(float16x4_t a) {
//   return vreinterpret_u8_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_f32(float32x2_t a) {
//   return vreinterpret_u8_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_f64(float64x1_t a) {
//   return vreinterpret_u8_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_p8(
// NYI:   ret <8 x i8> %a
// uint8x8_t test_vreinterpret_u8_p8(poly8x8_t a) {
//   return vreinterpret_u8_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_p16(poly16x4_t a) {
//   return vreinterpret_u8_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_u8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// uint8x8_t test_vreinterpret_u8_p64(poly64x1_t a) {
//   return vreinterpret_u8_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_s8(int8x8_t a) {
//   return vreinterpret_u16_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_s16(
// NYI:   ret <4 x i16> %a
// uint16x4_t test_vreinterpret_u16_s16(int16x4_t a) {
//   return vreinterpret_u16_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_s32(int32x2_t a) {
//   return vreinterpret_u16_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_s64(int64x1_t a) {
//   return vreinterpret_u16_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_u8(uint8x8_t a) {
//   return vreinterpret_u16_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_u32(uint32x2_t a) {
//   return vreinterpret_u16_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_u64(uint64x1_t a) {
//   return vreinterpret_u16_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_f16(float16x4_t a) {
//   return vreinterpret_u16_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_f32(float32x2_t a) {
//   return vreinterpret_u16_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_f64(float64x1_t a) {
//   return vreinterpret_u16_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_p8(poly8x8_t a) {
//   return vreinterpret_u16_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_p16(
// NYI:   ret <4 x i16> %a
// uint16x4_t test_vreinterpret_u16_p16(poly16x4_t a) {
//   return vreinterpret_u16_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_u16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// uint16x4_t test_vreinterpret_u16_p64(poly64x1_t a) {
//   return vreinterpret_u16_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_s8(int8x8_t a) {
//   return vreinterpret_u32_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_s16(int16x4_t a) {
//   return vreinterpret_u32_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_s32(
// NYI:   ret <2 x i32> %a
// uint32x2_t test_vreinterpret_u32_s32(int32x2_t a) {
//   return vreinterpret_u32_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_s64(int64x1_t a) {
//   return vreinterpret_u32_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_u8(uint8x8_t a) {
//   return vreinterpret_u32_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_u16(uint16x4_t a) {
//   return vreinterpret_u32_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_u64(uint64x1_t a) {
//   return vreinterpret_u32_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_f16(float16x4_t a) {
//   return vreinterpret_u32_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_f32(float32x2_t a) {
//   return vreinterpret_u32_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_f64(float64x1_t a) {
//   return vreinterpret_u32_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_p8(poly8x8_t a) {
//   return vreinterpret_u32_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_p16(poly16x4_t a) {
//   return vreinterpret_u32_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_u32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x i32>
// NYI:   ret <2 x i32> [[TMP0]]
// uint32x2_t test_vreinterpret_u32_p64(poly64x1_t a) {
//   return vreinterpret_u32_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_s8(int8x8_t a) {
//   return vreinterpret_u64_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_s16(int16x4_t a) {
//   return vreinterpret_u64_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_s32(int32x2_t a) {
//   return vreinterpret_u64_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_s64(
// NYI:   ret <1 x i64> %a
// uint64x1_t test_vreinterpret_u64_s64(int64x1_t a) {
//   return vreinterpret_u64_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_u8(uint8x8_t a) {
//   return vreinterpret_u64_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_u16(uint16x4_t a) {
//   return vreinterpret_u64_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_u32(uint32x2_t a) {
//   return vreinterpret_u64_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_f16(float16x4_t a) {
//   return vreinterpret_u64_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_f32(float32x2_t a) {
//   return vreinterpret_u64_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_f64(float64x1_t a) {
//   return vreinterpret_u64_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_p8(poly8x8_t a) {
//   return vreinterpret_u64_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// uint64x1_t test_vreinterpret_u64_p16(poly16x4_t a) {
//   return vreinterpret_u64_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_u64_p64(
// NYI:   ret <1 x i64> %a
// uint64x1_t test_vreinterpret_u64_p64(poly64x1_t a) {
//   return vreinterpret_u64_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_s8(int8x8_t a) {
//   return vreinterpret_f16_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_s16(int16x4_t a) {
//   return vreinterpret_f16_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_s32(int32x2_t a) {
//   return vreinterpret_f16_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_s64(int64x1_t a) {
//   return vreinterpret_f16_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_u8(uint8x8_t a) {
//   return vreinterpret_f16_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_u16(uint16x4_t a) {
//   return vreinterpret_f16_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_u32(uint32x2_t a) {
//   return vreinterpret_f16_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_u64(uint64x1_t a) {
//   return vreinterpret_f16_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_f32(float32x2_t a) {
//   return vreinterpret_f16_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_f64(float64x1_t a) {
//   return vreinterpret_f16_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_p8(poly8x8_t a) {
//   return vreinterpret_f16_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_p16(poly16x4_t a) {
//   return vreinterpret_f16_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_f16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x half>
// NYI:   ret <4 x half> [[TMP0]]
// float16x4_t test_vreinterpret_f16_p64(poly64x1_t a) {
//   return vreinterpret_f16_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_s8(int8x8_t a) {
//   return vreinterpret_f32_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_s16(int16x4_t a) {
//   return vreinterpret_f32_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_s32(int32x2_t a) {
//   return vreinterpret_f32_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_s64(int64x1_t a) {
//   return vreinterpret_f32_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_u8(uint8x8_t a) {
//   return vreinterpret_f32_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_u16(uint16x4_t a) {
//   return vreinterpret_f32_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_u32(uint32x2_t a) {
//   return vreinterpret_f32_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_u64(uint64x1_t a) {
//   return vreinterpret_f32_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_f16(float16x4_t a) {
//   return vreinterpret_f32_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_f64(float64x1_t a) {
//   return vreinterpret_f32_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_p8(poly8x8_t a) {
//   return vreinterpret_f32_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_p16(poly16x4_t a) {
//   return vreinterpret_f32_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_f32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <2 x float>
// NYI:   ret <2 x float> [[TMP0]]
// float32x2_t test_vreinterpret_f32_p64(poly64x1_t a) {
//   return vreinterpret_f32_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_s8(int8x8_t a) {
//   return vreinterpret_f64_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_s16(int16x4_t a) {
//   return vreinterpret_f64_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_s32(int32x2_t a) {
//   return vreinterpret_f64_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_s64(int64x1_t a) {
//   return vreinterpret_f64_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_u8(uint8x8_t a) {
//   return vreinterpret_f64_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_u16(uint16x4_t a) {
//   return vreinterpret_f64_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_u32(uint32x2_t a) {
//   return vreinterpret_f64_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_u64(uint64x1_t a) {
//   return vreinterpret_f64_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_f16(float16x4_t a) {
//   return vreinterpret_f64_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_f32(float32x2_t a) {
//   return vreinterpret_f64_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_p8(poly8x8_t a) {
//   return vreinterpret_f64_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_p16(poly16x4_t a) {
//   return vreinterpret_f64_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_f64_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <1 x double>
// NYI:   ret <1 x double> [[TMP0]]
// float64x1_t test_vreinterpret_f64_p64(poly64x1_t a) {
//   return vreinterpret_f64_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_s8(
// NYI:   ret <8 x i8> %a
// poly8x8_t test_vreinterpret_p8_s8(int8x8_t a) {
//   return vreinterpret_p8_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_s16(int16x4_t a) {
//   return vreinterpret_p8_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_s32(int32x2_t a) {
//   return vreinterpret_p8_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_s64(int64x1_t a) {
//   return vreinterpret_p8_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_u8(
// NYI:   ret <8 x i8> %a
// poly8x8_t test_vreinterpret_p8_u8(uint8x8_t a) {
//   return vreinterpret_p8_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_u16(uint16x4_t a) {
//   return vreinterpret_p8_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_u32(uint32x2_t a) {
//   return vreinterpret_p8_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_u64(uint64x1_t a) {
//   return vreinterpret_p8_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_f16(float16x4_t a) {
//   return vreinterpret_p8_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_f32(float32x2_t a) {
//   return vreinterpret_p8_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_f64(float64x1_t a) {
//   return vreinterpret_p8_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_p16(poly16x4_t a) {
//   return vreinterpret_p8_p16(a);
// }

// NYI-LABEL: @test_vreinterpret_p8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   ret <8 x i8> [[TMP0]]
// poly8x8_t test_vreinterpret_p8_p64(poly64x1_t a) {
//   return vreinterpret_p8_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_s8(int8x8_t a) {
//   return vreinterpret_p16_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_s16(
// NYI:   ret <4 x i16> %a
// poly16x4_t test_vreinterpret_p16_s16(int16x4_t a) {
//   return vreinterpret_p16_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_s32(int32x2_t a) {
//   return vreinterpret_p16_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_s64(int64x1_t a) {
//   return vreinterpret_p16_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_u8(uint8x8_t a) {
//   return vreinterpret_p16_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_u16(
// NYI:   ret <4 x i16> %a
// poly16x4_t test_vreinterpret_p16_u16(uint16x4_t a) {
//   return vreinterpret_p16_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_u32(uint32x2_t a) {
//   return vreinterpret_p16_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_u64(uint64x1_t a) {
//   return vreinterpret_p16_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_f16(float16x4_t a) {
//   return vreinterpret_p16_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_f32(float32x2_t a) {
//   return vreinterpret_p16_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_f64(float64x1_t a) {
//   return vreinterpret_p16_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_p8(poly8x8_t a) {
//   return vreinterpret_p16_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_p16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <4 x i16>
// NYI:   ret <4 x i16> [[TMP0]]
// poly16x4_t test_vreinterpret_p16_p64(poly64x1_t a) {
//   return vreinterpret_p16_p64(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_s8(int8x8_t a) {
//   return vreinterpret_p64_s8(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_s16(int16x4_t a) {
//   return vreinterpret_p64_s16(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_s32(int32x2_t a) {
//   return vreinterpret_p64_s32(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_s64(
// NYI:   ret <1 x i64> %a
// poly64x1_t test_vreinterpret_p64_s64(int64x1_t a) {
//   return vreinterpret_p64_s64(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_u8(uint8x8_t a) {
//   return vreinterpret_p64_u8(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_u16(uint16x4_t a) {
//   return vreinterpret_p64_u16(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_u32(uint32x2_t a) {
//   return vreinterpret_p64_u32(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_u64(
// NYI:   ret <1 x i64> %a
// poly64x1_t test_vreinterpret_p64_u64(uint64x1_t a) {
//   return vreinterpret_p64_u64(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x half> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_f16(float16x4_t a) {
//   return vreinterpret_p64_f16(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x float> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_f32(float32x2_t a) {
//   return vreinterpret_p64_f32(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_f64(float64x1_t a) {
//   return vreinterpret_p64_f64(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i8> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_p8(poly8x8_t a) {
//   return vreinterpret_p64_p8(a);
// }

// NYI-LABEL: @test_vreinterpret_p64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <1 x i64>
// NYI:   ret <1 x i64> [[TMP0]]
// poly64x1_t test_vreinterpret_p64_p16(poly16x4_t a) {
//   return vreinterpret_p64_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_s16(int16x8_t a) {
//   return vreinterpretq_s8_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_s32(int32x4_t a) {
//   return vreinterpretq_s8_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_s64(int64x2_t a) {
//   return vreinterpretq_s8_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_u8(
// NYI:   ret <16 x i8> %a
// int8x16_t test_vreinterpretq_s8_u8(uint8x16_t a) {
//   return vreinterpretq_s8_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_u16(uint16x8_t a) {
//   return vreinterpretq_s8_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_u32(uint32x4_t a) {
//   return vreinterpretq_s8_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_u64(uint64x2_t a) {
//   return vreinterpretq_s8_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_f16(float16x8_t a) {
//   return vreinterpretq_s8_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_f32(float32x4_t a) {
//   return vreinterpretq_s8_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_f64(float64x2_t a) {
//   return vreinterpretq_s8_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_p8(
// NYI:   ret <16 x i8> %a
// int8x16_t test_vreinterpretq_s8_p8(poly8x16_t a) {
//   return vreinterpretq_s8_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_p16(poly16x8_t a) {
//   return vreinterpretq_s8_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// int8x16_t test_vreinterpretq_s8_p64(poly64x2_t a) {
//   return vreinterpretq_s8_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_s8(int8x16_t a) {
//   return vreinterpretq_s16_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_s32(int32x4_t a) {
//   return vreinterpretq_s16_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_s64(int64x2_t a) {
//   return vreinterpretq_s16_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_u8(uint8x16_t a) {
//   return vreinterpretq_s16_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_u16(
// NYI:   ret <8 x i16> %a
// int16x8_t test_vreinterpretq_s16_u16(uint16x8_t a) {
//   return vreinterpretq_s16_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_u32(uint32x4_t a) {
//   return vreinterpretq_s16_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_u64(uint64x2_t a) {
//   return vreinterpretq_s16_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_f16(float16x8_t a) {
//   return vreinterpretq_s16_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_f32(float32x4_t a) {
//   return vreinterpretq_s16_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_f64(float64x2_t a) {
//   return vreinterpretq_s16_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_p8(poly8x16_t a) {
//   return vreinterpretq_s16_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_p16(
// NYI:   ret <8 x i16> %a
// int16x8_t test_vreinterpretq_s16_p16(poly16x8_t a) {
//   return vreinterpretq_s16_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// int16x8_t test_vreinterpretq_s16_p64(poly64x2_t a) {
//   return vreinterpretq_s16_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_s8(int8x16_t a) {
//   return vreinterpretq_s32_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_s16(int16x8_t a) {
//   return vreinterpretq_s32_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_s64(int64x2_t a) {
//   return vreinterpretq_s32_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_u8(uint8x16_t a) {
//   return vreinterpretq_s32_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_u16(uint16x8_t a) {
//   return vreinterpretq_s32_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_u32(
// NYI:   ret <4 x i32> %a
// int32x4_t test_vreinterpretq_s32_u32(uint32x4_t a) {
//   return vreinterpretq_s32_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_u64(uint64x2_t a) {
//   return vreinterpretq_s32_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_f16(float16x8_t a) {
//   return vreinterpretq_s32_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_f32(float32x4_t a) {
//   return vreinterpretq_s32_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_f64(float64x2_t a) {
//   return vreinterpretq_s32_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_p8(poly8x16_t a) {
//   return vreinterpretq_s32_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_p16(poly16x8_t a) {
//   return vreinterpretq_s32_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// int32x4_t test_vreinterpretq_s32_p64(poly64x2_t a) {
//   return vreinterpretq_s32_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_s8(int8x16_t a) {
//   return vreinterpretq_s64_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_s16(int16x8_t a) {
//   return vreinterpretq_s64_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_s32(int32x4_t a) {
//   return vreinterpretq_s64_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_u8(uint8x16_t a) {
//   return vreinterpretq_s64_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_u16(uint16x8_t a) {
//   return vreinterpretq_s64_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_u32(uint32x4_t a) {
//   return vreinterpretq_s64_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_u64(
// NYI:   ret <2 x i64> %a
// int64x2_t test_vreinterpretq_s64_u64(uint64x2_t a) {
//   return vreinterpretq_s64_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_f16(float16x8_t a) {
//   return vreinterpretq_s64_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_f32(float32x4_t a) {
//   return vreinterpretq_s64_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_f64(float64x2_t a) {
//   return vreinterpretq_s64_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_p8(poly8x16_t a) {
//   return vreinterpretq_s64_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// int64x2_t test_vreinterpretq_s64_p16(poly16x8_t a) {
//   return vreinterpretq_s64_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_s64_p64(
// NYI:   ret <2 x i64> %a
// int64x2_t test_vreinterpretq_s64_p64(poly64x2_t a) {
//   return vreinterpretq_s64_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_s8(
// NYI:   ret <16 x i8> %a
// uint8x16_t test_vreinterpretq_u8_s8(int8x16_t a) {
//   return vreinterpretq_u8_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_s16(int16x8_t a) {
//   return vreinterpretq_u8_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_s32(int32x4_t a) {
//   return vreinterpretq_u8_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_s64(int64x2_t a) {
//   return vreinterpretq_u8_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_u16(uint16x8_t a) {
//   return vreinterpretq_u8_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_u32(uint32x4_t a) {
//   return vreinterpretq_u8_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_u64(uint64x2_t a) {
//   return vreinterpretq_u8_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_f16(float16x8_t a) {
//   return vreinterpretq_u8_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_f32(float32x4_t a) {
//   return vreinterpretq_u8_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_f64(float64x2_t a) {
//   return vreinterpretq_u8_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_p8(
// NYI:   ret <16 x i8> %a
// uint8x16_t test_vreinterpretq_u8_p8(poly8x16_t a) {
//   return vreinterpretq_u8_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_p16(poly16x8_t a) {
//   return vreinterpretq_u8_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// uint8x16_t test_vreinterpretq_u8_p64(poly64x2_t a) {
//   return vreinterpretq_u8_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_s8(int8x16_t a) {
//   return vreinterpretq_u16_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_s16(
// NYI:   ret <8 x i16> %a
// uint16x8_t test_vreinterpretq_u16_s16(int16x8_t a) {
//   return vreinterpretq_u16_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_s32(int32x4_t a) {
//   return vreinterpretq_u16_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_s64(int64x2_t a) {
//   return vreinterpretq_u16_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_u8(uint8x16_t a) {
//   return vreinterpretq_u16_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_u32(uint32x4_t a) {
//   return vreinterpretq_u16_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_u64(uint64x2_t a) {
//   return vreinterpretq_u16_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_f16(float16x8_t a) {
//   return vreinterpretq_u16_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_f32(float32x4_t a) {
//   return vreinterpretq_u16_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_f64(float64x2_t a) {
//   return vreinterpretq_u16_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_p8(poly8x16_t a) {
//   return vreinterpretq_u16_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_p16(
// NYI:   ret <8 x i16> %a
// uint16x8_t test_vreinterpretq_u16_p16(poly16x8_t a) {
//   return vreinterpretq_u16_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// uint16x8_t test_vreinterpretq_u16_p64(poly64x2_t a) {
//   return vreinterpretq_u16_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_s8(int8x16_t a) {
//   return vreinterpretq_u32_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_s16(int16x8_t a) {
//   return vreinterpretq_u32_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_s32(
// NYI:   ret <4 x i32> %a
// uint32x4_t test_vreinterpretq_u32_s32(int32x4_t a) {
//   return vreinterpretq_u32_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_s64(int64x2_t a) {
//   return vreinterpretq_u32_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_u8(uint8x16_t a) {
//   return vreinterpretq_u32_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_u16(uint16x8_t a) {
//   return vreinterpretq_u32_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_u64(uint64x2_t a) {
//   return vreinterpretq_u32_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_f16(float16x8_t a) {
//   return vreinterpretq_u32_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_f32(float32x4_t a) {
//   return vreinterpretq_u32_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_f64(float64x2_t a) {
//   return vreinterpretq_u32_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_p8(poly8x16_t a) {
//   return vreinterpretq_u32_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_p16(poly16x8_t a) {
//   return vreinterpretq_u32_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x i32>
// NYI:   ret <4 x i32> [[TMP0]]
// uint32x4_t test_vreinterpretq_u32_p64(poly64x2_t a) {
//   return vreinterpretq_u32_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_s8(int8x16_t a) {
//   return vreinterpretq_u64_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_s16(int16x8_t a) {
//   return vreinterpretq_u64_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_s32(int32x4_t a) {
//   return vreinterpretq_u64_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_s64(
// NYI:   ret <2 x i64> %a
// uint64x2_t test_vreinterpretq_u64_s64(int64x2_t a) {
//   return vreinterpretq_u64_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_u8(uint8x16_t a) {
//   return vreinterpretq_u64_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_u16(uint16x8_t a) {
//   return vreinterpretq_u64_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_u32(uint32x4_t a) {
//   return vreinterpretq_u64_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_f16(float16x8_t a) {
//   return vreinterpretq_u64_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_f32(float32x4_t a) {
//   return vreinterpretq_u64_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_f64(float64x2_t a) {
//   return vreinterpretq_u64_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_p8(poly8x16_t a) {
//   return vreinterpretq_u64_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// uint64x2_t test_vreinterpretq_u64_p16(poly16x8_t a) {
//   return vreinterpretq_u64_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_u64_p64(
// NYI:   ret <2 x i64> %a
// uint64x2_t test_vreinterpretq_u64_p64(poly64x2_t a) {
//   return vreinterpretq_u64_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_s8(int8x16_t a) {
//   return vreinterpretq_f16_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_s16(int16x8_t a) {
//   return vreinterpretq_f16_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_s32(int32x4_t a) {
//   return vreinterpretq_f16_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_s64(int64x2_t a) {
//   return vreinterpretq_f16_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_u8(uint8x16_t a) {
//   return vreinterpretq_f16_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_u16(uint16x8_t a) {
//   return vreinterpretq_f16_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_u32(uint32x4_t a) {
//   return vreinterpretq_f16_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_u64(uint64x2_t a) {
//   return vreinterpretq_f16_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_f32(float32x4_t a) {
//   return vreinterpretq_f16_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_f64(float64x2_t a) {
//   return vreinterpretq_f16_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_p8(poly8x16_t a) {
//   return vreinterpretq_f16_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_p16(poly16x8_t a) {
//   return vreinterpretq_f16_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x half>
// NYI:   ret <8 x half> [[TMP0]]
// float16x8_t test_vreinterpretq_f16_p64(poly64x2_t a) {
//   return vreinterpretq_f16_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_s8(int8x16_t a) {
//   return vreinterpretq_f32_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_s16(int16x8_t a) {
//   return vreinterpretq_f32_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_s32(int32x4_t a) {
//   return vreinterpretq_f32_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_s64(int64x2_t a) {
//   return vreinterpretq_f32_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_u8(uint8x16_t a) {
//   return vreinterpretq_f32_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_u16(uint16x8_t a) {
//   return vreinterpretq_f32_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_u32(uint32x4_t a) {
//   return vreinterpretq_f32_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_u64(uint64x2_t a) {
//   return vreinterpretq_f32_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_f16(float16x8_t a) {
//   return vreinterpretq_f32_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_f64(float64x2_t a) {
//   return vreinterpretq_f32_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_p8(poly8x16_t a) {
//   return vreinterpretq_f32_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_p16(poly16x8_t a) {
//   return vreinterpretq_f32_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f32_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <4 x float>
// NYI:   ret <4 x float> [[TMP0]]
// float32x4_t test_vreinterpretq_f32_p64(poly64x2_t a) {
//   return vreinterpretq_f32_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_s8(int8x16_t a) {
//   return vreinterpretq_f64_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_s16(int16x8_t a) {
//   return vreinterpretq_f64_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_s32(int32x4_t a) {
//   return vreinterpretq_f64_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_s64(int64x2_t a) {
//   return vreinterpretq_f64_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_u8(uint8x16_t a) {
//   return vreinterpretq_f64_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_u16(uint16x8_t a) {
//   return vreinterpretq_f64_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_u32(uint32x4_t a) {
//   return vreinterpretq_f64_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_u64(uint64x2_t a) {
//   return vreinterpretq_f64_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_f16(float16x8_t a) {
//   return vreinterpretq_f64_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_f32(float32x4_t a) {
//   return vreinterpretq_f64_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_p8(poly8x16_t a) {
//   return vreinterpretq_f64_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_p16(poly16x8_t a) {
//   return vreinterpretq_f64_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_f64_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <2 x double>
// NYI:   ret <2 x double> [[TMP0]]
// float64x2_t test_vreinterpretq_f64_p64(poly64x2_t a) {
//   return vreinterpretq_f64_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_s8(
// NYI:   ret <16 x i8> %a
// poly8x16_t test_vreinterpretq_p8_s8(int8x16_t a) {
//   return vreinterpretq_p8_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_s16(int16x8_t a) {
//   return vreinterpretq_p8_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_s32(int32x4_t a) {
//   return vreinterpretq_p8_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_s64(int64x2_t a) {
//   return vreinterpretq_p8_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_u8(
// NYI:   ret <16 x i8> %a
// poly8x16_t test_vreinterpretq_p8_u8(uint8x16_t a) {
//   return vreinterpretq_p8_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_u16(uint16x8_t a) {
//   return vreinterpretq_p8_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_u32(uint32x4_t a) {
//   return vreinterpretq_p8_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_u64(uint64x2_t a) {
//   return vreinterpretq_p8_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_f16(float16x8_t a) {
//   return vreinterpretq_p8_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_f32(float32x4_t a) {
//   return vreinterpretq_p8_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_f64(float64x2_t a) {
//   return vreinterpretq_p8_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_p16(poly16x8_t a) {
//   return vreinterpretq_p8_p16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p8_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   ret <16 x i8> [[TMP0]]
// poly8x16_t test_vreinterpretq_p8_p64(poly64x2_t a) {
//   return vreinterpretq_p8_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_s8(int8x16_t a) {
//   return vreinterpretq_p16_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_s16(
// NYI:   ret <8 x i16> %a
// poly16x8_t test_vreinterpretq_p16_s16(int16x8_t a) {
//   return vreinterpretq_p16_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_s32(int32x4_t a) {
//   return vreinterpretq_p16_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_s64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_s64(int64x2_t a) {
//   return vreinterpretq_p16_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_u8(uint8x16_t a) {
//   return vreinterpretq_p16_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_u16(
// NYI:   ret <8 x i16> %a
// poly16x8_t test_vreinterpretq_p16_u16(uint16x8_t a) {
//   return vreinterpretq_p16_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_u32(uint32x4_t a) {
//   return vreinterpretq_p16_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_u64(uint64x2_t a) {
//   return vreinterpretq_p16_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_f16(float16x8_t a) {
//   return vreinterpretq_p16_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_f32(float32x4_t a) {
//   return vreinterpretq_p16_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_f64(float64x2_t a) {
//   return vreinterpretq_p16_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_p8(poly8x16_t a) {
//   return vreinterpretq_p16_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p16_p64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <8 x i16>
// NYI:   ret <8 x i16> [[TMP0]]
// poly16x8_t test_vreinterpretq_p16_p64(poly64x2_t a) {
//   return vreinterpretq_p16_p64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_s8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_s8(int8x16_t a) {
//   return vreinterpretq_p64_s8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_s16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_s16(int16x8_t a) {
//   return vreinterpretq_p64_s16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_s32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_s32(int32x4_t a) {
//   return vreinterpretq_p64_s32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_s64(
// NYI:   ret <2 x i64> %a
// poly64x2_t test_vreinterpretq_p64_s64(int64x2_t a) {
//   return vreinterpretq_p64_s64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_u8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_u8(uint8x16_t a) {
//   return vreinterpretq_p64_u8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_u16(uint16x8_t a) {
//   return vreinterpretq_p64_u16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_u32(uint32x4_t a) {
//   return vreinterpretq_p64_u32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_u64(
// NYI:   ret <2 x i64> %a
// poly64x2_t test_vreinterpretq_p64_u64(uint64x2_t a) {
//   return vreinterpretq_p64_u64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_f16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x half> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_f16(float16x8_t a) {
//   return vreinterpretq_p64_f16(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_f32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x float> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_f32(float32x4_t a) {
//   return vreinterpretq_p64_f32(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x double> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_f64(float64x2_t a) {
//   return vreinterpretq_p64_f64(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_p8(
// NYI:   [[TMP0:%.*]] = bitcast <16 x i8> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_p8(poly8x16_t a) {
//   return vreinterpretq_p64_p8(a);
// }

// NYI-LABEL: @test_vreinterpretq_p64_p16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <2 x i64>
// NYI:   ret <2 x i64> [[TMP0]]
// poly64x2_t test_vreinterpretq_p64_p16(poly16x8_t a) {
//   return vreinterpretq_p64_p16(a);
// }

float32_t test_vabds_f32(float32_t a, float32_t b) {
  return vabds_f32(a, b);

  // CIR-LABEL: vabds_f32
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.sisd.fabd" {{.*}}, {{.*}} : (!cir.float, !cir.float) -> !cir.float

  // LLVM-LABEL: @test_vabds_f32(
  // LLVM:   [[VABDS_F32:%.*]] = call float @llvm.aarch64.sisd.fabd.f32(float %0, float %1)
  // LLVM:   ret float [[VABDS_F32]]
}

float64_t test_vabdd_f64(float64_t a, float64_t b) {
  return vabdd_f64(a, b);

  // CIR-LABEL: vabdd_f64
  // CIR: [[TMP0:%.*]] = cir.llvm.intrinsic "aarch64.sisd.fabd" {{.*}}, {{.*}} : (!cir.double, !cir.double) -> !cir.double

  // LLVM-LABEL: @test_vabdd_f64(
  // LLVM:   [[VABDD_F64:%.*]] = call double @llvm.aarch64.sisd.fabd.f64(double %0, double %1)
  // LLVM:   ret double [[VABDD_F64]]
}

// NYI-LABEL: @test_vuqaddq_s8(
// NYI: entry:
// NYI-NEXT:  [[V:%.*]] = call <16 x i8> @llvm.aarch64.neon.suqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI-NEXT:  ret <16 x i8> [[V]]
// int8x16_t test_vuqaddq_s8(int8x16_t a, uint8x16_t b) {
//   return vuqaddq_s8(a, b);
// }

// NYI-LABEL: @test_vuqaddq_s32(
// NYI: [[V:%.*]] = call <4 x i32> @llvm.aarch64.neon.suqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI-NEXT:  ret <4 x i32> [[V]]
// int32x4_t test_vuqaddq_s32(int32x4_t a, uint32x4_t b) {
//   return vuqaddq_s32(a, b);
// }

// NYI-LABEL: @test_vuqaddq_s64(
// NYI: [[V:%.*]] = call <2 x i64> @llvm.aarch64.neon.suqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI-NEXT:  ret <2 x i64> [[V]]
// int64x2_t test_vuqaddq_s64(int64x2_t a, uint64x2_t b) {
//   return vuqaddq_s64(a, b);
// }

// NYI-LABEL: @test_vuqaddq_s16(
// NYI: [[V:%.*]] = call <8 x i16> @llvm.aarch64.neon.suqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI-NEXT:  ret <8 x i16> [[V]]
// int16x8_t test_vuqaddq_s16(int16x8_t a, uint16x8_t b) {
//   return vuqaddq_s16(a, b);
// }

// NYI-LABEL: @test_vuqadd_s8(
// NYI: entry:
// NYI-NEXT: [[V:%.*]] = call <8 x i8> @llvm.aarch64.neon.suqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI-NEXT: ret <8 x i8> [[V]]
// int8x8_t test_vuqadd_s8(int8x8_t a, uint8x8_t b) {
//   return vuqadd_s8(a, b);
// }

// NYI-LABEL: @test_vuqadd_s32(
// NYI: [[V:%.*]] = call <2 x i32> @llvm.aarch64.neon.suqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI-NEXT:  ret <2 x i32> [[V]]
// int32x2_t test_vuqadd_s32(int32x2_t a, uint32x2_t b) {
//   return vuqadd_s32(a, b);
// }

// NYI-LABEL: @test_vuqadd_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VUQADD2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.suqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   ret <1 x i64> [[VUQADD2_I]]
// int64x1_t test_vuqadd_s64(int64x1_t a, uint64x1_t b) {
//   return vuqadd_s64(a, b);
// }

// NYI-LABEL: @test_vuqadd_s16(
// NYI: [[V:%.*]] = call <4 x i16> @llvm.aarch64.neon.suqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI-NEXT:  ret <4 x i16> [[V]]
// int16x4_t test_vuqadd_s16(int16x4_t a, uint16x4_t b) {
//   return vuqadd_s16(a, b);
// }

// NYI-LABEL: @test_vsqadd_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.usqadd.v1i64(<1 x i64> %a, <1 x i64> %b)
// NYI:   ret <1 x i64> [[VSQADD2_I]]
// uint64x1_t test_vsqadd_u64(uint64x1_t a, int64x1_t b) {
//   return vsqadd_u64(a, b);
// }

// NYI-LABEL: @test_vsqadd_u8(
// NYI:   [[VSQADD_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.usqadd.v8i8(<8 x i8> %a, <8 x i8> %b)
// NYI:   ret <8 x i8> [[VSQADD_I]]
// uint8x8_t test_vsqadd_u8(uint8x8_t a, int8x8_t b) {
//   return vsqadd_u8(a, b);
// }

// NYI-LABEL: @test_vsqaddq_u8(
// NYI:   [[VSQADD_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.usqadd.v16i8(<16 x i8> %a, <16 x i8> %b)
// NYI:   ret <16 x i8> [[VSQADD_I]]
// uint8x16_t test_vsqaddq_u8(uint8x16_t a, int8x16_t b) {
//   return vsqaddq_u8(a, b);
// }

// NYI-LABEL: @test_vsqadd_u16(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i16> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i16> %b to <8 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <4 x i16> @llvm.aarch64.neon.usqadd.v4i16(<4 x i16> %a, <4 x i16> %b)
// NYI:   ret <4 x i16> [[VSQADD2_I]]
// uint16x4_t test_vsqadd_u16(uint16x4_t a, int16x4_t b) {
//   return vsqadd_u16(a, b);
// }

// NYI-LABEL: @test_vsqaddq_u16(
// NYI:   [[TMP0:%.*]] = bitcast <8 x i16> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <8 x i16> %b to <16 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <8 x i16> @llvm.aarch64.neon.usqadd.v8i16(<8 x i16> %a, <8 x i16> %b)
// NYI:   ret <8 x i16> [[VSQADD2_I]]
// uint16x8_t test_vsqaddq_u16(uint16x8_t a, int16x8_t b) {
//   return vsqaddq_u16(a, b);
// }

// NYI-LABEL: @test_vsqadd_u32(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i32> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i32> %b to <8 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <2 x i32> @llvm.aarch64.neon.usqadd.v2i32(<2 x i32> %a, <2 x i32> %b)
// NYI:   ret <2 x i32> [[VSQADD2_I]]
// uint32x2_t test_vsqadd_u32(uint32x2_t a, int32x2_t b) {
//   return vsqadd_u32(a, b);
// }

// NYI-LABEL: @test_vsqaddq_u32(
// NYI:   [[TMP0:%.*]] = bitcast <4 x i32> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <4 x i32> %b to <16 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <4 x i32> @llvm.aarch64.neon.usqadd.v4i32(<4 x i32> %a, <4 x i32> %b)
// NYI:   ret <4 x i32> [[VSQADD2_I]]
// uint32x4_t test_vsqaddq_u32(uint32x4_t a, int32x4_t b) {
//   return vsqaddq_u32(a, b);
// }

// NYI-LABEL: @test_vsqaddq_u64(
// NYI:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// NYI:   [[VSQADD2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.usqadd.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   ret <2 x i64> [[VSQADD2_I]]
// uint64x2_t test_vsqaddq_u64(uint64x2_t a, int64x2_t b) {
//   return vsqaddq_u64(a, b);
// }

// NYI-LABEL: @test_vabs_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VABS1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.abs.v1i64(<1 x i64> %a)
// NYI:   ret <1 x i64> [[VABS1_I]]
// int64x1_t test_vabs_s64(int64x1_t a) {
//   return vabs_s64(a);
// }

// NYI-LABEL: @test_vqabs_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VQABS_V1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqabs.v1i64(<1 x i64> %a)
// NYI:   [[VQABS_V2_I:%.*]] = bitcast <1 x i64> [[VQABS_V1_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQABS_V1_I]]
// int64x1_t test_vqabs_s64(int64x1_t a) {
//   return vqabs_s64(a);
// }

// NYI-LABEL: @test_vqneg_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VQNEG_V1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.sqneg.v1i64(<1 x i64> %a)
// NYI:   [[VQNEG_V2_I:%.*]] = bitcast <1 x i64> [[VQNEG_V1_I]] to <8 x i8>
// NYI:   ret <1 x i64> [[VQNEG_V1_I]]
// int64x1_t test_vqneg_s64(int64x1_t a) {
//   return vqneg_s64(a);
// }

// NYI-LABEL: @test_vneg_s64(
// NYI:   [[SUB_I:%.*]] = sub <1 x i64> zeroinitializer, %a
// NYI:   ret <1 x i64> [[SUB_I]]
// int64x1_t test_vneg_s64(int64x1_t a) {
//   return vneg_s64(a);
// }

float32_t test_vaddv_f32(float32x2_t a) {
  return vaddv_f32(a);

  // CIR-LABEL: vaddv_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.faddv" {{%.*}} : (!cir.vector<!cir.float x 2>) -> !cir.float

  // LLVM-LABEL: test_vaddv_f32
  // LLVM-SAME: (<2 x float> [[a:%.*]])
  // LLVM: [[VADDV_F32_I:%.*]] = call float @llvm.aarch64.neon.faddv.f32.v2f32(<2 x float> [[a]])
  // LLVM: ret float [[VADDV_F32_I]]
}

float32_t test_vaddvq_f32(float32x4_t a) {
  return vaddvq_f32(a);

  // CIR-LABEL: vaddvq_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.faddv" {{%.*}} : (!cir.vector<!cir.float x 4>) -> !cir.float

  // LLVM-LABEL: test_vaddvq_f32
  // LLVM-SAME: (<4 x float> [[a:%.*]])
  // LLVM: [[VADDVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.faddv.f32.v4f32(<4 x float> [[a]])
  // LLVM: ret float [[VADDVQ_F32_I]]
}

float64_t test_vaddvq_f64(float64x2_t a) {
  return vaddvq_f64(a);

  // CIR-LABEL: vaddvq_f64
  // CIR: cir.llvm.intrinsic "aarch64.neon.faddv" {{%.*}} : (!cir.vector<!cir.double x 2>) -> !cir.double

  // LLVM-LABEL: test_vaddvq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]])
  // LLVM: [[VADDVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.faddv.f64.v2f64(<2 x double> [[a]])
  // LLVM: ret double [[VADDVQ_F64_I]]
}

float32_t test_vmaxv_f32(float32x2_t a) {
  return vmaxv_f32(a);

  // CIR-LABEL: vmaxv_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fmaxv" {{%.*}} : (!cir.vector<!cir.float x 2>) -> !cir.float

  // LLVM-LABEL: test_vmaxv_f32
  // LLVM-SAME: (<2 x float> [[a:%.*]])
  // LLVM:   [[VMAXV_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxv.f32.v2f32(<2 x float> [[a]])
  // LLVM:   ret float [[VMAXV_F32_I]]
}

float64_t test_vmaxvq_f64(float64x2_t a) {
  return vmaxvq_f64(a);

  // CIR-LABEL: vmaxvq_f64
  // CIR: cir.llvm.intrinsic "aarch64.neon.fmaxv" {{%.*}} : (!cir.vector<!cir.double x 2>) -> !cir.double

  // LLVM-LABEL: test_vmaxvq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]])
  // LLVM:  [[VMAXVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxv.f64.v2f64(<2 x double> [[a]])
  // LLVM:  ret double [[VMAXVQ_F64_I]]
}

// NYI-LABEL: @test_vminv_f32(
// NYI:   [[VMINV_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v2f32(<2 x float> %a)
// NYI:   ret float [[VMINV_F32_I]]
// float32_t test_vminv_f32(float32x2_t a) {
//   return vminv_f32(a);
// }

float64_t test_vminvq_f64(float64x2_t a) {
  return vminvq_f64(a);

  // CIR-LABEL: vminvq_f64
  // CIR: cir.llvm.intrinsic "aarch64.neon.fminv" {{%.*}} : (!cir.vector<!cir.double x 2>) -> !cir.double

  // LLVM-LABEL: @test_vminvq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]])
  // LLVM:   [[VMINVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fminv.f64.v2f64(<2 x double> [[a]])
  // LLVM:   ret double [[VMINVQ_F64_I]]
}


float32_t test_vmaxnmvq_f32(float32x4_t a) {
  return vmaxnmvq_f32(a);

  // CIR-LABEL: vmaxnmvq_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fmaxnmv" {{%.*}} : (!cir.vector<!cir.float x 4>) -> !cir.float

  // LLVM-LABEL: @test_vmaxnmvq_f32
  // LLVM-SAME: (<4 x float> [[a:%.*]])
  // LLVM:  [[VMAXNMVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v4f32(<4 x float> [[a]])
  // LLVM:  ret float [[VMAXNMVQ_F32_I]]
}

float64_t test_vmaxnmvq_f64(float64x2_t a) {
  return vmaxnmvq_f64(a);

  // CIR-LABEL: vmaxnmvq_f64
  // CIR: cir.llvm.intrinsic "aarch64.neon.fmaxnmv" {{%.*}} : (!cir.vector<!cir.double x 2>) -> !cir.double

  // LLVM-LABEL: @test_vmaxnmvq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]])
  // LLVM:  [[VMAXNMVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fmaxnmv.f64.v2f64(<2 x double> [[a]])
  // LLVM:  ret double [[VMAXNMVQ_F64_I]]
}

float32_t test_vmaxnmv_f32(float32x2_t a) {
  return vmaxnmv_f32(a);

  // CIR-LABEL: vmaxnmv_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fmaxnmv" {{%.*}} : (!cir.vector<!cir.float x 2>) -> !cir.float

  // LLVM-LABEL: @test_vmaxnmv_f32
  // LLVM-SAME: (<2 x float> [[a:%.*]])
  // LLVM:   [[VMAXNMV_F32_I:%.*]] = call float @llvm.aarch64.neon.fmaxnmv.f32.v2f32(<2 x float> [[a]])
  // LLVM:   ret float [[VMAXNMV_F32_I]]
}

float64_t test_vminnmvq_f64(float64x2_t a) {
  return vminnmvq_f64(a);

  // CIR-LABEL: vminnmvq_f64
  // CIR: cir.llvm.intrinsic "aarch64.neon.fminnmv" {{%.*}} : (!cir.vector<!cir.double x 2>) -> !cir.double

  // LLVM-LABEL: @test_vminnmvq_f64
  // LLVM-SAME: (<2 x double> [[a:%.*]])
  // LLVM:   [[VMINNMVQ_F64_I:%.*]] = call double @llvm.aarch64.neon.fminnmv.f64.v2f64(<2 x double> [[a]])
  // LLVM:   ret double [[VMINNMVQ_F64_I]]
}

float32_t test_vminnmvq_f32(float32x4_t a) {
  return vminnmvq_f32(a);

  // CIR-LABEL: vminnmvq_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fminnmv" {{%.*}} : (!cir.vector<!cir.float x 4>) -> !cir.float

  // LLVM-LABEL: @test_vminnmvq_f32
  // LLVM-SAME: (<4 x float> [[a:%.*]])
  // LLVM:   [[VMINNMVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v4f32(<4 x float> [[a]])
  // LLVM:   ret float [[VMINNMVQ_F32_I]]
}

float32_t test_vminnmv_f32(float32x2_t a) {
  return vminnmv_f32(a);

  // CIR-LABEL: vminnmv_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fminnmv" {{%.*}} : (!cir.vector<!cir.float x 2>) -> !cir.float

  // LLVM-LABEL: @test_vminnmv_f32
  // LLVM-SAME: (<2 x float> [[a:%.*]])
  // LLVM:   [[VMINNMV_F32_I:%.*]] = call float @llvm.aarch64.neon.fminnmv.f32.v2f32(<2 x float> [[a]])
  // LLVM:   ret float [[VMINNMV_F32_I]]
}

// NYI-LABEL: @test_vpaddq_s64(
// NYI:   [[VPADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VPADDQ_V2_I]]
// int64x2_t test_vpaddq_s64(int64x2_t a, int64x2_t b) {
//   return vpaddq_s64(a, b);
// }

// NYI-LABEL: @test_vpaddq_u64(
// NYI:   [[VPADDQ_V2_I:%.*]] = call <2 x i64> @llvm.aarch64.neon.addp.v2i64(<2 x i64> %a, <2 x i64> %b)
// NYI:   [[VPADDQ_V3_I:%.*]] = bitcast <2 x i64> [[VPADDQ_V2_I]] to <16 x i8>
// NYI:   ret <2 x i64> [[VPADDQ_V2_I]]
// uint64x2_t test_vpaddq_u64(uint64x2_t a, uint64x2_t b) {
//   return vpaddq_u64(a, b);
// }

// NYI-LABEL: @test_vpaddd_u64(
// NYI:   [[VPADDD_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// NYI:   ret i64 [[VPADDD_U64_I]]
// uint64_t test_vpaddd_u64(uint64x2_t a) {
//   return vpaddd_u64(a);
// }

// NYI-LABEL: @test_vaddvq_s64(
// NYI:   [[VADDVQ_S64_I:%.*]] = call i64 @llvm.aarch64.neon.saddv.i64.v2i64(<2 x i64> %a)
// NYI:   ret i64 [[VADDVQ_S64_I]]
// int64_t test_vaddvq_s64(int64x2_t a) {
//   return vaddvq_s64(a);
// }

// NYI-LABEL: @test_vaddvq_u64(
// NYI:   [[VADDVQ_U64_I:%.*]] = call i64 @llvm.aarch64.neon.uaddv.i64.v2i64(<2 x i64> %a)
// NYI:   ret i64 [[VADDVQ_U64_I]]
// uint64_t test_vaddvq_u64(uint64x2_t a) {
//   return vaddvq_u64(a);
// }

// NYI-LABEL: @test_vadd_f64(
// NYI:   [[ADD_I:%.*]] = fadd <1 x double> %a, %b
// NYI:   ret <1 x double> [[ADD_I]]
// float64x1_t test_vadd_f64(float64x1_t a, float64x1_t b) {
//   return vadd_f64(a, b);
// }

// NYI-LABEL: @test_vmul_f64(
// NYI:   [[MUL_I:%.*]] = fmul <1 x double> %a, %b
// NYI:   ret <1 x double> [[MUL_I]]
// float64x1_t test_vmul_f64(float64x1_t a, float64x1_t b) {
//   return vmul_f64(a, b);
// }

// NYI-LABEL: @test_vdiv_f64(
// NYI:   [[DIV_I:%.*]] = fdiv <1 x double> %a, %b
// NYI:   ret <1 x double> [[DIV_I]]
// float64x1_t test_vdiv_f64(float64x1_t a, float64x1_t b) {
//   return vdiv_f64(a, b);
// }

// NYI-LABEL: @test_vmla_f64(
// NYI:   [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// NYI:   [[ADD_I:%.*]] = fadd <1 x double> %a, [[MUL_I]]
// NYI:   ret <1 x double> [[ADD_I]]
// float64x1_t test_vmla_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
//   return vmla_f64(a, b, c);
// }

// NYI-LABEL: @test_vmls_f64(
// NYI:   [[MUL_I:%.*]] = fmul <1 x double> %b, %c
// NYI:   [[SUB_I:%.*]] = fsub <1 x double> %a, [[MUL_I]]
// NYI:   ret <1 x double> [[SUB_I]]
// float64x1_t test_vmls_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
//   return vmls_f64(a, b, c);
// }

// NYI-LABEL: @test_vfma_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// NYI:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> %b, <1 x double> %c, <1 x double> %a)
// NYI:   ret <1 x double> [[TMP3]]
// float64x1_t test_vfma_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
//   return vfma_f64(a, b, c);
// }

// NYI-LABEL: @test_vfms_f64(
// NYI:   [[SUB_I:%.*]] = fneg <1 x double> %b
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> [[SUB_I]] to <8 x i8>
// NYI:   [[TMP2:%.*]] = bitcast <1 x double> %c to <8 x i8>
// NYI:   [[TMP3:%.*]] = call <1 x double> @llvm.fma.v1f64(<1 x double> [[SUB_I]], <1 x double> %c, <1 x double> %a)
// NYI:   ret <1 x double> [[TMP3]]
// float64x1_t test_vfms_f64(float64x1_t a, float64x1_t b, float64x1_t c) {
//   return vfms_f64(a, b, c);
// }

// NYI-LABEL: @test_vsub_f64(
// NYI:   [[SUB_I:%.*]] = fsub <1 x double> %a, %b
// NYI:   ret <1 x double> [[SUB_I]]
// float64x1_t test_vsub_f64(float64x1_t a, float64x1_t b) {
//   return vsub_f64(a, b);
// }

// NYI-LABEL: @test_vabd_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VABD2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fabd.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x double> [[VABD2_I]]
// float64x1_t test_vabd_f64(float64x1_t a, float64x1_t b) {
//   return vabd_f64(a, b);
// }

float64x1_t test_vmax_f64(float64x1_t a, float64x1_t b) {
  return vmax_f64(a, b);

  // CIR-LABEL: vmax_f64
  // CIR: cir.fmaximum {{%.*}}, {{%.*}} : !cir.vector<!cir.double x 1>

  // LLVM-LABEL: test_vmax_f64
  // LLVM-SAME: (<1 x double> [[a:%.*]], <1 x double> [[b:%.*]])
  // LLVM:   [[TMP0:%.*]] = bitcast <1 x double> [[a]] to <8 x i8>
  // LLVM:   [[TMP1:%.*]] = bitcast <1 x double> [[b]] to <8 x i8>
  // LLVM:   [[VMAX2_I:%.*]] = call <1 x double> @llvm.maximum.v1f64(<1 x double> [[a]], <1 x double> [[b]])
  // LLVM:   ret <1 x double> [[VMAX2_I]]
}

// NYI-LABEL: @test_vmaxnm_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VMAXNM2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fmaxnm.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x double> [[VMAXNM2_I]]
// float64x1_t test_vmaxnm_f64(float64x1_t a, float64x1_t b) {
//   return vmaxnm_f64(a, b);
// }

// NYI-LABEL: @test_vminnm_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VMINNM2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.fminnm.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x double> [[VMINNM2_I]]
// float64x1_t test_vminnm_f64(float64x1_t a, float64x1_t b) {
//   return vminnm_f64(a, b);
// }

// NYI-LABEL: @test_vabs_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VABS1_I:%.*]] = call <1 x double> @llvm.fabs.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VABS1_I]]
// float64x1_t test_vabs_f64(float64x1_t a) {
//   return vabs_f64(a);
// }

// NYI-LABEL: @test_vneg_f64(
// NYI:   [[SUB_I:%.*]] = fneg <1 x double> %a
// NYI:   ret <1 x double> [[SUB_I]]
// float64x1_t test_vneg_f64(float64x1_t a) {
//   return vneg_f64(a);
// }

// NYI-LABEL: @test_vcvt_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtzs.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[TMP1]]
// int64x1_t test_vcvt_s64_f64(float64x1_t a) {
//   return vcvt_s64_f64(a);
// }

// NYI-LABEL: @test_vcvt_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtzu.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[TMP1]]
// uint64x1_t test_vcvt_u64_f64(float64x1_t a) {
//   return vcvt_u64_f64(a);
// }

// NYI-LABEL: @test_vcvtn_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTN1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtns.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTN1_I]]
// int64x1_t test_vcvtn_s64_f64(float64x1_t a) {
//   return vcvtn_s64_f64(a);
// }

// NYI-LABEL: @test_vcvtn_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTN1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtnu.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTN1_I]]
// uint64x1_t test_vcvtn_u64_f64(float64x1_t a) {
//   return vcvtn_u64_f64(a);
// }

// NYI-LABEL: @test_vcvtp_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTP1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtps.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTP1_I]]
// int64x1_t test_vcvtp_s64_f64(float64x1_t a) {
//   return vcvtp_s64_f64(a);
// }

// NYI-LABEL: @test_vcvtp_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTP1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtpu.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTP1_I]]
// uint64x1_t test_vcvtp_u64_f64(float64x1_t a) {
//   return vcvtp_u64_f64(a);
// }

// NYI-LABEL: @test_vcvtm_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTM1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtms.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTM1_I]]
// int64x1_t test_vcvtm_s64_f64(float64x1_t a) {
//   return vcvtm_s64_f64(a);
// }

// NYI-LABEL: @test_vcvtm_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTM1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtmu.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTM1_I]]
// uint64x1_t test_vcvtm_u64_f64(float64x1_t a) {
//   return vcvtm_u64_f64(a);
// }

// NYI-LABEL: @test_vcvta_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTA1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtas.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTA1_I]]
// int64x1_t test_vcvta_s64_f64(float64x1_t a) {
//   return vcvta_s64_f64(a);
// }

// NYI-LABEL: @test_vcvta_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVTA1_I:%.*]] = call <1 x i64> @llvm.aarch64.neon.fcvtau.v1i64.v1f64(<1 x double> %a)
// NYI:   ret <1 x i64> [[VCVTA1_I]]
// uint64x1_t test_vcvta_u64_f64(float64x1_t a) {
//   return vcvta_u64_f64(a);
// }

// NYI-LABEL: @test_vcvt_f64_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VCVT_I:%.*]] = sitofp <1 x i64> %a to <1 x double>
// NYI:   ret <1 x double> [[VCVT_I]]
// float64x1_t test_vcvt_f64_s64(int64x1_t a) {
//   return vcvt_f64_s64(a);
// }

// NYI-LABEL: @test_vcvt_f64_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VCVT_I:%.*]] = uitofp <1 x i64> %a to <1 x double>
// NYI:   ret <1 x double> [[VCVT_I]]
// float64x1_t test_vcvt_f64_u64(uint64x1_t a) {
//   return vcvt_f64_u64(a);
// }

// NYI-LABEL: @test_vcvt_n_s64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// NYI:   [[VCVT_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxs.v1i64.v1f64(<1 x double> [[VCVT_N]], i32 64)
// NYI:   ret <1 x i64> [[VCVT_N1]]
// int64x1_t test_vcvt_n_s64_f64(float64x1_t a) {
//   return vcvt_n_s64_f64(a, 64);
// }

// NYI-LABEL: @test_vcvt_n_u64_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x double>
// NYI:   [[VCVT_N1:%.*]] = call <1 x i64> @llvm.aarch64.neon.vcvtfp2fxu.v1i64.v1f64(<1 x double> [[VCVT_N]], i32 64)
// NYI:   ret <1 x i64> [[VCVT_N1]]
// uint64x1_t test_vcvt_n_u64_f64(float64x1_t a) {
//   return vcvt_n_u64_f64(a, 64);
// }

// NYI-LABEL: @test_vcvt_n_f64_s64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VCVT_N1:%.*]] = call <1 x double> @llvm.aarch64.neon.vcvtfxs2fp.v1f64.v1i64(<1 x i64> [[VCVT_N]], i32 64)
// NYI:   ret <1 x double> [[VCVT_N1]]
// float64x1_t test_vcvt_n_f64_s64(int64x1_t a) {
//   return vcvt_n_f64_s64(a, 64);
// }

// NYI-LABEL: @test_vcvt_n_f64_u64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// NYI:   [[VCVT_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// NYI:   [[VCVT_N1:%.*]] = call <1 x double> @llvm.aarch64.neon.vcvtfxu2fp.v1f64.v1i64(<1 x i64> [[VCVT_N]], i32 64)
// NYI:   ret <1 x double> [[VCVT_N1]]
// float64x1_t test_vcvt_n_f64_u64(uint64x1_t a) {
//   return vcvt_n_f64_u64(a, 64);
// }

// NYI-LABEL: @test_vrndn_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDN1_I:%.*]] = call <1 x double> @llvm.roundeven.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDN1_I]]
// float64x1_t test_vrndn_f64(float64x1_t a) {
//   return vrndn_f64(a);
// }

// NYI-LABEL: @test_vrnda_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDA1_I:%.*]] = call <1 x double> @llvm.round.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDA1_I]]
// float64x1_t test_vrnda_f64(float64x1_t a) {
//   return vrnda_f64(a);
// }

// NYI-LABEL: @test_vrndp_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDP1_I:%.*]] = call <1 x double> @llvm.ceil.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDP1_I]]
// float64x1_t test_vrndp_f64(float64x1_t a) {
//   return vrndp_f64(a);
// }

// NYI-LABEL: @test_vrndm_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDM1_I:%.*]] = call <1 x double> @llvm.floor.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDM1_I]]
// float64x1_t test_vrndm_f64(float64x1_t a) {
//   return vrndm_f64(a);
// }

// NYI-LABEL: @test_vrndx_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDX1_I:%.*]] = call <1 x double> @llvm.rint.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDX1_I]]
// float64x1_t test_vrndx_f64(float64x1_t a) {
//   return vrndx_f64(a);
// }

// NYI-LABEL: @test_vrnd_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDZ1_I:%.*]] = call <1 x double> @llvm.trunc.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDZ1_I]]
// float64x1_t test_vrnd_f64(float64x1_t a) {
//   return vrnd_f64(a);
// }

// NYI-LABEL: @test_vrndi_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRNDI1_I:%.*]] = call <1 x double> @llvm.nearbyint.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRNDI1_I]]
// float64x1_t test_vrndi_f64(float64x1_t a) {
//   return vrndi_f64(a);
// }

// NYI-LABEL: @test_vrsqrte_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRSQRTE_V1_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frsqrte.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRSQRTE_V1_I]]
// float64x1_t test_vrsqrte_f64(float64x1_t a) {
//   return vrsqrte_f64(a);
// }

// NYI-LABEL: @test_vrecpe_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VRECPE_V1_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frecpe.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VRECPE_V1_I]]
// float64x1_t test_vrecpe_f64(float64x1_t a) {
//   return vrecpe_f64(a);
// }

// NYI-LABEL: @test_vsqrt_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[VSQRT_I:%.*]] = call <1 x double> @llvm.sqrt.v1f64(<1 x double> %a)
// NYI:   ret <1 x double> [[VSQRT_I]]
// float64x1_t test_vsqrt_f64(float64x1_t a) {
//   return vsqrt_f64(a);
// }

// NYI-LABEL: @test_vrecps_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VRECPS_V2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frecps.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   ret <1 x double> [[VRECPS_V2_I]]
// float64x1_t test_vrecps_f64(float64x1_t a, float64x1_t b) {
//   return vrecps_f64(a, b);
// }

// NYI-LABEL: @test_vrsqrts_f64(
// NYI:   [[TMP0:%.*]] = bitcast <1 x double> %a to <8 x i8>
// NYI:   [[TMP1:%.*]] = bitcast <1 x double> %b to <8 x i8>
// NYI:   [[VRSQRTS_V2_I:%.*]] = call <1 x double> @llvm.aarch64.neon.frsqrts.v1f64(<1 x double> %a, <1 x double> %b)
// NYI:   [[VRSQRTS_V3_I:%.*]] = bitcast <1 x double> [[VRSQRTS_V2_I]] to <8 x i8>
// NYI:   ret <1 x double> [[VRSQRTS_V2_I]]
// float64x1_t test_vrsqrts_f64(float64x1_t a, float64x1_t b) {
//   return vrsqrts_f64(a, b);
// }

int32_t test_vminv_s32(int32x2_t a) {
  return vminv_s32(a);

  // CIR-LABEL: vminv_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.sminv" {{%.*}} : (!cir.vector<!s32i x 2>) -> !s32i

  // LLVM-LABEL: @test_vminv_s32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VMINV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.sminv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i32 [[VMINV_S32_I]]
}

uint32_t test_vminv_u32(uint32x2_t a) {
  return vminv_u32(a);

  // CIR-LABEL: vminv_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.uminv" {{%.*}} : (!cir.vector<!u32i x 2>) -> !u32i

  // LLVM-LABEL: @test_vminv_u32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VMINV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uminv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i32 [[VMINV_U32_I]]
}

float32_t test_vminvq_f32(float32x4_t a) {
  return vminvq_f32(a);

  // CIR-LABEL: vminvq_f32
  // CIR: cir.llvm.intrinsic "aarch64.neon.fminv" {{%.*}} : (!cir.vector<!cir.float x 4>) -> !cir.float

  // LLVM-LABEL: @test_vminvq_f32
  // LLVM-SAME: (<4 x float> [[a:%.*]])
  // LLVM:  [[VMINVQ_F32_I:%.*]] = call float @llvm.aarch64.neon.fminv.f32.v4f32(<4 x float> [[a]])
  // LLVM:  ret float [[VMINVQ_F32_I]]
}

int32_t test_vmaxv_s32(int32x2_t a) {
  return vmaxv_s32(a);

  // CIR-LABEL: vmaxv_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.smaxv" {{%.*}} : (!cir.vector<!s32i x 2>) -> !s32i

  // LLVM-LABEL: @test_vmaxv_s32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:  [[VMAXV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.smaxv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:  ret i32 [[VMAXV_S32_I]]
}

// NYI-LABEL: @test_vmaxv_u32(
// NYI:   [[VMAXV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v2i32(<2 x i32> %a)
// NYI:   ret i32 [[VMAXV_U32_I]]
uint32_t test_vmaxv_u32(uint32x2_t a) {
  return vmaxv_u32(a);

  // CIR-LABEL: vmaxv_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.umaxv" {{%.*}} : (!cir.vector<!u32i x 2>) -> !u32i

  // LLVM-LABEL: @test_vmaxv_u32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:  [[VMAXV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.umaxv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:  ret i32 [[VMAXV_U32_I]]
}

int32_t test_vaddv_s32(int32x2_t a) {
  return vaddv_s32(a);

  // CIR-LABEL: vaddv_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.saddv" {{%.*}} : (!cir.vector<!s32i x 2>) -> !s32i

  // LLVM-LABEL: test_vaddv_s32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VADDV_S32_I:%.*]] = call i32 @llvm.aarch64.neon.saddv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i32 [[VADDV_S32_I]]
}

uint32_t test_vaddv_u32(uint32x2_t a) {
  return vaddv_u32(a);

  // CIR-LABEL: vaddv_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.uaddv" {{%.*}} : (!cir.vector<!u32i x 2>) -> !u32i

  // LLVM-LABEL: test_vaddv_u32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VADDV_U32_I:%.*]] = call i32 @llvm.aarch64.neon.uaddv.i32.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i32 [[VADDV_U32_I]]
}

int64_t test_vaddlv_s32(int32x2_t a) {
  return vaddlv_s32(a);

  // CIR-LABEL: vaddlv_s32
  // CIR: cir.llvm.intrinsic "aarch64.neon.saddlv" {{%.*}} : (!cir.vector<!s32i x 2>) -> !s64i

  // LLVM-LABEL: test_vaddlv_s32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VADDLV_S32_I:%.*]] = call i64 @llvm.aarch64.neon.saddlv.i64.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i64 [[VADDLV_S32_I]]
}

uint64_t test_vaddlv_u32(uint32x2_t a) {
  return vaddlv_u32(a);

  // CIR-LABEL: vaddlv_u32
  // CIR: cir.llvm.intrinsic "aarch64.neon.uaddlv" {{%.*}} : (!cir.vector<!u32i x 2>) -> !u64i

  // LLVM-LABEL: test_vaddlv_u32
  // LLVM-SAME: (<2 x i32> [[a:%.*]])
  // LLVM:   [[VADDLV_U32_I:%.*]] = call i64 @llvm.aarch64.neon.uaddlv.i64.v2i32(<2 x i32> [[a]])
  // LLVM:   ret i64 [[VADDLV_U32_I]]
}

uint8x8_t test_vmovn_u16(uint16x8_t a) {
  return vmovn_u16(a);
  // CIR-LABEL: vmovn_u16
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u16i x 8>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!u16i x 8>), !cir.vector<!u8i x 8>

  // LLVM: {{.*}}@test_vmovn_u16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <8 x i16> [[A]] to <8 x i8>
  // LLVM: ret <8 x i8> [[VMOVN_I]]
}

uint16x4_t test_vmovn_u32(uint32x4_t a) {
  return vmovn_u32(a);
  // CIR-LABEL: vmovn_u32
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u32i x 4>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!u32i x 4>), !cir.vector<!u16i x 4>

  // LLVM: {{.*}}@test_vmovn_u32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <4 x i32> [[A]] to <4 x i16>
  // LLVM: ret <4 x i16> [[VMOVN_I]]
}

uint32x2_t test_vmovn_u64(uint64x2_t a) {
  return vmovn_u64(a);
  // CIR-LABEL: vmovn_u64
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!u64i x 2>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!u64i x 2>), !cir.vector<!u32i x 2>

  // LLVM: {{.*}}@test_vmovn_u64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <2 x i64> [[A]] to <2 x i32>
  // LLVM: ret <2 x i32> [[VMOVN_I]]
}

int8x8_t test_vmovn_s16(int16x8_t a) {
  return vmovn_s16(a);
  // CIR-LABEL: vmovn_s16
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s16i x 8>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!s16i x 8>), !cir.vector<!s8i x 8>

  // LLVM: {{.*}}@test_vmovn_s16(<8 x i16>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <8 x i16> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <8 x i16> [[A]] to <8 x i8>
  // LLVM: ret <8 x i8> [[VMOVN_I]]
}

int16x4_t test_vmovn_s32(int32x4_t a) {
  return vmovn_s32(a);
  // CIR-LABEL: vmovn_s32
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s32i x 4>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!s32i x 4>), !cir.vector<!s16i x 4>

  // LLVM: {{.*}}@test_vmovn_s32(<4 x i32>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <4 x i32> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <4 x i32> [[A]] to <4 x i16>
  // LLVM: ret <4 x i16> [[VMOVN_I]]
}

int32x2_t test_vmovn_s64(int64x2_t a) {
  return vmovn_s64(a);
  // CIR-LABEL: vmovn_s64
  // CIR: [[ARG:%.*]] = cir.cast(bitcast, {{%.*}} : !cir.vector<!s8i x 16>), !cir.vector<!s64i x 2>
  // CIR: {{%.*}} = cir.cast(integral, [[ARG]] : !cir.vector<!s64i x 2>), !cir.vector<!s32i x 2>

  // LLVM: {{.*}}@test_vmovn_s64(<2 x i64>{{.*}}[[A:%.*]])
  // LLVM: [[VMOVN_1:%.*]] = bitcast <2 x i64> [[A]] to <16 x i8>
  // LLVM: [[VMOVN_I:%.*]] = trunc <2 x i64> [[A]] to <2 x i32>
  // LLVM: ret <2 x i32> [[VMOVN_I]]
}

uint8x8_t test_vld1_dup_u8(uint8_t const * ptr) {
  return vld1_dup_u8(ptr);
}

// CIR-LABEL: vld1_dup_u8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u8i, !cir.vector<!u8i x 8>

// LLVM: {{.*}}test_vld1_dup_u8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <8 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i8> [[VEC]], <8 x i8> poison, <8 x i32> zeroinitializer

int8x8_t test_vld1_dup_s8(int8_t const * ptr) {
  return vld1_dup_s8(ptr);
}

// CIR-LABEL: test_vld1_dup_s8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s8i, !cir.vector<!s8i x 8>

// LLVM: {{.*}}test_vld1_dup_s8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <8 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i8> [[VEC]], <8 x i8> poison, <8 x i32> zeroinitializer

uint16x4_t test_vld1_dup_u16(uint16_t const * ptr) {
  return vld1_dup_u16(ptr);
}

// CIR-LABEL: test_vld1_dup_u16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u16i, !cir.vector<!u16i x 4>

// LLVM: {{.*}}test_vld1_dup_u16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <4 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i16> [[VEC]], <4 x i16> poison, <4 x i32> zeroinitializer

int16x4_t test_vld1_dup_s16(int16_t const * ptr) {
  return vld1_dup_s16(ptr);
}

// CIR-LABEL: test_vld1_dup_s16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s16i, !cir.vector<!s16i x 4>

// LLVM: {{.*}}test_vld1_dup_s16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <4 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i16> [[VEC]], <4 x i16> poison, <4 x i32> zeroinitializer

int32x2_t test_vld1_dup_s32(int32_t const * ptr) {
  return vld1_dup_s32(ptr);
}

// CIR-LABEL: test_vld1_dup_s32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s32i, !cir.vector<!s32i x 2>

// LLVM: {{.*}}test_vld1_dup_s32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i32, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <2 x i32> poison, i32 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x i32> [[VEC]], <2 x i32> poison, <2 x i32> zeroinitializer

int64x1_t test_vld1_dup_s64(int64_t const * ptr) {
  return vld1_dup_s64(ptr);
}

// CIR-LABEL: test_vld1_dup_s64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s64i, !cir.vector<!s64i x 1>

// LLVM: {{.*}}test_vld1_dup_s64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i64, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <1 x i64> poison, i64 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <1 x i64> [[VEC]], <1 x i64> poison, <1 x i32> zeroinitializer

float32x2_t test_vld1_dup_f32(float32_t const * ptr) {
  return vld1_dup_f32(ptr);
}

// CIR-LABEL: test_vld1_dup_f32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.float, !cir.vector<!cir.float x 2>

// LLVM: {{.*}}test_vld1_dup_f32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load float, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <2 x float> poison, float [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x float> [[VEC]], <2 x float> poison, <2 x i32> zeroinitializer

float64x1_t test_vld1_dup_f64(float64_t const * ptr) {
  return vld1_dup_f64(ptr);
}

// CIR-LABEL: test_vld1_dup_f64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.double, !cir.vector<!cir.double x 1>

// LLVM: {{.*}}test_vld1_dup_f64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load double, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <1 x double> poison, double [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <1 x double> [[VEC]], <1 x double> poison, <1 x i32> zeroinitializer

uint8x16_t test_vld1q_dup_u8(uint8_t const * ptr) {
  return vld1q_dup_u8(ptr);
}

// CIR-LABEL: test_vld1q_dup_u8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u8i>, !u8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u8i, !cir.vector<!u8i x 16>

// LLVM: {{.*}}test_vld1q_dup_u8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <16 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <16 x i8> [[VEC]], <16 x i8> poison, <16 x i32> zeroinitializer

int8x16_t test_vld1q_dup_s8(int8_t const * ptr) {
  return vld1q_dup_s8(ptr);
}

// CIR-LABEL: test_vld1q_dup_s8
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s8i>, !s8i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s8i, !cir.vector<!s8i x 16>

// LLVM: {{.*}}test_vld1q_dup_s8(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i8, ptr [[PTR]], align 1
// LLVM: [[VEC:%.*]] = insertelement <16 x i8> poison, i8 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <16 x i8> [[VEC]], <16 x i8> poison, <16 x i32> zeroinitializer

uint16x8_t test_vld1q_dup_u16(uint16_t const * ptr) {
  return vld1q_dup_u16(ptr);
}

// CIR-LABEL: test_vld1q_dup_u16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!u16i>, !u16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !u16i, !cir.vector<!u16i x 8>

// LLVM: {{.*}}test_vld1q_dup_u16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <8 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i16> [[VEC]], <8 x i16> poison, <8 x i32> zeroinitializer

int16x8_t test_vld1q_dup_s16(int16_t const * ptr) {
  return vld1q_dup_s16(ptr);
}

// CIR-LABEL: test_vld1q_dup_s16
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s16i>, !s16i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s16i, !cir.vector<!s16i x 8>

// LLVM: {{.*}}test_vld1q_dup_s16(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i16, ptr [[PTR]], align 2
// LLVM: [[VEC:%.*]] = insertelement <8 x i16> poison, i16 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <8 x i16> [[VEC]], <8 x i16> poison, <8 x i32> zeroinitializer

int32x4_t test_vld1q_dup_s32(int32_t const * ptr) {
  return vld1q_dup_s32(ptr);
}

// CIR-LABEL: test_vld1q_dup_s32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s32i>, !s32i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s32i, !cir.vector<!s32i x 4>

// LLVM: {{.*}}test_vld1q_dup_s32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i32, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <4 x i32> poison, i32 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x i32> [[VEC]], <4 x i32> poison, <4 x i32> zeroinitializer

int64x2_t test_vld1q_dup_s64(int64_t const * ptr) {
  return vld1q_dup_s64(ptr);
}

// CIR-LABEL: test_vld1q_dup_s64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!s64i>, !s64i
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !s64i, !cir.vector<!s64i x 2>

// LLVM: {{.*}}test_vld1q_dup_s64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load i64, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <2 x i64> poison, i64 [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x i64> [[VEC]], <2 x i64> poison, <2 x i32> zeroinitializer

float32x4_t test_vld1q_dup_f32(float32_t const * ptr) {
  return vld1q_dup_f32(ptr);
}

// CIR-LABEL: test_vld1q_dup_f32
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.float, !cir.vector<!cir.float x 4>

// LLVM: {{.*}}test_vld1q_dup_f32(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load float, ptr [[PTR]], align 4
// LLVM: [[VEC:%.*]] = insertelement <4 x float> poison, float [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <4 x float> [[VEC]], <4 x float> poison, <4 x i32> zeroinitializer

float64x2_t test_vld1q_dup_f64(float64_t const * ptr) {
  return vld1q_dup_f64(ptr);
}

// CIR-LABEL: test_vld1q_dup_f64
// CIR: [[VAL:%.*]] = cir.load {{%.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR: {{%.*}} = cir.vec.splat [[VAL]] : !cir.double, !cir.vector<!cir.double x 2>

// LLVM: {{.*}}test_vld1q_dup_f64(ptr{{.*}}[[PTR:%.*]])
// LLVM: [[VAL:%.*]] = load double, ptr [[PTR]], align 8
// LLVM: [[VEC:%.*]] = insertelement <2 x double> poison, double [[VAL]], i64 0
// LLVM: {{%.*}} = shufflevector <2 x double> [[VEC]], <2 x double> poison, <2 x i32> zeroinitializer
