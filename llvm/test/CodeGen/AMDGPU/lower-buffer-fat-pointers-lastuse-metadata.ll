; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 < %s | FileCheck --check-prefix=GFX12 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+cumode < %s | FileCheck --check-prefix=GFX12 %s


define amdgpu_kernel void @buffer_last_use_load_0(ptr addrspace(7) %in, ptr addrspace(7) %out) {
; GFX12-LABEL: buffer_last_use_load_0:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_clause 0x2
; GFX12-NEXT:    s_load_b128 s[0:3], s[4:5], 0x0
; GFX12-NEXT:    s_load_b128 s[8:11], s[4:5], 0x20
; GFX12-NEXT:    s_load_b32 s6, s[4:5], 0x10
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s1
; GFX12-NEXT:    v_dual_mov_b32 v2, s2 :: v_dual_mov_b32 v3, s3
; GFX12-NEXT:    v_dual_mov_b32 v7, s8 :: v_dual_mov_b32 v8, s9
; GFX12-NEXT:    v_dual_mov_b32 v9, s10 :: v_dual_mov_b32 v10, s11
; GFX12-NEXT:    scratch_store_b128 off, v[0:3], off offset:32
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[5:6], off, off offset:40
; GFX12-NEXT:    scratch_load_b32 v4, off, off offset:36
; GFX12-NEXT:    s_load_b32 s1, s[4:5], 0x30
; GFX12-NEXT:    scratch_store_b128 off, v[7:10], off
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[1:2], off, off offset:8
; GFX12-NEXT:    scratch_load_b32 v0, off, off offset:4
; GFX12-NEXT:    v_mov_b32_e32 v7, s6
; GFX12-NEXT:    v_mov_b32_e32 v9, s0
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_mov_b32_e32 v3, s1
; GFX12-NEXT:    s_mov_b32 s1, exec_lo
; GFX12-NEXT:  .LBB0_1: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x2
; GFX12-NEXT:    v_readfirstlane_b32 s4, v4
; GFX12-NEXT:    v_readfirstlane_b32 s5, v5
; GFX12-NEXT:    v_readfirstlane_b32 s6, v6
; GFX12-NEXT:    v_readfirstlane_b32 s7, v7
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[4:5]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[6:7]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_load_b32 v8, v9, s[4:7], null offen th:TH_LOAD_LU
; GFX12-NEXT:    ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
; GFX12-NEXT:    ; implicit-def: $vgpr9
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB0_1
; GFX12-NEXT:  ; %bb.2:
; GFX12-NEXT:    s_mov_b32 exec_lo, s1
; GFX12-NEXT:    v_mov_b32_e32 v4, s8
; GFX12-NEXT:    s_mov_b32 s0, exec_lo
; GFX12-NEXT:  .LBB0_3: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x1
; GFX12-NEXT:    v_readfirstlane_b32 s4, v0
; GFX12-NEXT:    v_readfirstlane_b32 s5, v1
; GFX12-NEXT:    v_readfirstlane_b32 s6, v2
; GFX12-NEXT:    v_readfirstlane_b32 s7, v3
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[0:1]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[2:3]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_store_b32 v8, v4, s[4:7], null offen
; GFX12-NEXT:    ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
; GFX12-NEXT:    ; implicit-def: $vgpr8
; GFX12-NEXT:    ; implicit-def: $vgpr4
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB0_3
; GFX12-NEXT:  ; %bb.4:
; GFX12-NEXT:    s_endpgm
entry:
  %val = load i32, ptr addrspace(7) %in, !amdgpu.last.use !{}
  store i32 %val, ptr addrspace(7) %out
  ret void
}

define amdgpu_kernel void @buffer_last_use_load_1(ptr addrspace(7) %in, ptr addrspace(7) %out) {
; GFX12-LABEL: buffer_last_use_load_1:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_clause 0x2
; GFX12-NEXT:    s_load_b128 s[0:3], s[4:5], 0x0
; GFX12-NEXT:    s_load_b128 s[8:11], s[4:5], 0x20
; GFX12-NEXT:    s_load_b32 s6, s[4:5], 0x10
; GFX12-NEXT:    v_and_b32_e32 v0, 0x3ff, v0
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_dual_mov_b32 v4, s3 :: v_dual_mov_b32 v3, s2
; GFX12-NEXT:    v_dual_mov_b32 v2, s1 :: v_dual_mov_b32 v1, s0
; GFX12-NEXT:    v_dual_mov_b32 v8, s8 :: v_dual_mov_b32 v9, s9
; GFX12-NEXT:    v_dual_mov_b32 v10, s10 :: v_dual_mov_b32 v11, s11
; GFX12-NEXT:    scratch_store_b128 off, v[1:4], off offset:32
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[6:7], off, off offset:40
; GFX12-NEXT:    scratch_load_b32 v5, off, off offset:36
; GFX12-NEXT:    s_load_b32 s1, s[4:5], 0x30
; GFX12-NEXT:    scratch_store_b128 off, v[8:11], off
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[2:3], off, off offset:8
; GFX12-NEXT:    scratch_load_b32 v1, off, off offset:4
; GFX12-NEXT:    v_mov_b32_e32 v8, s6
; GFX12-NEXT:    v_lshl_add_u32 v9, v0, 2, s0
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_mov_b32_e32 v4, s1
; GFX12-NEXT:    s_mov_b32 s1, exec_lo
; GFX12-NEXT:  .LBB1_1: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x2
; GFX12-NEXT:    v_readfirstlane_b32 s4, v5
; GFX12-NEXT:    v_readfirstlane_b32 s5, v6
; GFX12-NEXT:    v_readfirstlane_b32 s6, v7
; GFX12-NEXT:    v_readfirstlane_b32 s7, v8
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[5:6]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[7:8]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_load_b32 v0, v9, s[4:7], null offen th:TH_LOAD_LU
; GFX12-NEXT:    ; implicit-def: $vgpr5_vgpr6_vgpr7_vgpr8
; GFX12-NEXT:    ; implicit-def: $vgpr9
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB1_1
; GFX12-NEXT:  ; %bb.2:
; GFX12-NEXT:    s_mov_b32 exec_lo, s1
; GFX12-NEXT:    v_mov_b32_e32 v5, s8
; GFX12-NEXT:    s_mov_b32 s0, exec_lo
; GFX12-NEXT:  .LBB1_3: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x1
; GFX12-NEXT:    v_readfirstlane_b32 s4, v1
; GFX12-NEXT:    v_readfirstlane_b32 s5, v2
; GFX12-NEXT:    v_readfirstlane_b32 s6, v3
; GFX12-NEXT:    v_readfirstlane_b32 s7, v4
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[1:2]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[3:4]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_store_b32 v0, v5, s[4:7], null offen
; GFX12-NEXT:    ; implicit-def: $vgpr1_vgpr2_vgpr3_vgpr4
; GFX12-NEXT:    ; implicit-def: $vgpr0
; GFX12-NEXT:    ; implicit-def: $vgpr5
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB1_3
; GFX12-NEXT:  ; %bb.4:
; GFX12-NEXT:    s_endpgm
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, ptr addrspace(7) %in, i32 %tid
  %val = load i32, ptr addrspace(7) %val.gep, align 4, !amdgpu.last.use !{}
  store i32 %val, ptr addrspace(7) %out
  ret void
}

define amdgpu_kernel void @buffer_last_use_and_volatile_load(ptr addrspace(7) %in, ptr addrspace(7) %out) {
; GFX12-LABEL: buffer_last_use_and_volatile_load:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_clause 0x2
; GFX12-NEXT:    s_load_b128 s[0:3], s[4:5], 0x0
; GFX12-NEXT:    s_load_b128 s[8:11], s[4:5], 0x20
; GFX12-NEXT:    s_load_b32 s6, s[4:5], 0x10
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s1
; GFX12-NEXT:    v_dual_mov_b32 v2, s2 :: v_dual_mov_b32 v3, s3
; GFX12-NEXT:    v_dual_mov_b32 v7, s8 :: v_dual_mov_b32 v8, s9
; GFX12-NEXT:    v_dual_mov_b32 v9, s10 :: v_dual_mov_b32 v10, s11
; GFX12-NEXT:    scratch_store_b128 off, v[0:3], off offset:32
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[5:6], off, off offset:40
; GFX12-NEXT:    scratch_load_b32 v4, off, off offset:36
; GFX12-NEXT:    s_load_b32 s1, s[4:5], 0x30
; GFX12-NEXT:    scratch_store_b128 off, v[7:10], off
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[1:2], off, off offset:8
; GFX12-NEXT:    scratch_load_b32 v0, off, off offset:4
; GFX12-NEXT:    v_mov_b32_e32 v7, s6
; GFX12-NEXT:    v_mov_b32_e32 v9, s0
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_mov_b32_e32 v3, s1
; GFX12-NEXT:    s_mov_b32 s1, exec_lo
; GFX12-NEXT:  .LBB2_1: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x2
; GFX12-NEXT:    v_readfirstlane_b32 s4, v4
; GFX12-NEXT:    v_readfirstlane_b32 s5, v5
; GFX12-NEXT:    v_readfirstlane_b32 s6, v6
; GFX12-NEXT:    v_readfirstlane_b32 s7, v7
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[4:5]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[6:7]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_load_b32 v8, v9, s[4:7], null offen th:TH_LOAD_BYPASS scope:SCOPE_SYS
; GFX12-NEXT:    ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
; GFX12-NEXT:    ; implicit-def: $vgpr9
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB2_1
; GFX12-NEXT:  ; %bb.2:
; GFX12-NEXT:    s_mov_b32 exec_lo, s1
; GFX12-NEXT:    v_mov_b32_e32 v4, s8
; GFX12-NEXT:    s_mov_b32 s0, exec_lo
; GFX12-NEXT:  .LBB2_3: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x1
; GFX12-NEXT:    v_readfirstlane_b32 s4, v0
; GFX12-NEXT:    v_readfirstlane_b32 s5, v1
; GFX12-NEXT:    v_readfirstlane_b32 s6, v2
; GFX12-NEXT:    v_readfirstlane_b32 s7, v3
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[0:1]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[2:3]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_store_b32 v8, v4, s[4:7], null offen
; GFX12-NEXT:    ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
; GFX12-NEXT:    ; implicit-def: $vgpr8
; GFX12-NEXT:    ; implicit-def: $vgpr4
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB2_3
; GFX12-NEXT:  ; %bb.4:
; GFX12-NEXT:    s_endpgm
entry:
  %val = load volatile i32, ptr addrspace(7) %in, !amdgpu.last.use !{}
  store i32 %val, ptr addrspace(7) %out
  ret void
}

define amdgpu_kernel void @buffer_last_use_and_nontemporal_load(ptr addrspace(7) %in, ptr addrspace(7) %out) {
; GFX12-LABEL: buffer_last_use_and_nontemporal_load:
; GFX12:       ; %bb.0: ; %entry
; GFX12-NEXT:    s_clause 0x2
; GFX12-NEXT:    s_load_b128 s[0:3], s[4:5], 0x0
; GFX12-NEXT:    s_load_b128 s[8:11], s[4:5], 0x20
; GFX12-NEXT:    s_load_b32 s6, s[4:5], 0x10
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_dual_mov_b32 v0, s0 :: v_dual_mov_b32 v1, s1
; GFX12-NEXT:    v_dual_mov_b32 v2, s2 :: v_dual_mov_b32 v3, s3
; GFX12-NEXT:    v_dual_mov_b32 v7, s8 :: v_dual_mov_b32 v8, s9
; GFX12-NEXT:    v_dual_mov_b32 v9, s10 :: v_dual_mov_b32 v10, s11
; GFX12-NEXT:    scratch_store_b128 off, v[0:3], off offset:32
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[5:6], off, off offset:40
; GFX12-NEXT:    scratch_load_b32 v4, off, off offset:36
; GFX12-NEXT:    s_load_b32 s1, s[4:5], 0x30
; GFX12-NEXT:    scratch_store_b128 off, v[7:10], off
; GFX12-NEXT:    s_clause 0x1
; GFX12-NEXT:    scratch_load_b64 v[1:2], off, off offset:8
; GFX12-NEXT:    scratch_load_b32 v0, off, off offset:4
; GFX12-NEXT:    v_mov_b32_e32 v7, s6
; GFX12-NEXT:    v_mov_b32_e32 v9, s0
; GFX12-NEXT:    s_wait_kmcnt 0x0
; GFX12-NEXT:    v_mov_b32_e32 v3, s1
; GFX12-NEXT:    s_mov_b32 s1, exec_lo
; GFX12-NEXT:  .LBB3_1: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x2
; GFX12-NEXT:    v_readfirstlane_b32 s4, v4
; GFX12-NEXT:    v_readfirstlane_b32 s5, v5
; GFX12-NEXT:    v_readfirstlane_b32 s6, v6
; GFX12-NEXT:    v_readfirstlane_b32 s7, v7
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[4:5]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[6:7]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_load_b32 v8, v9, s[4:7], null offen th:TH_LOAD_LU
; GFX12-NEXT:    ; implicit-def: $vgpr4_vgpr5_vgpr6_vgpr7
; GFX12-NEXT:    ; implicit-def: $vgpr9
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB3_1
; GFX12-NEXT:  ; %bb.2:
; GFX12-NEXT:    s_mov_b32 exec_lo, s1
; GFX12-NEXT:    v_mov_b32_e32 v4, s8
; GFX12-NEXT:    s_mov_b32 s0, exec_lo
; GFX12-NEXT:  .LBB3_3: ; =>This Inner Loop Header: Depth=1
; GFX12-NEXT:    s_wait_loadcnt 0x1
; GFX12-NEXT:    v_readfirstlane_b32 s4, v0
; GFX12-NEXT:    v_readfirstlane_b32 s5, v1
; GFX12-NEXT:    v_readfirstlane_b32 s6, v2
; GFX12-NEXT:    v_readfirstlane_b32 s7, v3
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
; GFX12-NEXT:    v_cmp_eq_u64_e32 vcc_lo, s[4:5], v[0:1]
; GFX12-NEXT:    v_cmp_eq_u64_e64 s0, s[6:7], v[2:3]
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_delay_alu instid0(VALU_DEP_1)
; GFX12-NEXT:    s_and_b32 s0, vcc_lo, s0
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_and_saveexec_b32 s0, s0
; GFX12-NEXT:    s_wait_loadcnt 0x0
; GFX12-NEXT:    buffer_store_b32 v8, v4, s[4:7], null offen
; GFX12-NEXT:    ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
; GFX12-NEXT:    ; implicit-def: $vgpr8
; GFX12-NEXT:    ; implicit-def: $vgpr4
; GFX12-NEXT:    s_wait_alu 0xfffe
; GFX12-NEXT:    s_xor_b32 exec_lo, exec_lo, s0
; GFX12-NEXT:    s_cbranch_execnz .LBB3_3
; GFX12-NEXT:  ; %bb.4:
; GFX12-NEXT:    s_endpgm
entry:
  %val = load i32, ptr addrspace(7) %in, !amdgpu.last.use !{}, !nontemporal !0
  store i32 %val, ptr addrspace(7) %out
  ret void
}

!0 = !{i32 1}
declare i32 @llvm.amdgcn.workitem.id.x()
