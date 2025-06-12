module attributes {dlti.dl_spec = #dlti.dl_spec<f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<[32, 64]> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, "dlti.endianness" = "little">, llvm.target_triple = ""} {
  llvm.mlir.global weak @v(1 : i8) {addr_space = 0 : i32} : i8
  llvm.mlir.global_ctors ctors = [@f], priorities = [65535 : i32], data = [@v]
  llvm.func weak @f() {
    llvm.return
  }
}
