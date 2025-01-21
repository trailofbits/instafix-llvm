#ifndef CORE_OPS
#define CORE_OPS

include "mlir/IR/OpBase.td"

def Core_Dialect : Dialect {
  let name = "core";
  let cppNamespace = "::mlir::core";
  let description = [{
    Core dialect providing basic operations and attributes including linkage.
  }];
}

class Core_Op<string mnemonic, list<Trait> traits = []> :
    Op<Core_Dialect, mnemonic, traits>;

#endif // CORE_OPS