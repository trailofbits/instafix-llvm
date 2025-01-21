#ifndef MLIR_DIALECT_CORE_IR_COREDIALECT_H_
#define MLIR_DIALECT_CORE_IR_COREDIALECT_H_

#include "mlir/IR/Dialect.h"

#include "CoreOpsDialect.h.inc"

#define GET_ATTRDEF_DECLARATIONS
#include "CoreAttrs.h.inc"

#endif // MLIR_DIALECT_CORE_IR_COREDIALECT_H_