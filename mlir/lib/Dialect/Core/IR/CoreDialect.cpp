#include "mlir/Dialect/Core/IR/CoreDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::core;

#include "CoreOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "CoreAttrs.cpp.inc"

void CoreDialect::initialize() {
  addAttributes
#define GET_ATTRDEF_LIST
#include "CoreAttrs.cpp.inc"
      >();
}