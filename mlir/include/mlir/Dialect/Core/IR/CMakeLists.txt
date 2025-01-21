set(LLVM_TARGET_DEFINITIONS CoreAttrs.td)
mlir_tablegen(CoreAttrs.h.inc -gen-attrdef-decls -dialect=core)
mlir_tablegen(CoreAttrs.cpp.inc -gen-attrdef-defs -dialect=core)
add_public_tablegen_target(MLIRCoreAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS CoreOps.td)
mlir_tablegen(CoreOps.h.inc -gen-op-decls -dialect=core)
mlir_tablegen(CoreOps.cpp.inc -gen-op-defs -dialect=core)
mlir_tablegen(CoreOpsDialect.h.inc -gen-dialect-decls -dialect=core)
add_public_tablegen_target(MLIRCoreOpsIncGen)

add_mlir_dialect_library(MLIRCoreDialect
  CoreDialect.cpp

  DEPENDS
  MLIRCoreAttrsIncGen
  MLIRCoreOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)