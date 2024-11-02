add_mlir_dialect_library(MLIRLinkageDialect
  Linkage.cpp

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  )
