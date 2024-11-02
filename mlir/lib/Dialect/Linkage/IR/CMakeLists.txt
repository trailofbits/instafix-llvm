add_mlir_dialect_library(MLIRLinkageDialect
  Linkage.cpp

  DEPENDS
  MLIRLinkageEnumsGen
  MLIRLinkageIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  )
