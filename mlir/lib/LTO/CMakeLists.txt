add_mlir_library(MLIRLTO
  LTO.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/LTO

  LINK_LIBS PUBLIC
  MLIRAsmParser
  MLIRBytecodeReader
  MLIRIR
  )
