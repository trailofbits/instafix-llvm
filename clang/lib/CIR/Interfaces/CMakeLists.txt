add_clang_library(MLIRCIRInterfaces
  ASTAttrInterfaces.cpp
  CIROpInterfaces.cpp
  CIRLinkerInterface.cpp
  CIRLoopOpInterface.cpp
  CIRFPTypeInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIREnumsGen
  MLIRCIRFPTypeInterfaceIncGen
  MLIRCIRLoopOpInterfaceIncGen
  MLIRCIROpInterfacesIncGen

  LINK_LIBS
  ${dialect_libs}
  MLIRIR
  MLIRSupport
 )
