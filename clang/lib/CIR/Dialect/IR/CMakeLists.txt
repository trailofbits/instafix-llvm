add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIROpenCLAttrs.cpp
  CIRDataLayout.cpp
  CIRDialect.cpp
  CIRMemorySlot.cpp
  CIRTypes.cpp
  FPEnv.cpp
  CIRLinkerInterface.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen
  MLIRCIROpsIncGen
  MLIRCIRTypeConstraintsIncGen
  MLIRCIREnumsGen
  MLIRSymbolInterfacesIncGen
  MLIRCIRASTAttrInterfacesIncGen
  MLIRCIROpInterfacesIncGen
  MLIRCIRLoopOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinker
  MLIRCIRInterfaces
  MLIRDLTIDialect
  MLIRDataLayoutInterfaces
  MLIRFuncDialect
  MLIRLoopLikeInterface
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  clangAST
  )
