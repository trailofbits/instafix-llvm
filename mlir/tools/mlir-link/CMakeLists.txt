set(LLVM_OPTIONAL_SOURCES
  null.cpp
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

# TODO: No idea what libs are required
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  ${test_libs}

  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  # TODO: Remove when registerAllGPUToLLVMIRTranslations is no longer
  # registered directly in mlir-link.cpp.
  MLIRToLLVMIRTranslationRegistration
  )

# Exclude from libMLIR.so because this has static options intended for
# opt-like tools only.
# add_mlir_library(MLIRMlirOptMain
#   mlir-link.cpp

#   EXCLUDE_FROM_LIBMLIR

#   LINK_LIBS PUBLIC
#   ${LIBS}
#   )

add_mlir_tool(mlir-link
  mlir-link.cpp

  DEPENDS
  ${LIBS}
  SUPPORT_PLUGINS
  )
target_link_libraries(mlir-link PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-link)

mlir_check_all_link_libraries(mlir-link)
export_executable_symbols_for_plugins(mlir-link)
