get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

if(MLIR_INCLUDE_TESTS)
  set(test_libs
    )
endif()

set(LIBS
  ${conversion_libs}
  ${dialect_libs}
  ${test_libs}
  MLIRDialect
  MLIRFuncAllExtensions
  MLIRIR
  MLIRLTO
  MLIRLTOLib
  MLIRSupport
  MLIRTensorAllExtensions
  )

add_mlir_tool(mlir-lto
  mlir-lto.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(mlir-lto PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-lto)

mlir_check_all_link_libraries(mlir-lto)
export_executable_symbols_for_plugins(mlir-lto)
