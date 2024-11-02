add_mlir_dialect(Linkage linkage)
add_mlir_doc(Linkage Linkage Dialects/ -gen-dialect-doc -dialect=linkage)

# Generate extra headers for custom enum and attrs.
set(LLVM_TARGET_DEFINITIONS Linkage.td)
mlir_tablegen(LinkageEnums.h.inc -gen-enum-decls)
mlir_tablegen(LinkageEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRLinkageEnumsGen)
