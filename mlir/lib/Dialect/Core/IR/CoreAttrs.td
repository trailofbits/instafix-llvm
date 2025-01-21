#ifndef CORE_ATTRS
#define CORE_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "CoreOps.td"  // This includes the Core_Dialect definition

def Core_Linkage : I32EnumAttr<
  "Linkage",
  "Linkage type for symbols",
  [
    I32EnumAttrCase<"External", 0>,
    I32EnumAttrCase<"AvailableExternally", 1>,
    I32EnumAttrCase<"LinkOnceAny", 2>,
    I32EnumAttrCase<"LinkOnceODR", 3>,
    I32EnumAttrCase<"WeakAny", 4>,
    I32EnumAttrCase<"WeakODR", 5>,
    I32EnumAttrCase<"Appending", 6>,
    I32EnumAttrCase<"Internal", 7>,
    I32EnumAttrCase<"Private", 8>,
    I32EnumAttrCase<"ExternalWeak", 9>,
    I32EnumAttrCase<"Common", 10>
  ]
>;

def LinkageAttr : EnumAttr<Core_Dialect, Core_Linkage, "linkage">;

#endif // CORE_ATTRS