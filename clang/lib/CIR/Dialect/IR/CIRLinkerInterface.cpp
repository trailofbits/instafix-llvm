//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to link CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Interfaces/CIRLinkerInterface.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::link;
using namespace cir;

namespace {
// TODO: expose convertLinkage from LowerToLLVM.cpp
Linkage toLLVMLinkage(GlobalLinkageKind linkage) {
  using CIR = GlobalLinkageKind;
  using LLVM = LLVM::Linkage;

  switch (linkage) {
  case CIR::AvailableExternallyLinkage:
    return LLVM::AvailableExternally;
  case CIR::CommonLinkage:
    return LLVM::Common;
  case CIR::ExternalLinkage:
    return LLVM::External;
  case CIR::ExternalWeakLinkage:
    return LLVM::ExternWeak;
  case CIR::InternalLinkage:
    return LLVM::Internal;
  case CIR::LinkOnceAnyLinkage:
    return LLVM::Linkonce;
  case CIR::LinkOnceODRLinkage:
    return LLVM::LinkonceODR;
  case CIR::PrivateLinkage:
    return LLVM::Private;
  case CIR::WeakAnyLinkage:
    return LLVM::Weak;
  case CIR::WeakODRLinkage:
    return LLVM::WeakODR;
  };
}

// TODO: expose lowerCIRVisibilityToLLVMVisibility from LowerToLLVM.cpp
Visibility toLLVMVisibility(VisibilityKind visibility) {
  using CIR = VisibilityKind;
  using LLVM = LLVM::Visibility;

  switch (visibility) {
  case CIR::Default:
    return LLVM::Default;
  case CIR::Hidden:
    return LLVM::Hidden;
  case CIR::Protected:
    return LLVM::Protected;
  };
}

VisibilityKind toCIRVisibility(Visibility visibility) {
  using CIR = VisibilityKind;
  using LLVM = LLVM::Visibility;

  switch (visibility) {
  case LLVM::Default:
    return CIR::Default;
  case LLVM::Hidden:
    return CIR::Hidden;
  case LLVM::Protected:
    return CIR::Protected;
  };
}

VisibilityAttr toCIRVisibilityAttr(Visibility visibility,
                                                MLIRContext *mlirContext) {
  return VisibilityAttr::get(mlirContext, toCIRVisibility(visibility));
}
} // namespace

bool CIRSymbolLinkerInterface::canBeLinked(Operation *op) const {
  return isa<GlobalOp>(op) || isa<FuncOp>(op);
}

Linkage CIRSymbolLinkerInterface::getLinkage(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return toLLVMLinkage(gv.getLinkage());
  if (auto fn = dyn_cast<FuncOp>(op))
    return toLLVMLinkage(fn.getLinkage());
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isComdat(Operation *op) {
  // TODO(frabert): Extracting comdat info from CIR is not implemented yet
  return false;
}

std::optional<link::ComdatSelector>
CIRSymbolLinkerInterface::getComdatSelector(Operation *op) {
  // TODO(frabert): Extracting comdat info from CIR is not implemented yet
  return std::nullopt;
}

Visibility CIRSymbolLinkerInterface::getVisibility(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return toLLVMVisibility(gv.getGlobalVisibility());
  if (auto fn = dyn_cast<FuncOp>(op))
    return toLLVMVisibility(fn.getGlobalVisibility());
  llvm_unreachable("unexpected operation");
}

void
CIRSymbolLinkerInterface::setVisibility(Operation *op, Visibility visibility) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setGlobalVisibilityAttr(
        toCIRVisibilityAttr(visibility, op->getContext()));
  if (auto fn = dyn_cast<FuncOp>(op))
    return fn.setGlobalVisibilityAttr(
        toCIRVisibilityAttr(visibility, op->getContext()));
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isDeclaration(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.isDeclaration();
  if (auto fn = dyn_cast<FuncOp>(op))
    return fn.isDeclaration();
  llvm_unreachable("unexpected operation");
}

unsigned CIRSymbolLinkerInterface::getBitWidth(Operation *op) {
  llvm_unreachable("NYI");
}

// FIXME: CIR does not yet have UnnamedAddr attribute
UnnamedAddr CIRSymbolLinkerInterface::getUnnamedAddr(Operation * /* op*/) {
  return UnnamedAddr::Global;
}

// FIXME: CIR does not yet have UnnamedAddr attribute
void
CIRSymbolLinkerInterface::setUnnamedAddr(Operation * /* op*/, UnnamedAddr addr) {}

std::optional<uint64_t> CIRSymbolLinkerInterface::getAlignment(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.getAlignment();
  // FIXME: CIR does not (yet?) have alignment for functions
  llvm_unreachable("unexpected operation");
}

void
CIRSymbolLinkerInterface::setAlignment(Operation *op, std::optional<uint64_t> align) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setAlignment(align);
  // FIXME: CIR does not (yet?) have alignment for functions
  llvm_unreachable("unexpected operation");
}

bool CIRSymbolLinkerInterface::isConstant(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.getConstant();
  llvm_unreachable("unexpected operation");
}

void CIRSymbolLinkerInterface::setIsConstant(Operation *op, bool value) {
  if (auto gv = dyn_cast<GlobalOp>(op))
    return gv.setConstant(value);
  llvm_unreachable("constness setting allowed only for globals");
}

bool CIRSymbolLinkerInterface::isGlobalVar(Operation *op) {
  return isa<GlobalOp>(op);
}

llvm::StringRef CIRSymbolLinkerInterface::getSection(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op)) {
    auto section = gv.getSection();
    return section ? section.value() : llvm::StringRef();
  }
  // FIXME: CIR func does not yet have section attribute
  llvm_unreachable("unexpected operation");
}

std::optional<AddressSpace>
CIRSymbolLinkerInterface::getAddressSpace(Operation *op) {
  if (auto gv = dyn_cast<GlobalOp>(op)) {
    if (auto addrSpace = gv.getAddrSpaceAttr()) {
      return addrSpace.getValue();
    }
    return std::nullopt;
  }

  llvm_unreachable("unexpected operation");
}

//===----------------------------------------------------------------------===//
// finalize - Fix cir.get_global types after linking
//===----------------------------------------------------------------------===//

// Walk all GetGlobalOp operations and fix type mismatches.
// This handles the case where a declaration with unspecified parameters
// (e.g., `int foo();` producing `!cir.func<(...) -> !s32i>`) is linked
// with a definition (e.g., `int foo() {}` producing `!cir.func<() -> !s32i>`).
// The GetGlobalOp may still have the declaration's type while the FuncOp
// has the definition's type after linking.

LogicalResult CIRSymbolLinkerInterface::finalize(ModuleOp dst) const {

  SmallVector<GetGlobalOp> toFix;

  dst.walk([&](GetGlobalOp getGlobal) {
    auto *symOp = SymbolTable::lookupNearestSymbolFrom(
        getGlobal, getGlobal.getNameAttr());
    if (!symOp)
      return;

    auto funcOp = dyn_cast<FuncOp>(symOp);
    if (!funcOp)
      return;

    auto expectedPointeeType = funcOp.getFunctionType();

    auto currentPtrType = dyn_cast<PointerType>(getGlobal.getAddr().getType());
    if (!currentPtrType)
      return;

    auto currentPointeeType = currentPtrType.getPointee();

    if (currentPointeeType == expectedPointeeType)
      return;

    toFix.push_back(getGlobal);
  });

  // Fix the mismatched GetGlobalOps
  for (GetGlobalOp getGlobal : toFix) {
    auto *symOp = SymbolTable::lookupNearestSymbolFrom(
        getGlobal, getGlobal.getNameAttr());
    auto funcOp = cast<FuncOp>(symOp);
    auto expectedPointeeType = funcOp.getFunctionType();

    auto currentPtrType = cast<PointerType>(getGlobal.getAddr().getType());

    auto newPtrType = PointerType::get(
        dst.getContext(), expectedPointeeType, currentPtrType.getAddrSpace());

    OpBuilder builder(getGlobal);
    auto newGetGlobal = builder.create<GetGlobalOp>(
        getGlobal.getLoc(),
        newPtrType,
        getGlobal.getName(),
        getGlobal.getTls());

    getGlobal.getAddr().replaceAllUsesWith(newGetGlobal.getAddr());
    getGlobal.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// verifyLinkageCompatibility - Track function signature mismatches
//===----------------------------------------------------------------------===//

// During linking, when two translation units define the same function with
// different signatures (e.g., `extern int bar(int)` vs `char bar(char)`),
// we need to track these mismatches so we can fix call sites later.
//
// This is valid (though questionable) C code that the linker must handle.
// We record the mismatch but don't fail - the actual fix happens in
// fixMismatchedCallSites() after linking completes.

LogicalResult
CIRSymbolLinkerInterface::verifyLinkageCompatibility(Conflict pair) const {
  // Delegate to base class for standard linkage compatibility checks
  auto &mixin =
      static_cast<const SymbolAttrLLVMLinkerInterface<CIRSymbolLinkerInterface>
                      &>(*this);
  if (failed(mixin.LinkerMixin::verifyLinkageCompatibility(pair)))
    return failure();

  // Only track mismatches for function symbols
  auto srcFunc = dyn_cast<FuncOp>(pair.src);
  auto dstFunc = dyn_cast<FuncOp>(pair.dst);
  if (!srcFunc || !dstFunc)
    return success();

  // If the function types differ, record the mismatch for later processing.
  // We don't fail here because this is technically valid C (the linker just
  // picks one definition), but we need to fix up call sites that use the
  // "wrong" signature.
  auto srcType = srcFunc.getFunctionType();
  auto dstType = dstFunc.getFunctionType();
  if (srcType != dstType) {
    mismatchedFunctions[srcFunc.getSymName()] = {srcType, dstType};
    pair.src->emitWarning()
        << "function has mismatched signatures - call sites will be converted "
           "to indirect calls";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// fixMismatchedCallSites - Convert direct calls to indirect calls
//===----------------------------------------------------------------------===//

// When a direct call (e.g., `cir.call @bar(%0)`) targets a function whose
// signature was different in the caller's translation unit, we must convert
// it to an indirect call through a function pointer. This allows the call
// to proceed even though the actual function has a different signature.
//
// Transformation:
//   Before: %r = cir.call @bar(%arg) : (!s32i) -> !s32i
//   After:  %ptr = cir.get_global @bar : !cir.ptr<!cir.func<(!s8i) -> !s8i>>
//           %r = cir.call %ptr(%arg) : (!cir.ptr<!cir.func<...>>, !s32i) -> !s32i
//
// Note: This complements finalize(), which fixes existing cir.get_global ops.
// - finalize() handles: existing indirect calls with wrong get_global type
// - fixMismatchedCallSites() handles: direct calls that need conversion

LogicalResult
CIRSymbolLinkerInterface::fixMismatchedCallSites(ModuleOp module) const {
  if (mismatchedFunctions.empty())
    return success();

  auto result = module->walk([&](CallOp callOp) -> WalkResult {
    // Skip indirect calls - they don't have a callee attribute and are
    // handled by finalize() instead
    auto calleeAttr = callOp.getCalleeAttr();
    if (!calleeAttr)
      return WalkResult::advance();

    // Skip calls to functions that don't have signature mismatches
    StringRef calleeName = calleeAttr.getValue();
    if (mismatchedFunctions.find(calleeName) == mismatchedFunctions.end())
      return WalkResult::advance();

    // Found a direct call to a function with mismatched signature.
    // Convert it to an indirect call through cir.get_global.
    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();

    // Look up the linked function to get its actual (resolved) type
    auto funcOp = dyn_cast<FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(callOp, calleeAttr));
    if (!funcOp) {
      callOp.emitError("cannot find function '") << calleeName << "'";
      return WalkResult::interrupt();
    }

    // Create a cir.get_global to get the function's address.
    // The pointer type uses the linked function's actual type.
    auto linkedFuncType = funcOp.getFunctionType();
    auto ptrType = PointerType::get(module->getContext(), linkedFuncType);
    auto getGlobal = builder.create<GetGlobalOp>(
        loc, ptrType, mlir::FlatSymbolRefAttr::get(funcOp.getSymNameAttr()));

    // Reconstruct the caller's original view of the function type from
    // the call's operand and result types. This preserves the caller's
    // expectations even though the actual function signature differs.
    SmallVector<Type> argTypes;
    for (Value arg : callOp.getArgOperands())
      argTypes.push_back(arg.getType());

    Type resultType = callOp.getNumResults() > 0
                          ? callOp.getResult().getType()
                          : cir::VoidType::get(module->getContext());

    // Preserve variadic status from the linked function
    bool isVarArg = linkedFuncType.isVarArg();
    auto callerFuncType = cir::FuncType::get(argTypes, resultType, isVarArg);

    // Create the indirect call, preserving all attributes from the original
    auto newCall =
        builder.create<CallOp>(loc, getGlobal.getAddr(), callerFuncType,
                               callOp.getArgOperands(), callOp.getCallingConv(),
                               callOp.getSideEffect(), callOp.getExceptionAttr());
    newCall.setExtraAttrsAttr(callOp.getExtraAttrs());

    // Replace the original direct call with our new indirect call
    callOp->replaceAllUsesWith(newCall);
    callOp.erase();
    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// link - Override to fix mismatched call sites after linking
//===----------------------------------------------------------------------===//

// Override the link method to insert our call site fixup pass after the
// standard linking process completes but before verification runs.

LogicalResult CIRSymbolLinkerInterface::link(link::LinkState &state) {
  // First, perform the standard linking process
  if (failed(SymbolAttrLinkerInterface::link(state)))
    return failure();

  // Then fix any direct calls to functions with mismatched signatures
  auto dst = dyn_cast<ModuleOp>(state.getDestinationOp());
  if (!dst)
    return success();

  return fixMismatchedCallSites(dst);
}

//===----------------------------------------------------------------------===//
// registerLinkerInterface
//===----------------------------------------------------------------------===//

void cir::registerLinkerInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, CIRDialect *dialect) {
    dialect->addInterfaces<CIRSymbolLinkerInterface>();
  });
}
