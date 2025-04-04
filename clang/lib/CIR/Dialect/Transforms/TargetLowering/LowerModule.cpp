//===--- LowerModule.cpp - Lower CIR Module to a Target -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CodeGenModule.cpp. The queries
// are adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"

#include "CIRLowerContext.h"
#include "LowerFunction.h"
#include "LowerModule.h"
#include "TargetInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/CIR/Target/AArch64.h"
#include "llvm/Support/ErrorHandling.h"

using MissingFeatures = cir::MissingFeatures;
using AArch64ABIKind = cir::AArch64ABIKind;
using X86AVXABILevel = cir::X86AVXABILevel;

namespace cir {

static CIRCXXABI *createCXXABI(LowerModule &CGM) {
  switch (CGM.getCXXABIKind()) {
  case clang::TargetCXXABI::AppleARM64:
  case clang::TargetCXXABI::Fuchsia:
  case clang::TargetCXXABI::GenericAArch64:
  case clang::TargetCXXABI::GenericARM:
  case clang::TargetCXXABI::iOS:
  case clang::TargetCXXABI::WatchOS:
  case clang::TargetCXXABI::GenericMIPS:
  case clang::TargetCXXABI::GenericItanium:
  case clang::TargetCXXABI::WebAssembly:
  case clang::TargetCXXABI::XL:
    return CreateItaniumCXXABI(CGM);
  case clang::TargetCXXABI::Microsoft:
    cir_cconv_unreachable("Windows ABI NYI");
  }

  cir_cconv_unreachable("invalid C++ ABI kind");
}

static std::unique_ptr<TargetLoweringInfo>
createTargetLoweringInfo(LowerModule &LM) {
  const clang::TargetInfo &Target = LM.getTarget();
  const llvm::Triple &Triple = Target.getTriple();

  switch (Triple.getArch()) {
  case llvm::Triple::aarch64_be:
  case llvm::Triple::aarch64: {
    AArch64ABIKind Kind = AArch64ABIKind::AAPCS;
    if (Target.getABI() == "darwinpcs")
      cir_cconv_unreachable("DarwinPCS ABI NYI");
    else if (Triple.isOSWindows())
      cir_cconv_unreachable("Windows ABI NYI");
    else if (Target.getABI() == "aapcs-soft")
      cir_cconv_unreachable("AAPCS-soft ABI NYI");

    return createAArch64TargetLoweringInfo(LM, Kind);
  }
  case llvm::Triple::x86_64: {
    switch (Triple.getOS()) {
    case llvm::Triple::Win32:
      cir_cconv_unreachable("Windows ABI NYI");
    default:
      return createX86_64TargetLoweringInfo(LM, X86AVXABILevel::None);
    }
  }
  case llvm::Triple::spirv64:
    return createSPIRVTargetLoweringInfo(LM);

  case llvm::Triple::nvptx:
  case llvm::Triple::nvptx64:
    return createNVPTXTargetLoweringInfo(LM);

  default:
    cir_cconv_unreachable("ABI NYI");
  }
}

LowerModule::LowerModule(clang::LangOptions langOpts,
                         clang::CodeGenOptions codeGenOpts,
                         mlir::ModuleOp &module,
                         std::unique_ptr<clang::TargetInfo> target,
                         mlir::PatternRewriter &rewriter)
    : context(module, std::move(langOpts), std::move(codeGenOpts)),
      module(module), Target(std::move(target)), ABI(createCXXABI(*this)),
      types(*this), rewriter(rewriter) {
  context.initBuiltinTypes(*Target);
}

const TargetLoweringInfo &LowerModule::getTargetLoweringInfo() {
  if (!TheTargetCodeGenInfo)
    TheTargetCodeGenInfo = createTargetLoweringInfo(*this);
  return *TheTargetCodeGenInfo;
}

void LowerModule::setCIRFunctionAttributes(FuncOp GD,
                                           const LowerFunctionInfo &Info,
                                           FuncOp F, bool IsThunk) {
  unsigned CallingConv;
  // NOTE(cir): The method below will update the F function in-place with the
  // proper attributes.
  constructAttributeList(GD.getName(), Info, GD, F, CallingConv,
                         /*AttrOnCallSite=*/false, IsThunk);
  // TODO(cir): Set Function's calling convention.
}

/// Set function attributes for a function declaration.
///
/// This method is based on CodeGenModule::SetFunctionAttributes but it
/// altered to consider only the ABI/Target-related bits.
void LowerModule::setFunctionAttributes(FuncOp oldFn, FuncOp newFn,
                                        bool IsIncompleteFunction,
                                        bool IsThunk) {

  // TODO(cir): There's some special handling from attributes related to LLVM
  // intrinsics. Should we do that here as well?

  // Setup target-specific attributes.
  if (!IsIncompleteFunction)
    setCIRFunctionAttributes(oldFn, getTypes().arrangeGlobalDeclaration(oldFn),
                             newFn, IsThunk);

  // TODO(cir): Handle attributes for returned "this" objects.

  // NOTE(cir): Skipping some linkage and other global value attributes here as
  // it might be better for CIRGen to handle them.

  // TODO(cir): Skipping section attributes here.

  // TODO(cir): Skipping error attributes here.

  // If we plan on emitting this inline builtin, we can't treat it as a builtin.
  if (MissingFeatures::funcDeclIsInlineBuiltinDeclaration()) {
    cir_cconv_unreachable("NYI");
  }

  if (MissingFeatures::funcDeclIsReplaceableGlobalAllocationFunction()) {
    cir_cconv_unreachable("NYI");
  }

  if (MissingFeatures::funcDeclIsCXXConstructorDecl() ||
      MissingFeatures::funcDeclIsCXXDestructorDecl())
    cir_cconv_unreachable("NYI");
  else if (MissingFeatures::funcDeclIsCXXMethodDecl())
    cir_cconv_unreachable("NYI");

  // NOTE(cir) Skipping emissions that depend on codegen options, as well as
  // sanitizers handling here. Do this in CIRGen.

  if (MissingFeatures::langOpts() && MissingFeatures::openMP())
    cir_cconv_unreachable("NYI");

  // NOTE(cir): Skipping more things here that depend on codegen options.

  if (MissingFeatures::extParamInfo()) {
    cir_cconv_unreachable("NYI");
  }
}

/// Rewrites an existing function to conform to the ABI.
///
/// This method is based on CodeGenModule::EmitGlobalFunctionDefinition but it
/// considerably simplified as it tries to remove any CodeGen related code.
llvm::LogicalResult LowerModule::rewriteFunctionDefinition(FuncOp op) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  // Get ABI/target-specific function information.
  const LowerFunctionInfo &FI = this->getTypes().arrangeGlobalDeclaration(op);

  // Get ABI/target-specific function type.
  FuncType Ty = this->getTypes().getFunctionType(FI);

  // NOTE(cir): Skipping getAddrOfFunction and getOrCreateCIRFunction methods
  // here, as they are mostly codegen logic.

  // Create a new function with the ABI-specific types.
  FuncOp newFn = mlir::cast<FuncOp>(rewriter.cloneWithoutRegions(op));
  newFn.setType(Ty);

  // NOTE(cir): The clone above will preserve any existing attributes. If there
  // are high-level attributes that ought to be dropped, do it here.

  // Set up ABI-specific function attributes.
  setFunctionAttributes(op, newFn, false, /*IsThunk=*/false);
  if (MissingFeatures::extParamInfo()) {
    cir_cconv_unreachable("ExtraAttrs are NYI");
  }

  // Is a function definition: handle the body.
  if (!op.isDeclaration()) {
    if (LowerFunction(*this, rewriter, op, newFn)
            .generateCode(op, newFn, FI)
            .failed())
      return llvm::failure();
  }

  // Erase original ABI-agnostic function.
  rewriter.eraseOp(op);
  return llvm::success();
}

llvm::LogicalResult LowerModule::rewriteFunctionCall(CallOp callOp,
                                                     FuncOp funcOp) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(callOp);

  // Create a new function with the ABI-specific calling convention.
  if (LowerFunction(*this, rewriter, funcOp, callOp)
          .rewriteCallOp(callOp)
          .failed())
    return llvm::failure();

  return llvm::success();
}

// TODO: not to create it every time
std::unique_ptr<LowerModule>
createLowerModule(mlir::ModuleOp module, mlir::PatternRewriter &rewriter) {
  // Fetch target information.
  llvm::Triple triple(mlir::cast<mlir::StringAttr>(
                          module->getAttr(cir::CIRDialect::getTripleAttrName()))
                          .getValue());
  clang::TargetOptions targetOptions;
  targetOptions.Triple = triple.str();
  auto targetInfo = clang::targets::AllocateTarget(triple, targetOptions);

  // FIXME(cir): This just uses the default language options. We need to account
  // for custom options.
  // Create context.
  cir_cconv_assert(!cir::MissingFeatures::langOpts());
  clang::LangOptions langOpts;

  // FIXME(cir): This just uses the default code generation options. We need to
  // account for custom options.
  cir_cconv_assert(!cir::MissingFeatures::codeGenOpts());
  clang::CodeGenOptions codeGenOpts;

  if (auto optInfo = mlir::cast_if_present<cir::OptInfoAttr>(
          module->getAttr(cir::CIRDialect::getOptInfoAttrName()))) {
    codeGenOpts.OptimizationLevel = optInfo.getLevel();
    codeGenOpts.OptimizeSize = optInfo.getSize();
  }

  return std::make_unique<LowerModule>(std::move(langOpts),
                                       std::move(codeGenOpts), module,
                                       std::move(targetInfo), rewriter);
}

} // namespace cir
