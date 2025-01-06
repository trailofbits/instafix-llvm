//===- MlirLTOMain.cpp - MLIR Link main -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-lto/MlirLTOMain.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace llvm;


LogicalResult mlir::MlirLTOMain(int argc, char **argv,
                                 DialectRegistry &registry) {
  static cl::OptionCategory ltoCategory("LTO options");

  static cl::list<std::string> inputFilenames(cl::Positional, cl::OneOrMore,
                                              cl::desc("<input mlir files>"),
                                              cl::cat(ltoCategory));

  static cl::opt<std::string> outputFilename(
      "o", cl::desc("Override output filename"), cl::init("-"),
      cl::value_desc("filename"), cl::cat(ltoCategory));

  static ExitOnError exitOnErr;

  InitLLVM y(argc, argv);
  exitOnErr.setBanner(std::string(argv[0]) + ": ");

  cl::HideUnrelatedOptions({&ltoCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "mlir linker\n");

  return success();
}
