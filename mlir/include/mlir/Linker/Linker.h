//===- Linker.h - MLIR Module Linker ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LINKER_LINKER_H
#define MLIR_LINKER_LINKER_H

#include "mlir/Linker/IRMover.h"

#include "mlir/Interfaces/LinkageInterfaces.h"

namespace mlir::link {

using InternalizeCallbackFn =
    std::function<void(ModuleOp, const StringSet<> &)>;

/// These are gathered alphabetically sorted linker options
class LinkerConfig {
public:
  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  LinkerConfig &allowUnregisteredDialects(bool allow) {
    allowUnregisteredDialectsFlag = allow;
    return *this;
  }
  bool shouldAllowUnregisteredDialects() const {
    return allowUnregisteredDialectsFlag;
  }

  LinkerConfig &internalizeLinkedSymbols(bool allow) {
    internalizeLinkedSymbolsFlag = allow;
    return *this;
  }
  bool shouldInternalizeLinkedSymbols() const {
    return internalizeLinkedSymbolsFlag;
  }

  LinkerConfig &linkOnlyNeeded(bool allow) {
    linkOnlyNeededFlag = allow;
    return *this;
  }
  bool shouldLinkOnlyNeeded() const { return linkOnlyNeededFlag; }

protected:
  /// Allow operation with no registered dialects.
  /// This option is for convenience during testing only and discouraged in
  /// general.
  bool allowUnregisteredDialectsFlag = false;

  bool internalizeLinkedSymbolsFlag = false;

  bool linkOnlyNeededFlag = false;
};

/// This class provides the core functionality of linking in MLIR, it mirrors
/// functionality from `llvm/Linker/Linker.h` for MLIR. It keeps a pointer to
/// the merged module so far. It doesn't take ownership of the module since it
/// is assumed that the user of this class will want to do something with it
/// after the linking.
class Linker {
public:
  enum Flags {
    None = 0,
    OverrideFromSrc = (1 << 0),
    LinkOnlyNeeded = (1 << 1),
  };

  struct LinkFileConfig {
    unsigned flags = Flags::None;
    bool internalize = false;
  };

  Linker(const LinkerConfig &config, MLIRContext *context)
      : config(config), context(context) {}

  MLIRContext *getContext() { return context; }

  LogicalResult linkInModule(OwningOpRef<ModuleOp> src, unsigned flags = None);

  unsigned getFlags() const;

  // Infer how to link file from linker config
  LinkFileConfig linkFileConfig(unsigned fileFlags = None) const;

  /// The first file is linked without internalization and with the
  /// OverrideFromSrc flag set
  LinkFileConfig firstFileConfig(unsigned fileFlags = None) const;

  OwningOpRef<ModuleOp> takeModule() { return std::move(composite); }

  LogicalResult emitFileError(const Twine &fileName, const Twine &message) {
    return emitError("Error processing file '" + fileName + "': " + message);
  }

  LogicalResult emitError(const Twine &message) {
    return mlir::emitError(UnknownLoc::get(context), message);
  }

private:
  const LinkerConfig &config;
  MLIRContext *context;
  OwningOpRef<ModuleOp> composite;
};

} // namespace mlir::link

#endif
