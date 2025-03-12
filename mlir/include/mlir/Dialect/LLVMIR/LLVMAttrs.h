//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
#define MLIR_DIALECT_LLVMIR_LLVMATTRS_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/OpImplementation.h"
#include <optional>

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"

namespace mlir {
namespace LLVM {

namespace detail {
struct ConstantRangeAttrStorage : public AttributeStorage {
  ConstantRangeAttrStorage(llvm::APInt lower, llvm::APInt upper)
      : lower(lower), upper(upper) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::pair<llvm::APInt, llvm::APInt>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    if (lower.getBitWidth() != key.first.getBitWidth() ||
        upper.getBitWidth() != key.second.getBitWidth()) {
      return false;
    }
    return lower == key.first && upper == key.second;
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(llvm::APInt lower, llvm::APInt upper) {
    return KeyTy(lower, upper);
  }

  /// Define a construction method for creating a new instance of this storage.
  static ConstantRangeAttrStorage *construct(mlir::StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<ConstantRangeAttrStorage>())
        ConstantRangeAttrStorage(key.first, key.second);
  }

  /// Construct an instance of the key from this storage class.
  KeyTy getAsKey() const {
    return KeyTy(lower, upper);
  }

  llvm::APInt getLower() const { return lower; }
  llvm::APInt getUpper() const { return upper; }

  /// The parametric data held by the storage class.
  llvm::APInt lower;
  llvm::APInt upper;
};
}

/// This class represents the base attribute for all debug info attributes.
class DINodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  // Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info scope.
class DIScopeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a local debug info
/// scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info type.
class DITypeAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents a LLVM attribute that describes a debug info variable.
class DIVariableAttr : public DINodeAttr {
public:
  using DINodeAttr::DINodeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// Base class for LLVM attributes participating in the TBAA graph.
class TBAANodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);

  /// Required by DenseMapInfo to create empty and tombstone key.
  static TBAANodeAttr getFromOpaquePointer(const void *pointer) {
    return TBAANodeAttr(reinterpret_cast<const ImplType *>(pointer));
  }
};

// Inline the LLVM generated Linkage enum and utility.
// This is only necessary to isolate the "enum generated code" from the
// attribute definition itself.
// TODO: this shouldn't be needed after we unify the attribute generation, i.e.
// --gen-attr-* and --gen-attrdef-*.
using cconv::CConv;
using tailcallkind::TailCallKind;
using linkage::Linkage;
} // namespace LLVM
} // namespace mlir

#include "mlir/Dialect/LLVMIR/LLVMAttrInterfaces.h.inc"




#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"

#endif // MLIR_DIALECT_LLVMIR_LLVMATTRS_H_
