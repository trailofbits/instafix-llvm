#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h" 


namespace mlir {

class MLIRMover {
public:
  MLIRMover(MLIRContext *context) : context(context) {}

  LogicalResult moveModule(ModuleOp srcModule, ModuleOp destModule) {
    SymbolTable destSymbolTable(destModule);

    // Walk through all operations in the source module's body
    for (Operation &op : srcModule.getBody()->getOperations()) {
      if (failed(moveOperation(&op, destModule, destSymbolTable))) {
        return failure();
      }
    }
    return success();
  }

private:
  MLIRContext *context;

  /// Get linkage attribute from operation
  IntegerAttr getLinkageAttr(Operation *op) {
    return op->getAttrOfType<IntegerAttr>("global_linkage_kind");
  }

  /// Resolve symbol based on linkage types
  LogicalResult resolveSymbol(Operation *srcOp, Operation *destOp,
                              ModuleOp destModule,
                              SymbolTable &destSymbolTable) {
    auto srcLinkage = getLinkageAttr(srcOp);
    auto destLinkage = getLinkageAttr(destOp);

    if (!srcLinkage || !destLinkage) {
      return srcOp->emitError("missing linkage attribute");
    }

    switch (srcLinkage.getInt()) {
    case 0: // ExternalLinkage
      return handleExternalLinkage(srcOp, destOp);

    case 1: // AvailableExternallyLinkage
      return handleAvailableExternally(srcOp, destOp);

    case 2: // LinkOnceAnyLinkage
      return handleLinkOnceAny(srcOp, destOp, destModule);

    case 3: // LinkOnceODRLinkage
      return handleLinkOnceODR(srcOp, destOp, destModule);

    case 4: // WeakAnyLinkage
      return handleWeakAny(srcOp, destOp);

    case 5: // WeakODRLinkage
      return handleWeakODR(srcOp, destOp);

    case 6: // AppendingLinkage
      return handleAppending(srcOp, destOp, destModule);

    case 7: // InternalLinkage
      return handleInternal(srcOp, destOp, destModule, destSymbolTable);

    case 8: // PrivateLinkage
      return handlePrivate(srcOp, destOp, destModule, destSymbolTable);

    case 9: // ExternalWeakLinkage
      return handleExternalWeak(srcOp, destOp);

    case 10: // CommonLinkage
      return handleCommon(srcOp, destOp);

    default:
      return srcOp->emitError("unknown linkage kind");
    }
  }

  // Handlers for each linkage type
  LogicalResult handleExternalLinkage(Operation *srcOp, Operation *destOp) {
    // If destination is external, replace it
    // If destination is defined, keep destination
    if (isDeclaration(destOp)) {
      replaceOperation(destOp, srcOp);
    }
    return success();
  }

  LogicalResult handleAvailableExternally(Operation *srcOp, Operation *destOp) {
    // Always keep destination, source is just for optimization
    return success();
  }

  LogicalResult handleLinkOnceAny(Operation *srcOp, Operation *destOp,
                                  ModuleOp destModule) {
    // Keep one copy, either is fine
    if (!destOp) {
      cloneAndInsertOperation(srcOp, destModule);
    }
    return success();
  }

  LogicalResult handleLinkOnceODR(Operation *srcOp, Operation *destOp,
                                  ModuleOp destModule) {
    // Keep one copy, must be equivalent
    if (!destOp) {
      cloneAndInsertOperation(srcOp, destModule);
    } else if (!areEquivalent(srcOp, destOp)) {
      return srcOp->emitError("ODR violation: operations not equivalent");
    }
    return success();
  }

  LogicalResult handleWeakAny(Operation *srcOp, Operation *destOp) {
    // Keep destination if it exists
    return success();
  }

  LogicalResult handleWeakODR(Operation *srcOp, Operation *destOp) {
    // Keep destination if it exists, must be equivalent
    if (destOp && !areEquivalent(srcOp, destOp)) {
      return srcOp->emitError("ODR violation: operations not equivalent");
    }
    return success();
  }

  LogicalResult handleAppending(Operation *srcOp, Operation *destOp,
                                ModuleOp destModule) {
    // Merge arrays by appending elements
    if (!destOp) {
      cloneAndInsertOperation(srcOp, destModule);
    } else {
      return appendArrayElements(srcOp, destOp);
    }
    return success();
  }

  LogicalResult handleInternal(Operation *srcOp, Operation *destOp,
                               ModuleOp destModule, SymbolTable &symbolTable) {
    // Convert StringRef to StringAttr
    StringAttr name =
        StringAttr::get(context, cast<SymbolOpInterface>(srcOp).getName());
    std::string newName = (name.getValue() + "_internal").str();
    auto newNameAttr = StringAttr::get(context, newName);

    Operation *clonedOp = srcOp->clone();
    cast<SymbolOpInterface>(clonedOp).setName(newNameAttr);
    destModule.getBody()->getOperations().push_back(clonedOp);

    return success();
  }

  LogicalResult handlePrivate(Operation *srcOp, Operation *destOp,
                              ModuleOp destModule, SymbolTable &symbolTable) {
    StringAttr name =
        StringAttr::get(context, cast<SymbolOpInterface>(srcOp).getName());
    std::string newName = (name.getValue() + "_private").str();
    auto newNameAttr = StringAttr::get(context, newName);

    Operation *clonedOp = srcOp->clone();
    cast<SymbolOpInterface>(clonedOp).setName(newNameAttr);
    // Use erase instead of removeSymbol
    symbolTable.erase(cast<SymbolOpInterface>(clonedOp));
    destModule.getBody()->getOperations().push_back(clonedOp);

    return success();
  }

  LogicalResult handleExternalWeak(Operation *srcOp, Operation *destOp) {
    // Keep destination if it exists, otherwise keep weak external
    if (!destOp) {
      Operation *clonedOp = srcOp->clone();
      setAsWeakExternal(clonedOp);
      return success();
    }
    return success();
  }

  LogicalResult handleCommon(Operation *srcOp, Operation *destOp) {
    // Merge tentative definitions
    if (!destOp) {
      return mergeTentativeDefinitions(srcOp, destOp);
    }
    return success();
  }

  // Utility functions
  bool isDeclaration(Operation *op) {
    // Check if operation is just a declaration (no body)
    if (auto func = dyn_cast<FunctionOpInterface>(op)) {
      return func.isExternal();
    }
    // Add other cases as needed
    return false;
  }

  void replaceOperation(Operation *oldOp, Operation *newOp) {
    Operation *clonedOp = newOp->clone();
    oldOp->replaceAllUsesWith(clonedOp);
    oldOp->erase();
  }

  bool areEquivalent(Operation *op1, Operation *op2) {
    // Implement detailed equivalence checking
    // This should do structural comparison appropriate for ODR
    return false; // Placeholder
  }

  LogicalResult appendArrayElements(Operation *srcOp, Operation *destOp) {
    // Implement array appending logic
    return success(); // Placeholder
  }

  void setAsWeakExternal(Operation *op) {
    // Set appropriate attributes for weak external linkage
  }

  LogicalResult mergeTentativeDefinitions(Operation *srcOp, Operation *destOp) {
    // Implement merging of tentative definitions
    return success(); // Placeholder
  }

  void cloneAndInsertOperation(Operation *op, ModuleOp destModule) {
    Operation *clonedOp = op->clone();
    destModule.getBody()->getOperations().push_back(clonedOp);
  }
  LogicalResult moveOperation(Operation *op, ModuleOp destModule,
                              SymbolTable &destSymbolTable) {
    if (auto symbolOp = dyn_cast<SymbolOpInterface>(op)) {
      StringRef symbolName = symbolOp.getName();
      if (Operation *existing = destSymbolTable.lookup(symbolName)) {
        return resolveSymbol(op, existing, destModule, destSymbolTable);
      }
      // No conflict, just clone and insert
      cloneAndInsertOperation(op, destModule);
    }
    return success();
  }
};

LogicalResult mergeMLIRFiles(StringRef sourceFile, StringRef destFile) {
  // Create MLIRContext and register all dialects
  MLIRContext context;
  registerAllDialects(context);
  context.loadAllAvailableDialects();

  // Source manager for loading files
  llvm::SourceMgr sourceMgr;
  llvm::SourceMgr destMgr;

  // Load source file
  std::string sourceError;
  std::unique_ptr<llvm::MemoryBuffer> sourceBuffer =
      mlir::openInputFile(sourceFile, &sourceError);
  if (!sourceBuffer) {
    llvm::errs() << "failed to open source file '" << sourceFile
                 << "': " << sourceError << "\n";
    return failure();
  }
  sourceMgr.AddNewSourceBuffer(std::move(sourceBuffer), llvm::SMLoc());

  // Load destination file
  std::string destError;
  std::unique_ptr<llvm::MemoryBuffer> destBuffer =
      mlir::openInputFile(destFile, &destError);
  if (!destBuffer) {
    llvm::errs() << "failed to open destination file '" << destFile
                 << "': " << destError << "\n";
    return failure();
  }
  destMgr.AddNewSourceBuffer(std::move(destBuffer), llvm::SMLoc());

  // Parse both modules
  OwningOpRef<ModuleOp> sourceModule =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!sourceModule) {
    llvm::errs() << "failed to parse source module\n";
    return failure();
  }

  OwningOpRef<ModuleOp> destModule =
      parseSourceFile<ModuleOp>(destMgr, &context);
  if (!destModule) {
    llvm::errs() << "failed to parse destination module\n";
    return failure();
  }

  // Create MLIRMover and perform the merge
  MLIRMover mover(&context);
  if (failed(mover.moveModule(*sourceModule, *destModule))) {
    llvm::errs() << "failed to merge modules\n";
    return failure();
  }

  // Write the merged module back to the destination file
  // Write the merged module back to the destination file
  std::string outputError;
  auto output = openOutputFile(destFile, &outputError);
  if (!output) {
    llvm::errs() << "failed to open output file: " << outputError << "\n";
    return failure();
  }

  destModule->print(output->os());
  output->keep(); // Prevent deletion of the output file on destruction

  return success();
}

} // namespace mlir

// Main driver
int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "Usage: " << argv[0] << " <source.mlir> <dest.mlir>\n";
    return 1;
  }

  std::string sourceFile = argv[1];
  std::string destFile = argv[2];

  if (failed(mlir::mergeMLIRFiles(sourceFile, destFile))) {
    return 1;
  }

  return 0;
}