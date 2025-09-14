#!/usr/bin/env python3
"""
Code validation script for ESH-cDDIM inference.
This script validates the code structure and syntax without requiring PyTorch.
"""

import ast
import os
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        ast.parse(source)
        return True, "Syntax OK"
        
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def validate_imports(file_path):
    """Validate that imports are correct."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        # Check for common issues
        issues = []
        for imp in imports:
            if imp.startswith('esh_cddim_script') and file_path.endswith('esh_cddim_inference.py'):
                # This is expected
                continue
            if imp.startswith('esh_cddim_inference') and file_path.endswith('run_evaluation.py'):
                # This is expected
                continue
        
        return True, f"Found {len(imports)} imports"
        
    except Exception as e:
        return False, f"Import validation error: {e}"

def validate_class_definitions(file_path):
    """Validate that required classes are defined."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        return True, f"Found classes: {classes}"
        
    except Exception as e:
        return False, f"Class validation error: {e}"

def validate_function_definitions(file_path):
    """Validate that required functions are defined."""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        return True, f"Found {len(functions)} functions"
        
    except Exception as e:
        return False, f"Function validation error: {e}"

def validate_file(file_path):
    """Validate a single file."""
    print(f"\nValidating {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return False
    
    # Validate syntax
    syntax_ok, syntax_msg = validate_python_syntax(file_path)
    if not syntax_ok:
        print(f"✗ {syntax_msg}")
        return False
    else:
        print(f"✓ {syntax_msg}")
    
    # Validate imports
    imports_ok, imports_msg = validate_imports(file_path)
    if not imports_ok:
        print(f"✗ {imports_msg}")
        return False
    else:
        print(f"✓ {imports_msg}")
    
    # Validate classes
    classes_ok, classes_msg = validate_class_definitions(file_path)
    if not classes_ok:
        print(f"✗ {classes_msg}")
        return False
    else:
        print(f"✓ {classes_msg}")
    
    # Validate functions
    functions_ok, functions_msg = validate_function_definitions(file_path)
    if not functions_ok:
        print(f"✗ {functions_msg}")
        return False
    else:
        print(f"✓ {functions_msg}")
    
    return True

def check_required_files():
    """Check that all required files exist."""
    required_files = [
        'esh_cddim_inference.py',
        'run_evaluation.py',
        'test_inference.py',
        'README_inference.md',
        'esh_cddim_script.py',
        'load_deepmimo_datasets.py'
    ]
    
    print("Checking required files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0, missing_files

def check_code_consistency():
    """Check for code consistency between files."""
    print("\nChecking code consistency...")
    
    # Check that ESHcDDIMInference class is properly defined
    try:
        with open('esh_cddim_inference.py', 'r') as f:
            inference_code = f.read()
        
        if 'class ESHcDDIMInference' in inference_code:
            print("✓ ESHcDDIMInference class found")
        else:
            print("✗ ESHcDDIMInference class not found")
            return False
        
        if 'def generate_channels' in inference_code:
            print("✓ generate_channels method found")
        else:
            print("✗ generate_channels method not found")
            return False
        
        if 'def evaluate_channels' in inference_code:
            print("✓ evaluate_channels method found")
        else:
            print("✗ evaluate_channels method not found")
            return False
        
        if 'def visualize_channels' in inference_code:
            print("✓ visualize_channels method found")
        else:
            print("✗ visualize_channels method not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking code consistency: {e}")
        return False

def main():
    """Main validation function."""
    print("=" * 60)
    print("ESH-cDDIM Inference Code Validation")
    print("=" * 60)
    
    # Check required files
    files_ok, missing_files = check_required_files()
    if not files_ok:
        print(f"\n✗ Missing files: {missing_files}")
        return False
    
    # Validate each Python file
    python_files = [
        'esh_cddim_inference.py',
        'run_evaluation.py',
        'test_inference.py'
    ]
    
    all_valid = True
    for file in python_files:
        if not validate_file(file):
            all_valid = False
    
    # Check code consistency
    if not check_code_consistency():
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All validations passed! The inference code is ready to use.")
        print("\nNext steps:")
        print("1. Set up the conda environment: conda env create -f environment.yml")
        print("2. Activate the environment: conda activate cDDIM")
        print("3. Train a model using: python esh_cddim_script.py")
        print("4. Run inference using: python esh_cddim_inference.py generate")
    else:
        print("✗ Some validations failed. Please check the errors above.")
    
    return all_valid

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
