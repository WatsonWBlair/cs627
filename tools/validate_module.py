#!/usr/bin/env python3
"""
Module Validation Script

Functional testing for production-ready (non-experimental) modules.
Tests that modules can be imported and instantiated without errors.

Usage:
    python tools/validate_module.py src/Encoders/text_2_vec.py
    python tools/validate_module.py --all
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple, Dict


def is_experimental(file_path: Path) -> bool:
    """Check if module is marked as experimental."""
    with open(file_path, 'r') as f:
        content = f.read()
        return 'EXPERIMENTAL' in content[:500]  # Check first 500 chars


def validate_module(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate that a module can be imported and instantiated.

    Args:
        file_path: Path to module file

    Returns:
        (success, errors)
    """
    errors = []

    if not file_path.exists():
        errors.append(f"File not found: {file_path}")
        return False, errors

    # Check if experimental - skip functional tests for experimental modules
    if is_experimental(file_path):
        return True, []  # Experimental modules don't need to pass functional tests

    # Add src directory to Python path for imports
    src_dir = Path("src")
    if src_dir not in sys.path:
        sys.path.insert(0, str(src_dir.absolute()))

    # Try to import the module
    module_name = file_path.stem
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None or spec.loader is None:
        errors.append(f"Failed to load module spec")
        return False, errors

    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        errors.append(f"Import failed: {e}")
        return False, errors

    # Find all encoder/decoder classes (can be multiple per file)
    is_encoder = 'Encoders' in str(file_path)
    is_decoder = 'Decoders' in str(file_path)

    classes = [item for item in dir(module) if not item.startswith('_')]

    # Filter for encoder/decoder classes based on naming convention
    coder_classes = []
    for cls_name in classes:
        obj = getattr(module, cls_name)
        if not isinstance(obj, type):
            continue
        if is_encoder and cls_name.endswith('_to_Vec'):
            coder_classes.append(cls_name)
        elif is_decoder and cls_name.startswith('Vec_to_'):
            coder_classes.append(cls_name)

    if not coder_classes:
        errors.append("No encoder/decoder classes found matching naming convention")
        return False, errors

    # Try to instantiate each class
    for class_name in coder_classes:
        try:
            cls = getattr(module, class_name)
            instance = cls()

            # Check that forward method exists
            if not hasattr(instance, 'forward'):
                errors.append(f"{class_name}: missing forward() method")

        except Exception as e:
            errors.append(f"{class_name}: instantiation failed: {e}")

    success = len(errors) == 0
    return success, errors


def validate_all_modules() -> Dict[str, Tuple[bool, List[str], bool]]:
    """
    Validate all encoder and decoder modules.

    Returns:
        {file_path: (success, errors, is_experimental)}
    """
    results = {}

    # Find all encoder modules
    encoders_dir = Path("src/Encoders")
    if encoders_dir.exists():
        for file in encoders_dir.glob("*_2_vec.py"):
            if file.name != "encoder_boilerplate.py":
                success, errors = validate_module(file)
                results[str(file)] = (success, errors, is_experimental(file))

    # Find all decoder modules
    decoders_dir = Path("src/Decoders")
    if decoders_dir.exists():
        for file in decoders_dir.glob("vec_2_*.py"):
            if file.name != "decoder_boilerplate.py":
                success, errors = validate_module(file)
                results[str(file)] = (success, errors, is_experimental(file))

    return results


def print_results(file_path: str, success: bool, errors: List[str], experimental: bool):
    """Print validation results."""
    if experimental:
        status = "[EXPERIMENTAL]"
    elif success:
        status = "[PASS]"
    else:
        status = "[FAIL]"

    print(f"\n{status}: {file_path}")

    if errors:
        print(f"  Errors ({len(errors)}):")
        for error in errors:
            print(f"    - {error}")
    elif not experimental:
        print("  Module imports and instantiates successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Validate production-ready modules can be imported and instantiated",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single module
  python tools/validate_module.py src/Encoders/text_2_vec.py

  # Validate all modules
  python tools/validate_module.py --all

Note: Experimental modules are skipped (no functional testing required)
        """
    )

    parser.add_argument(
        'file',
        type=Path,
        nargs='?',
        help='Module file to validate'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all encoder and decoder modules'
    )

    args = parser.parse_args()

    if not args.file and not args.all:
        parser.print_help()
        sys.exit(1)

    # Run validation
    if args.all:
        results = validate_all_modules()
        all_success = True

        for file_path, (success, errors, experimental) in results.items():
            print_results(file_path, success, errors, experimental)
            if not experimental:  # Only count non-experimental modules
                all_success = all_success and success

        # Summary
        total = len(results)
        experimental_count = sum(1 for (s, e, exp) in results.values() if exp)
        production_count = total - experimental_count
        passed = sum(1 for (s, e, exp) in results.values() if s and not exp)

        print(f"\n{'=' * 60}")
        print(f"Summary:")
        print(f"  Production modules: {passed}/{production_count} passed")
        print(f"  Experimental modules: {experimental_count} (skipped)")
        print(f"{'=' * 60}")

        sys.exit(0 if all_success else 1)
    else:
        experimental = is_experimental(args.file)
        success, errors = validate_module(args.file)
        print_results(str(args.file), success, errors, experimental)
        sys.exit(0 if success or experimental else 1)


if __name__ == "__main__":
    main()
