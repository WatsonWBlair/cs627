#!/usr/bin/env python
"""
Validation Script for Evaluation Pipeline

This script validates that all evaluation components work correctly:
1. Import validation for all modules
2. Basic functionality tests
3. Dependency checks
4. Quick smoke tests

Usage:
    python scripts/validate_evaluation_pipeline.py
    
    # Verbose output
    python scripts/validate_evaluation_pipeline.py --verbose
    
    # Test specific components
    python scripts/validate_evaluation_pipeline.py --components encoders evaluators
"""

import os
import sys
import argparse
import importlib
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Define components to test
VALIDATION_TESTS = {
    'dependencies': {
        'description': 'Check required dependencies',
        'modules': [
            'torch',
            'transformers',
            'datasets',
            'numpy',
            'pandas',
            'sklearn',
            'matplotlib',
            'seaborn',
            'tqdm',
            'evaluate',
            'sentence_transformers'
        ]
    },
    'encoders': {
        'description': 'Validate encoder modules',
        'modules': [
            'src.Encoders.text.semantic_to_vec',
            'src.Encoders.audio.waveform_to_vec',
            'src.Encoders.image.visual_to_vec'
        ],
        'classes': [
            ('src.Encoders', 'Text_to_Vec'),
            ('src.Encoders', 'Audio_to_Vec'),
            ('src.Encoders', 'Image_to_Vec')
        ]
    },
    'decoders': {
        'description': 'Validate decoder modules',
        'modules': [
            'src.Decoders.text.vec_to_semantic',
            'src.Decoders.audio.vec_to_waveform',
            'src.Decoders.image.vec_to_visual'
        ],
        'classes': [
            ('src.Decoders', 'Vec_to_Text'),
            ('src.Decoders', 'Vec_to_Audio'),
            ('src.Decoders', 'Vec_to_Image')
        ]
    },
    'evaluation_core': {
        'description': 'Validate evaluation metrics',
        'modules': [
            'src.Evaluation.encoder_metrics',
            'src.Evaluation.decoder_metrics',
            'src.Evaluation.cross_modal_evaluator',
            'src.Evaluation.visualization'
        ]
    },
    'benchmark_adapters': {
        'description': 'Validate benchmark adapters',
        'modules': [
            'src.Evaluation.benchmarks.mteb_adapter',
            'src.Evaluation.benchmarks.glue_adapter',
            'src.Evaluation.benchmarks.multibench_adapter'
        ]
    },
    'data_wrangling': {
        'description': 'Validate data wrangling modules',
        'modules': [
            'src.Training.Data_Wrangling.mosi_dataset',
            'src.Training.Data_Wrangling.benchmark_dataset'
        ]
    },
    'training': {
        'description': 'Validate training modules',
        'modules': [
            'src.Training.encoder_trainers',
            'src.Training.decoder_trainers',
            'src.Training.train_encoders',
            'src.Training.train_with_benchmarks'
        ]
    },
    'utils': {
        'description': 'Validate utility modules',
        'modules': [
            'src.utils.Adapter'
        ]
    }
}


def check_import(module_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Check if a module can be imported.
    
    Args:
        module_name: Name of module to import
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if verbose:
            print(f"  Importing {module_name}...", end=" ")
        
        # Handle different import styles
        if '.' in module_name:
            parts = module_name.split('.')
            module = importlib.import_module('.'.join(parts))
        else:
            module = importlib.import_module(module_name)
        
        if verbose:
            print("✓")
        return True, ""
    except ImportError as e:
        error_msg = f"ImportError: {str(e)}"
        if verbose:
            print(f"✗ - {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"✗ - {error_msg}")
        return False, error_msg


def check_class(module_name: str, class_name: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Check if a class can be imported from a module.
    
    Args:
        module_name: Name of module
        class_name: Name of class
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if verbose:
            print(f"  Loading {class_name} from {module_name}...", end=" ")
        
        module = importlib.import_module(module_name)
        
        if hasattr(module, class_name):
            cls = getattr(module, class_name)
            # Try to instantiate if it's a class
            if isinstance(cls, type):
                if verbose:
                    print("✓")
                return True, ""
            else:
                error_msg = f"{class_name} is not a class"
                if verbose:
                    print(f"✗ - {error_msg}")
                return False, error_msg
        else:
            error_msg = f"{class_name} not found in {module_name}"
            if verbose:
                print(f"✗ - {error_msg}")
            return False, error_msg
            
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"✗ - {error_msg}")
        return False, error_msg


def validate_component(component_name: str, config: Dict, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate a specific component.
    
    Args:
        component_name: Name of component
        config: Configuration for component
        verbose: Whether to print verbose output
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Testing: {config['description']}")
    print('='*60)
    
    results = {
        'component': component_name,
        'description': config['description'],
        'modules': {},
        'classes': {},
        'success': True
    }
    
    # Test module imports
    if 'modules' in config:
        for module_name in config['modules']:
            success, error = check_import(module_name, verbose)
            results['modules'][module_name] = {
                'success': success,
                'error': error
            }
            if not success:
                results['success'] = False
    
    # Test class imports
    if 'classes' in config:
        for module_name, class_name in config['classes']:
            key = f"{module_name}.{class_name}"
            success, error = check_class(module_name, class_name, verbose)
            results['classes'][key] = {
                'success': success,
                'error': error
            }
            if not success:
                results['success'] = False
    
    # Print summary
    total_tests = len(results['modules']) + len(results['classes'])
    passed_tests = sum(1 for r in results['modules'].values() if r['success'])
    passed_tests += sum(1 for r in results['classes'].values() if r['success'])
    
    if results['success']:
        print(f"✅ All {total_tests} tests passed")
    else:
        print(f"⚠️  {passed_tests}/{total_tests} tests passed")
        
        # Show failures
        failed_modules = [m for m, r in results['modules'].items() if not r['success']]
        failed_classes = [c for c, r in results['classes'].items() if not r['success']]
        
        if failed_modules:
            print("\nFailed module imports:")
            for module in failed_modules:
                print(f"  ✗ {module}")
                if verbose:
                    print(f"    Error: {results['modules'][module]['error']}")
        
        if failed_classes:
            print("\nFailed class imports:")
            for cls in failed_classes:
                print(f"  ✗ {cls}")
                if verbose:
                    print(f"    Error: {results['classes'][cls]['error']}")
    
    return results


def run_smoke_tests(verbose: bool = False) -> Dict[str, Any]:
    """
    Run basic smoke tests on key components.
    
    Args:
        verbose: Whether to print verbose output
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print("Running Smoke Tests")
    print('='*60)
    
    results = {
        'encoder_test': False,
        'dataset_test': False,
        'metrics_test': False
    }
    
    # Test 1: Can we create an encoder?
    try:
        if verbose:
            print("\n1. Testing encoder instantiation...")
        from src.Encoders import Text_to_Vec
        encoder = Text_to_Vec()
        results['encoder_test'] = True
        print("  ✓ Text encoder instantiation successful")
    except Exception as e:
        print(f"  ✗ Text encoder instantiation failed: {e}")
        if verbose:
            traceback.print_exc()
    
    # Test 2: Can we create a dataset?
    try:
        if verbose:
            print("\n2. Testing dataset creation...")
        from src.Training.Data_Wrangling.benchmark_dataset import BenchmarkDataset
        # Create with empty sources to test the fix
        dataset = BenchmarkDataset(
            data_sources=[],
            split='train',
            modalities=['text']
        )
        results['dataset_test'] = True
        print("  ✓ Benchmark dataset creation successful")
    except Exception as e:
        print(f"  ✗ Benchmark dataset creation failed: {e}")
        if verbose:
            traceback.print_exc()
    
    # Test 3: Can we import metrics?
    try:
        if verbose:
            print("\n3. Testing metrics import...")
        from src.Evaluation.encoder_metrics import compute_cosine_similarity
        import torch
        # Test with dummy tensors
        a = torch.randn(10, 1024)
        b = torch.randn(10, 1024)
        sim = compute_cosine_similarity(a, b)
        results['metrics_test'] = True
        print("  ✓ Metrics computation successful")
    except Exception as e:
        print(f"  ✗ Metrics computation failed: {e}")
        if verbose:
            traceback.print_exc()
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{'='*40}")
    if passed == total:
        print(f"✅ All {total} smoke tests passed!")
    else:
        print(f"⚠️  {passed}/{total} smoke tests passed")
    
    return results


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate evaluation pipeline components')
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output'
    )
    parser.add_argument(
        '--components',
        nargs='+',
        choices=list(VALIDATION_TESTS.keys()) + ['all'],
        default=['all'],
        help='Components to validate'
    )
    parser.add_argument(
        '--smoke-tests',
        action='store_true',
        help='Run smoke tests after validation'
    )
    
    args = parser.parse_args()
    
    # Determine which components to test
    if 'all' in args.components:
        components_to_test = list(VALIDATION_TESTS.keys())
    else:
        components_to_test = args.components
    
    # Header
    print("="*80)
    print("EVALUATION PIPELINE VALIDATION")
    print("="*80)
    print(f"Project root: {project_root}")
    print(f"Python path includes: {sys.path[0]}")
    print(f"Components to test: {', '.join(components_to_test)}")
    
    # Run validation for each component
    all_results = {}
    total_components = len(components_to_test)
    passed_components = 0
    
    for component in components_to_test:
        if component in VALIDATION_TESTS:
            results = validate_component(
                component,
                VALIDATION_TESTS[component],
                args.verbose
            )
            all_results[component] = results
            if results['success']:
                passed_components += 1
    
    # Run smoke tests if requested
    if args.smoke_tests:
        smoke_results = run_smoke_tests(args.verbose)
        all_results['smoke_tests'] = smoke_results
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\nComponents tested: {total_components}")
    print(f"Components passed: {passed_components}")
    print(f"Success rate: {passed_components/total_components*100:.1f}%")
    
    if passed_components == total_components:
        print("\n✅ ALL VALIDATION TESTS PASSED!")
        print("\nThe evaluation pipeline is ready for use.")
        print("\nNext steps:")
        print("1. Install any missing dependencies: pip install -r requirements.txt")
        print("2. Download required data (MOSI, MTEB, GLUE)")
        print("3. Run training: python src/Training/train_with_benchmarks.py")
        print("4. Run evaluation: python scripts/run_full_evaluation.py")
        print("5. Generate figures: python scripts/generate_performance_figures.py")
    else:
        print("\n⚠️ SOME VALIDATION TESTS FAILED")
        print("\nFailed components:")
        for component, results in all_results.items():
            if not results.get('success', True):
                print(f"  - {component}: {results.get('description', 'Unknown')}")
        
        print("\nRecommended actions:")
        print("1. Check Python path and imports")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Ensure you're running from the project root directory")
        print("4. Check the detailed errors above for specific issues")
    
    # Exit code
    return 0 if passed_components == total_components else 1


if __name__ == "__main__":
    sys.exit(main())