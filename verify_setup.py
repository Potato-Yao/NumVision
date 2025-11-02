"""
Verify NumVision project setup and installation.
"""
import sys
import os

def check_python_version():
    """Check Python version."""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} (Need 3.7+)")
        return False

def check_dependencies():
    """Check if all dependencies are installed."""
    print("\nChecking dependencies...")
    dependencies = {
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'matplotlib': 'matplotlib',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn'
    }
    
    all_ok = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (missing)")
            all_ok = False
    
    return all_ok

def check_project_files():
    """Check if all project files exist."""
    print("\nChecking project files...")
    required_files = [
        'main.py',
        'quickstart.py',
        'demo.py',
        'examples.py',
        'test_model.py',
        'create_test_images.py',
        'requirements.txt',
        'README.md',
        'GETTING_STARTED.md',
        'PROJECT_SUMMARY.md',
        'src/__init__.py',
        'src/model.py',
        'src/train.py',
        'src/predict.py',
        'src/utils.py',
        'src/config.py',
        'src/draw_interface.py',
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (missing)")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if directories exist or can be created."""
    print("\nChecking directories...")
    directories = ['models', 'data', 'tests', 'results']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(directory):
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ (cannot create)")

def run_quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test...")
    
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from src.model import DigitRecognitionModel
        
        print("  Creating model...", end=" ")
        model = DigitRecognitionModel()
        print("✅")
        
        print("  Building architecture...", end=" ")
        model.build_model()
        print("✅")
        
        print("  Compiling model...", end=" ")
        model.compile_model()
        print("✅")
        
        print("  Testing prediction shape...", end=" ")
        import numpy as np
        dummy_input = np.random.random((1, 28, 28, 1))
        predictions = model.model.predict(dummy_input, verbose=0)
        assert predictions.shape == (1, 10)
        print("✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    """Main verification function."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║            NumVision - Setup Verification                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    all_checks = []
    
    # Run all checks
    all_checks.append(check_python_version())
    all_checks.append(check_dependencies())
    all_checks.append(check_project_files())
    check_directories()
    all_checks.append(run_quick_test())
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    if all(all_checks):
        print("\n✅ All checks passed! NumVision is ready to use.")
        print("\nNext steps:")
        print("  1. Train your first model:")
        print("     python quickstart.py")
        print("\n  2. Test with drawing interface:")
        print("     python src/draw_interface.py")
        print("\n  3. Run examples:")
        print("     python examples.py")
        print("\n  4. Read the documentation:")
        print("     - README.md")
        print("     - GETTING_STARTED.md")
        print("     - PROJECT_SUMMARY.md")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

