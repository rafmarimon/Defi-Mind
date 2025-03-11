#!/usr/bin/env python3
"""
Update import statements in Python files after project reorganization.
This script scans Python files in the core directory and updates imports
to reflect the new project structure.
"""

import os
import re
import glob
from pathlib import Path

# List of core modules that have been moved
CORE_MODULES = [
    "dashboard",
    "pyth_searcher",
    "langchain_agent",
    "defimind_persistence",
    "defimind_runner", 
    "machine_learning",
    "trading_strategy",
    "protocol_analytics",
    "live_data_fetcher",
    "yield_scanner"
]

def update_file_imports(file_path):
    """Update import statements in a single file"""
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern for direct imports: `import module_name`
    for module in CORE_MODULES:
        # Don't modify imports of the current file
        if os.path.basename(file_path) == f"{module}.py":
            continue
            
        # Update direct imports
        content = re.sub(
            rf'import\s+{module}(\s+|$)',
            f'import core.{module}\\1',
            content
        )
        
        # Update from imports
        content = re.sub(
            rf'from\s+{module}\s+import',
            f'from core.{module} import',
            content
        )
    
    # Check if content was modified
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated imports in {file_path}")
    else:
        print(f"  No changes needed in {file_path}")

def update_imports():
    """Update import statements in all Python files"""
    # Get all Python files in core directory
    python_files = glob.glob("core/*.py")
    
    for file_path in python_files:
        update_file_imports(file_path)
    
    print("\nImport update complete!")
    print("Please review the changes and test your code to ensure everything works correctly.")

if __name__ == "__main__":
    confirm = input("This will update import statements in Python files. Continue? (y/n): ")
    if confirm.lower() in ["y", "yes"]:
        update_imports()
    else:
        print("Import update cancelled.") 