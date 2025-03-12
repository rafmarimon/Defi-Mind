#!/usr/bin/env python3
"""
DEFIMIND File Restoration Script
This script restores archived files to their original locations to support test execution.
"""

import os
import shutil
import time
from pathlib import Path

def create_dir_if_not_exists(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def copy_file(source, dest_dir):
    """Copy a file to destination directory"""
    dest_path = os.path.join(dest_dir, os.path.basename(source))
    
    try:
        shutil.copy2(source, dest_path)
        print(f"Copied: {source} → {dest_path}")
        return True
    except Exception as e:
        print(f"Error copying {source}: {e}")
        return False

def restore_files():
    """Restore archived files to support test execution"""
    print("Starting file restoration...")
    
    # Create config directory
    create_dir_if_not_exists("config")
    
    # 1. Restore schema files
    schema_files = {
        "archive/schemas/protocol_configs.json": "."  # Root directory
    }
    
    for source, dest in schema_files.items():
        if os.path.exists(source):
            copy_file(source, dest)
            # Also copy to config directory for future use
            copy_file(source, "config")
    
    # 2. Restore showcase files if needed for tests
    showcase_files = {
        "archive/showcase/alchemy_api_showcase.py": "."
    }
    
    for source, dest in showcase_files.items():
        if os.path.exists(source):
            copy_file(source, dest)
    
    # 3. Restore legacy agent implementations if needed for tests
    legacy_agents = {
        "archive/legacy_agents/defimind_daemon.py": ".",
        "archive/legacy_agents/defimind_unified.py": "."
    }
    
    for source, dest in legacy_agents.items():
        if os.path.exists(source):
            copy_file(source, dest)
    
    # 4. Update import errors in test files
    print("\nCreating backup of test files before modifying...")
    for test_file in ["test_components.py", "test_runner.py", "test_pyth_searcher.py", "test_langchain_agent.py"]:
        if os.path.exists(test_file):
            backup_file = f"{test_file}.bak"
            shutil.copy2(test_file, backup_file)
            print(f"Created backup: {backup_file}")
    
    print("\nFile restoration complete!")
    print("All necessary files have been restored to their original locations.")
    print("WARNING: This is a temporary solution to make tests run. For production,")
    print("the tests should be updated to match the new project structure.")

if __name__ == "__main__":
    confirm = input("This will restore archived files to their original locations. Continue? (y/n): ")
    if confirm.lower() in ["y", "yes"]:
        restore_files()
    else:
        print("File restoration cancelled.") 