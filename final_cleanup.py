#!/usr/bin/env python3
"""
DEFIMIND Final Cleanup Script
This script handles additional files that need organization beyond the initial cleanup.
"""

import os
import shutil
import time
from pathlib import Path

def create_dir_if_not_exists(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def move_file(source, dest_dir):
    """Move a file to destination directory"""
    dest_path = os.path.join(dest_dir, os.path.basename(source))
    
    # Check if destination exists
    if os.path.exists(dest_path):
        # Append timestamp to avoid overwriting
        filename, ext = os.path.splitext(os.path.basename(source))
        timestamp = int(time.time())
        dest_path = os.path.join(dest_dir, f"{filename}_{timestamp}{ext}")
    
    try:
        shutil.move(source, dest_path)
        print(f"Moved: {source} → {dest_path}")
    except Exception as e:
        print(f"Error moving {source}: {e}")

def final_cleanup():
    """Handle additional files beyond the initial cleanup"""
    # Create additional archive directories
    create_dir_if_not_exists("archive/showcase")
    create_dir_if_not_exists("archive/legacy_agents")
    create_dir_if_not_exists("archive/schemas")
    
    # 1. Organize showcase and demo files
    showcase_files = [
        "alchemy_api_showcase.py",
    ]
    
    for file in showcase_files:
        if os.path.exists(file):
            move_file(file, "archive/showcase")
    
    # 2. Move legacy agent implementations
    legacy_agents = [
        "defimind_daemon.py",
        "defimind_unified.py"
    ]
    
    for file in legacy_agents:
        if os.path.exists(file):
            move_file(file, "archive/legacy_agents")
    
    # 3. Move JSON schema files
    schema_files = [
        "protocol_configs.json"
    ]
    
    for file in schema_files:
        if os.path.exists(file):
            move_file(file, "archive/schemas")
    
    # 4. Clean up .DS_Store files
    ds_store_files = [".DS_Store"]
    for file in ds_store_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    print("\nFinal cleanup complete!")
    print("Additional files have been organized.")

if __name__ == "__main__":
    confirm = input("This will organize additional files. Continue? (y/n): ")
    if confirm.lower() in ["y", "yes"]:
        final_cleanup()
    else:
        print("Final cleanup cancelled.") 