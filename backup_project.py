#!/usr/bin/env python3
"""
Simple backup script for DEFIMIND project
Creates a timestamped ZIP backup of the entire project
"""

import os
import zipfile
import datetime
import shutil

def backup_project():
    """Create a backup of the entire project directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"defimind_backup_{timestamp}"
    
    # Create backup directory if it doesn't exist
    if not os.path.exists("backups"):
        os.makedirs("backups")
    
    # Create ZIP file
    zip_path = os.path.join("backups", f"{backup_name}.zip")
    
    print(f"Creating backup: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk('.'):
            # Skip venv, node_modules, and other large directories
            if ('venv' in dirs):
                dirs.remove('venv')
            if ('node_modules' in dirs):
                dirs.remove('node_modules')
            if ('__pycache__' in dirs):
                dirs.remove('__pycache__')
            if ('.git' in dirs):
                dirs.remove('.git')
            if ('backups' in dirs):
                dirs.remove('backups')
            if ('cache' in dirs):
                dirs.remove('cache')
            if ('artifacts' in dirs):
                dirs.remove('artifacts')
                
            # Add files to zip
            for file in files:
                if file.endswith('.zip'):
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.join(root, file)[2:]  # Remove ./ from the beginning
                try:
                    zipf.write(file_path, arcname)
                except Exception as e:
                    print(f"Error adding {file_path}: {e}")
    
    print(f"Backup created successfully at: {zip_path}")
    return zip_path

if __name__ == "__main__":
    backup_project() 