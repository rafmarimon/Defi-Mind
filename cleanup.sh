#!/bin/bash
# DEFIMIND Project Cleanup Script

echo "========== DEFIMIND Project Cleanup =========="
echo "This script will clean up and organize your DEFIMIND project."
echo "It will perform the following steps:"
echo "1. Back up the current project state"
echo "2. Clean up and organize files"
echo "3. Update import statements"
echo ""
echo "Make sure you have committed any important changes to git before proceeding."
echo ""

read -p "Continue with cleanup? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Cleanup cancelled."
    exit 1
fi

# Step 1: Back up the project
echo ""
echo "Step 1: Backing up the project..."
python backup_project.py
if [ $? -ne 0 ]; then
    echo "Backup failed. Aborting cleanup."
    exit 1
fi

# Step 2: Reorganize files
echo ""
echo "Step 2: Reorganizing project files..."
python cleanup_project.py <<< "y"
if [ $? -ne 0 ]; then
    echo "Cleanup failed. Check error messages above."
    exit 1
fi

# Step 3: Update imports
echo ""
echo "Step 3: Updating import statements..."
python update_imports.py <<< "y"
if [ $? -ne 0 ]; then
    echo "Import update failed. Check error messages above."
    exit 1
fi

# Create an __init__.py file in the core directory to make it a proper package
echo "Creating __init__.py in core directory..."
mkdir -p core
touch core/__init__.py
echo "# DEFIMIND Core Package" > core/__init__.py

echo ""
echo "========== Cleanup Complete =========="
echo "Your project has been reorganized and cleaned up."
echo "Please review the changes and test your code to ensure everything works correctly."
echo ""
echo "To run the dashboard, use: python -m core.dashboard"
echo "To run the agent, use: python -m core.defimind_runner run"
echo ""
echo "See PROJECT_STRUCTURE.md for more details on the new organization." 