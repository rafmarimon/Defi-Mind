#!/usr/bin/env python3
"""
DEFIMIND Project Cleanup Script
This script organizes and cleans up the DEFIMIND project files.
"""

import os
import shutil
import time
from pathlib import Path
import glob

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

def cleanup_project():
    """Main cleanup function"""
    # Create organization directories
    create_dir_if_not_exists("archive")
    create_dir_if_not_exists("archive/logs")
    create_dir_if_not_exists("archive/tests")
    create_dir_if_not_exists("archive/deployment")
    create_dir_if_not_exists("archive/ethereum")
    create_dir_if_not_exists("archive/scripts")
    create_dir_if_not_exists("core")
    
    # 1. Handle empty log files
    log_files = glob.glob("*.log")
    for log_file in log_files:
        # Check if file is empty
        if os.path.getsize(log_file) == 0:
            os.remove(log_file)
            print(f"Removed empty log file: {log_file}")
        else:
            move_file(log_file, "archive/logs")
    
    # 2. Move test files to tests directory
    test_files = glob.glob("test_*.py") + glob.glob("*_test.py")
    for test_file in test_files:
        move_file(test_file, "archive/tests")
    
    # 3. Move deployment related files
    deployment_files = [
        "deploy_dashboard.py",
        "DEPLOYMENT.md",
        "production.env.template"
    ]
    
    for file in deployment_files:
        if os.path.exists(file):
            move_file(file, "archive/deployment")
    
    # 4. Move Ethereum/Hardhat related files
    ethereum_files = [
        "hardhat.config.js",
        "package.json",
        "package-lock.json"
    ]
    
    for file in ethereum_files:
        if os.path.exists(file):
            move_file(file, "archive/ethereum")
    
    # Move directories
    eth_dirs = ["contracts", "scripts", "ignition", "artifacts", "cache", "test"]
    for dir_name in eth_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            try:
                if dir_name == "scripts":
                    # Handle scripts directory specially
                    if not os.path.exists("archive/scripts"):
                        os.makedirs("archive/scripts")
                    
                    # Move JavaScript/Solidity scripts to archive, keep Python scripts
                    for script in os.listdir(dir_name):
                        if script.endswith(".js") or script.endswith(".sol"):
                            src = os.path.join(dir_name, script)
                            dst = os.path.join("archive/scripts", script)
                            shutil.move(src, dst)
                            print(f"Moved: {src} → {dst}")
                else:
                    dest = os.path.join("archive/ethereum", dir_name)
                    shutil.move(dir_name, dest)
                    print(f"Moved directory: {dir_name} → {dest}")
            except Exception as e:
                print(f"Error moving directory {dir_name}: {e}")
    
    # 5. Move core files to core directory
    core_files = [
        "dashboard.py",
        "pyth_searcher.py",
        "langchain_agent.py",
        "defimind_persistence.py",
        "defimind_runner.py",
        "machine_learning.py",
        "trading_strategy.py",
        "protocol_analytics.py",
        "live_data_fetcher.py",
        "yield_scanner.py"
    ]
    
    for file in core_files:
        if os.path.exists(file):
            move_file(file, "core")
    
    # 6. Clean .env files
    env_files = [".env.backup", ".env.testnet", ".env.example"]
    env_dir = "config"
    create_dir_if_not_exists(env_dir)
    
    for env_file in env_files:
        if os.path.exists(env_file):
            move_file(env_file, env_dir)
    
    # Keep .env.template in root directory for easy access
    if os.path.exists(".env"):
        move_file(".env", env_dir)
    
    # 7. Duplicate agent files - consolidate into one directory
    agent_files = [
        "agent_brain.py", 
        "agent_communication.py",
        "autonomous_agent_main.py", 
        "continuous_agent_runner.py",
        "ai_agent.py", 
        "run_agent.py"
    ]
    
    agent_dir = "archive/agent_variants"
    create_dir_if_not_exists(agent_dir)
    
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            move_file(agent_file, agent_dir)
    
    # 8. Update .gitignore to reflect new structure
    with open(".gitignore", "a") as f:
        f.write("\n# Archived files\narchive/\n")
    
    print("\nProject cleanup complete!")
    print("Core files moved to 'core/' directory")
    print("Archived files moved to 'archive/' directory")
    print("Environment files moved to 'config/' directory")
    print("\nYou may need to update your import paths in Python files!")

if __name__ == "__main__":
    confirm = input("This will reorganize your project files. Have you backed up your project first? (y/n): ")
    if confirm.lower() in ["y", "yes"]:
        cleanup_project()
    else:
        print("Cleanup cancelled. Please run backup_project.py first.") 