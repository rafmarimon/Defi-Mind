#!/usr/bin/env python3
"""
SelfImprovement module for DEFIMIND

This module enables the agent to analyze its own code, identify areas
for improvement, and implement changes.
"""

import os
import sys
import logging
import inspect
import importlib
import traceback
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/self_improvement.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfImprovement")

class SelfImprovement:
    """
    SelfImprovement module that enables the agent to monitor its own performance,
    identify issues, and implement fixes and enhancements.
    """
    
    def __init__(self):
        """Initialize the SelfImprovement module"""
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.improvement_log = []
        self.error_log = []
        self.performance_metrics = {}
        self.improvement_ideas = []
        
        # Ensure logs directory exists
        os.makedirs(os.path.join(self.project_root, "logs"), exist_ok=True)
        
        # Load previous improvements if they exist
        self._load_improvement_history()
        
        logger.info("Self-improvement module initialized")
    
    def _load_improvement_history(self):
        """Load the history of improvements from disk"""
        history_path = os.path.join(self.project_root, "logs", "improvement_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                    self.improvement_log = data.get("improvements", [])
                    self.improvement_ideas = data.get("ideas", [])
                logger.info(f"Loaded {len(self.improvement_log)} previous improvements")
            except Exception as e:
                logger.error(f"Failed to load improvement history: {str(e)}")
    
    def _save_improvement_history(self):
        """Save the history of improvements to disk"""
        history_path = os.path.join(self.project_root, "logs", "improvement_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump({
                    "improvements": self.improvement_log,
                    "ideas": self.improvement_ideas
                }, f, indent=2)
            logger.info("Saved improvement history")
        except Exception as e:
            logger.error(f"Failed to save improvement history: {str(e)}")
    
    def scan_for_errors(self) -> List[Dict[str, Any]]:
        """
        Scans log files for errors and categorizes them by module
        """
        new_errors = []
        log_dir = os.path.join(self.project_root, "logs")
        
        if not os.path.exists(log_dir):
            logger.warning("Logs directory not found")
            return new_errors
        
        for log_file in os.listdir(log_dir):
            if not log_file.endswith('.log'):
                continue
                
            module_name = log_file.replace('.log', '')
            logger.info(f"Scanning {log_file} for errors")
            
            try:
                with open(os.path.join(log_dir, log_file), 'r') as f:
                    log_content = f.read()
                
                # Extract error patterns from logs
                error_matches = re.finditer(
                    r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*? - (ERROR|CRITICAL) - (.*?)(?=\d{4}-\d{2}-\d{2}|\Z)',
                    log_content, re.DOTALL
                )
                
                for match in error_matches:
                    timestamp_str, level, message = match.groups()
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    error_entry = {
                        "timestamp": timestamp.isoformat(),
                        "module": module_name,
                        "level": level,
                        "message": message.strip(),
                        "analyzed": False,
                        "fix_implemented": False
                    }
                    
                    # Check if this is a new error
                    is_new = True
                    for existing_error in self.error_log:
                        if existing_error["message"] == message.strip() and \
                           existing_error["module"] == module_name:
                            is_new = False
                            break
                    
                    if is_new:
                        new_errors.append(error_entry)
                        self.error_log.append(error_entry)
            
            except Exception as e:
                logger.error(f"Error scanning log file {log_file}: {str(e)}")
        
        logger.info(f"Found {len(new_errors)} new errors")
        return new_errors
    
    def analyze_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes an error and determines potential fixes
        """
        logger.info(f"Analyzing error in {error['module']}: {error['message'][:100]}...")
        
        analysis = {
            "error": error,
            "root_cause": None,
            "affected_components": [],
            "potential_fixes": [],
            "implementation_difficulty": "unknown",
            "priority": "medium"
        }
        
        # Extract more details about the error
        if "Traceback" in error["message"]:
            # This is a Python exception with traceback
            analysis["error_type"] = "exception"
            
            # Extract file and line information
            file_match = re.search(r'File "([^"]+)", line (\d+)', error["message"])
            if file_match:
                file_path, line_number = file_match.groups()
                analysis["file_path"] = file_path
                analysis["line_number"] = int(line_number)
                
                # Get relative path
                if file_path.startswith(self.project_root):
                    analysis["relative_path"] = os.path.relpath(file_path, self.project_root)
                else:
                    analysis["relative_path"] = file_path
            
            # Extract exception type
            exception_match = re.search(r'(\w+Error|Exception):\s*(.+)', error["message"])
            if exception_match:
                exception_type, exception_message = exception_match.groups()
                analysis["exception_type"] = exception_type
                analysis["exception_message"] = exception_message.strip()
                
                # Set priority based on exception type
                if exception_type in ["ImportError", "ModuleNotFoundError"]:
                    analysis["root_cause"] = "missing_dependency"
                    analysis["priority"] = "high"
                    analysis["potential_fixes"].append({
                        "type": "install_dependency",
                        "description": f"Install the missing module mentioned in the error"
                    })
                elif exception_type in ["KeyError", "IndexError", "AttributeError"]:
                    analysis["root_cause"] = "data_access"
                    analysis["priority"] = "high"
                    analysis["potential_fixes"].append({
                        "type": "code_fix",
                        "description": "Add proper validation before accessing the data"
                    })
                elif exception_type in ["TypeError", "ValueError"]:
                    analysis["root_cause"] = "type_mismatch"
                    analysis["priority"] = "high"
                    analysis["potential_fixes"].append({
                        "type": "code_fix",
                        "description": "Fix type conversion or validation issue"
                    })
        elif "ModuleNotFoundError" in error["message"] or "ImportError" in error["message"]:
            analysis["error_type"] = "import_error"
            analysis["root_cause"] = "missing_dependency"
            analysis["priority"] = "high"
            
            # Extract the module name
            module_match = re.search(r'No module named \'([^\']+)\'', error["message"])
            if module_match:
                missing_module = module_match.group(1)
                analysis["missing_module"] = missing_module
                analysis["potential_fixes"].append({
                    "type": "install_dependency",
                    "description": f"Install the missing module: {missing_module}",
                    "command": f"pip install {missing_module}"
                })
        else:
            analysis["error_type"] = "general"
            
        # Mark as analyzed
        error["analyzed"] = True
        
        logger.info(f"Error analysis complete with {len(analysis['potential_fixes'])} potential fixes")
        return analysis
    
    def implement_fix(self, analysis: Dict[str, Any]) -> bool:
        """
        Implements a fix for the analyzed error
        Returns True if fix was successfully implemented
        """
        if not analysis["potential_fixes"]:
            logger.warning("No potential fixes available for this error")
            return False
        
        logger.info(f"Implementing fix for error in {analysis['error']['module']}")
        
        # Choose the first potential fix for now
        # In a more advanced version, we could score and select the best fix
        fix = analysis["potential_fixes"][0]
        success = False
        
        try:
            if fix["type"] == "install_dependency" and "command" in fix:
                # Install missing dependency
                logger.info(f"Installing dependency using: {fix['command']}")
                result = subprocess.run(
                    fix["command"].split(),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed dependency: {result.stdout}")
                    success = True
                else:
                    logger.error(f"Failed to install dependency: {result.stderr}")
            
            elif fix["type"] == "code_fix" and "file_path" in analysis and "line_number" in analysis:
                # For code fixes, we'd need more sophisticated handling
                # For now, we'll just log the suggested fix
                logger.info(f"Code fix required in {analysis['file_path']} at line {analysis['line_number']}")
                logger.info(f"Suggested fix: {fix['description']}")
                
                # Add to improvement ideas for manual implementation
                self.improvement_ideas.append({
                    "timestamp": datetime.now().isoformat(),
                    "file_path": analysis['file_path'],
                    "line_number": analysis['line_number'],
                    "description": fix['description'],
                    "status": "pending"
                })
                
                # Save ideas
                self._save_improvement_history()
                
                # We'll count this as partial success
                success = False
        
        except Exception as e:
            logger.error(f"Error implementing fix: {str(e)}")
            success = False
        
        if success:
            # Update the error record
            analysis["error"]["fix_implemented"] = True
            
            # Add to improvement log
            self.improvement_log.append({
                "timestamp": datetime.now().isoformat(),
                "module": analysis["error"]["module"],
                "error_message": analysis["error"]["message"][:100] + "...",
                "fix_description": fix["description"],
                "successful": True
            })
            
            # Save the update
            self._save_improvement_history()
        
        return success
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """
        Collects performance metrics from various components
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Collect memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            metrics["memory_usage_mb"] = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            logger.warning("psutil not available, skipping memory metrics")
        
        # Try to collect metrics from each component
        core_dir = os.path.join(self.project_root, "core")
        if os.path.exists(core_dir):
            for file_name in os.listdir(core_dir):
                if file_name.endswith('.py') and not file_name.startswith('__'):
                    module_name = file_name[:-3]
                    
                    try:
                        # Try to import the module and check if it has a get_metrics method
                        module_path = f"core.{module_name}"
                        module = importlib.import_module(module_path)
                        
                        if hasattr(module, 'get_metrics'):
                            component_metrics = module.get_metrics()
                            metrics["components"][module_name] = component_metrics
                            logger.info(f"Collected metrics from {module_name}")
                    except Exception as e:
                        logger.warning(f"Failed to collect metrics from {module_name}: {str(e)}")
        
        self.performance_metrics = metrics
        return metrics
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """
        Analyzes performance metrics and suggests improvements
        """
        suggestions = []
        
        # Check if we have enough data to make suggestions
        if not self.performance_metrics:
            logger.warning("No performance metrics available for improvement suggestions")
            return suggestions
        
        # Check memory usage
        if "memory_usage_mb" in self.performance_metrics:
            memory_usage = self.performance_metrics["memory_usage_mb"]
            if memory_usage > 500:  # More than 500MB
                suggestions.append({
                    "area": "memory_usage",
                    "issue": f"High memory usage: {memory_usage:.2f} MB",
                    "suggestion": "Review code for memory leaks or optimize data structures",
                    "priority": "medium"
                })
        
        # Check component-specific metrics
        for component, metrics in self.performance_metrics.get("components", {}).items():
            # Response time suggestions
            if "avg_response_time" in metrics:
                response_time = metrics["avg_response_time"]
                if response_time > 1.0:  # More than 1 second
                    suggestions.append({
                        "area": f"{component}.response_time",
                        "issue": f"Slow response time in {component}: {response_time:.2f}s",
                        "suggestion": "Optimize algorithms or add caching",
                        "priority": "high" if response_time > 5.0 else "medium"
                    })
            
            # API call failures
            if "api_call_failures" in metrics and "total_api_calls" in metrics:
                failure_rate = metrics["api_call_failures"] / metrics["total_api_calls"] if metrics["total_api_calls"] > 0 else 0
                if failure_rate > 0.05:  # More than 5% failure rate
                    suggestions.append({
                        "area": f"{component}.api_reliability",
                        "issue": f"High API failure rate in {component}: {failure_rate:.2%}",
                        "suggestion": "Implement better error handling and retries",
                        "priority": "high" if failure_rate > 0.2 else "medium"
                    })
            
            # Cache hit rate
            if "cache_hits" in metrics and "cache_misses" in metrics:
                total_cache_accesses = metrics["cache_hits"] + metrics["cache_misses"]
                hit_rate = metrics["cache_hits"] / total_cache_accesses if total_cache_accesses > 0 else 0
                if hit_rate < 0.7:  # Less than 70% hit rate
                    suggestions.append({
                        "area": f"{component}.caching",
                        "issue": f"Low cache hit rate in {component}: {hit_rate:.2%}",
                        "suggestion": "Adjust cache size or caching strategy",
                        "priority": "low"
                    })
        
        # Add any suggestions to our improvement ideas
        for suggestion in suggestions:
            if suggestion not in self.improvement_ideas:
                idea = {
                    "timestamp": datetime.now().isoformat(),
                    "area": suggestion["area"],
                    "description": f"{suggestion['issue']}. {suggestion['suggestion']}",
                    "priority": suggestion["priority"],
                    "status": "pending"
                }
                self.improvement_ideas.append(idea)
        
        # Save updated ideas
        self._save_improvement_history()
        
        return suggestions
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the self-improvement system
        """
        return {
            "total_errors_identified": len(self.error_log),
            "errors_fixed": len([e for e in self.error_log if e.get("fix_implemented", False)]),
            "pending_improvements": len([i for i in self.improvement_ideas if i.get("status") == "pending"]),
            "implemented_improvements": len([i for i in self.improvement_ideas if i.get("status") == "implemented"]),
            "performance_metrics": self.performance_metrics,
            "recent_improvements": self.improvement_log[-5:] if self.improvement_log else []
        }

    def run_improvement_cycle(self) -> Dict[str, Any]:
        """
        Runs a complete improvement cycle:
        1. Scan for errors
        2. Analyze errors
        3. Implement fixes
        4. Collect performance metrics
        5. Suggest improvements
        
        Returns a summary of actions taken
        """
        start_time = datetime.now()
        results = {
            "cycle_start": start_time.isoformat(),
            "errors_found": 0,
            "errors_analyzed": 0,
            "fixes_implemented": 0,
            "improvement_suggestions": 0,
            "actions": []
        }
        
        try:
            # Step 1: Scan for errors
            logger.info("Starting error scan")
            new_errors = self.scan_for_errors()
            results["errors_found"] = len(new_errors)
            results["actions"].append({"action": "scan_for_errors", "found": len(new_errors)})
            
            # Step 2: Analyze errors
            logger.info("Analyzing errors")
            analyses = []
            for error in new_errors:
                analysis = self.analyze_error(error)
                analyses.append(analysis)
                results["errors_analyzed"] += 1
            
            results["actions"].append({"action": "analyze_errors", "analyzed": len(analyses)})
            
            # Step 3: Implement fixes
            logger.info("Implementing fixes")
            for analysis in analyses:
                if analysis["potential_fixes"]:
                    success = self.implement_fix(analysis)
                    if success:
                        results["fixes_implemented"] += 1
                        results["actions"].append({
                            "action": "implemented_fix",
                            "for": analysis["error"]["module"],
                            "fix_type": analysis["potential_fixes"][0]["type"]
                        })
            
            # Step 4: Collect performance metrics
            logger.info("Collecting performance metrics")
            metrics = self.collect_performance_metrics()
            results["actions"].append({"action": "collect_metrics", "components": len(metrics.get("components", {}))})
            
            # Step 5: Suggest improvements
            logger.info("Suggesting improvements")
            suggestions = self.suggest_improvements()
            results["improvement_suggestions"] = len(suggestions)
            results["actions"].append({"action": "suggest_improvements", "suggestions": len(suggestions)})
            
        except Exception as e:
            logger.error(f"Error in improvement cycle: {str(e)}")
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        # Finalize results
        end_time = datetime.now()
        results["cycle_end"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"Improvement cycle completed in {results['duration_seconds']:.2f} seconds")
        return results


# For testing
if __name__ == "__main__":
    improver = SelfImprovement()
    result = improver.run_improvement_cycle()
    print(json.dumps(result, indent=2))
    
    status = improver.get_improvement_status()
    print("\nCurrent Improvement Status:")
    print(json.dumps(status, indent=2)) 