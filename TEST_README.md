# DEFIMIND Test Suite

This directory contains test scripts for verifying the functionality of DEFIMIND components.

## Overview

The test suite consists of several individual test scripts and a master script that runs all tests:

- `test_components.py`: Tests basic imports and functionality of all core components
- `test_runner.py`: Tests the DEFIMIND runner component
- `test_pyth_searcher.py`: Tests the Pyth SVM searcher component
- `test_langchain_agent.py`: Tests the LangChain agent integration
- `run_all_tests.py`: Master script that runs all tests and generates a summary report

## Running Tests

### Running All Tests

To run all tests at once:

```bash
python run_all_tests.py
```

This will execute all test scripts and generate a summary report in the `test_outputs` directory.

### Running Individual Tests

You can also run individual test scripts:

```bash
python test_components.py
python test_runner.py
python test_pyth_searcher.py
python test_langchain_agent.py
```

## Test Results

Test results are stored in the `test_outputs` directory:

- `master_test_report_YYYYMMDD_HHMMSS.json`: Summary report of all tests
- `component_test_YYYYMMDD_HHMMSS.json`: Detailed results of component tests

## Troubleshooting

If tests fail due to missing files, run the `restore_files.py` script to restore archived files:

```bash
python restore_files.py
```

This script will:
1. Copy necessary configuration files from the archive
2. Restore showcase and legacy agent files needed for testing
3. Create backups of test files before modifying them

## Notes

- These tests are designed to verify the basic functionality of each component
- Some tests may require environment variables to be set (see `.env.template`)
- The tests are designed to run in isolation without external dependencies when possible 