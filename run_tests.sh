#!/bin/bash
# Script to run ConnectomeBench tests

set -e  # Exit on error

echo "==================================="
echo "ConnectomeBench Test Runner"
echo "==================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing test dependencies..."
    if command -v uv &> /dev/null; then
        echo "Using uv to install test dependencies..."
        uv sync --extra test
    else
        echo "Using pip to install test dependencies..."
        pip install -e ".[test]"
    fi
fi

# Parse command line arguments
MODE="${1:-all}"
VERBOSE="${2:--v}"

case "$MODE" in
    all)
        echo "Running all tests..."
        pytest $VERBOSE
        ;;
    unit)
        echo "Running unit tests only..."
        pytest -m unit $VERBOSE
        ;;
    integration)
        echo "Running integration tests only..."
        pytest -m integration $VERBOSE
        ;;
    fast)
        echo "Running fast tests (excluding slow tests)..."
        pytest -m "not slow" $VERBOSE
        ;;
    coverage)
        echo "Running tests with coverage report..."
        pytest --cov=src --cov-report=html --cov-report=term-missing $VERBOSE
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    watch)
        echo "Running tests in watch mode (requires pytest-watch)..."
        if ! command -v ptw &> /dev/null; then
            echo "Installing pytest-watch..."
            pip install pytest-watch
        fi
        ptw -- $VERBOSE
        ;;
    single)
        if [ -z "$3" ]; then
            echo "Usage: ./run_tests.sh single <test_file_or_function>"
            exit 1
        fi
        echo "Running single test: $3"
        pytest "$3" $VERBOSE
        ;;
    parallel)
        echo "Running tests in parallel..."
        if ! command -v pytest-xdist &> /dev/null; then
            echo "Installing pytest-xdist..."
            pip install pytest-xdist
        fi
        pytest -n auto $VERBOSE
        ;;
    help)
        echo "Usage: ./run_tests.sh [MODE] [VERBOSE]"
        echo ""
        echo "Modes:"
        echo "  all         - Run all tests (default)"
        echo "  unit        - Run only unit tests"
        echo "  integration - Run only integration tests"
        echo "  fast        - Run fast tests (exclude slow tests)"
        echo "  coverage    - Run tests with coverage report"
        echo "  parallel    - Run tests in parallel"
        echo "  watch       - Run tests in watch mode"
        echo "  single      - Run a single test file or function"
        echo "  help        - Show this help message"
        echo ""
        echo "Verbose options:"
        echo "  -v          - Verbose (default)"
        echo "  -vv         - Very verbose"
        echo "  -q          - Quiet"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh"
        echo "  ./run_tests.sh unit -vv"
        echo "  ./run_tests.sh coverage"
        echo "  ./run_tests.sh single tests/test_util.py::TestLLMProcessorInit"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "==================================="
echo "Tests complete!"
echo "==================================="
