# Makefile for Python TransDecoder

# Variables
PYTHON ?= python3
TEST_DIR = tests
BENCHMARK_DIR = benchmark_results
TEST_DATA_DIR = tests/test_data
INPUT_DATA_DIR = $(TEST_DATA_DIR)/input
EXPECTED_DATA_DIR = $(TEST_DATA_DIR)/expected
SMALL_TEST = transcripts_small.fa
MEDIUM_TEST = transcripts_medium.fa
LARGE_TEST = transcripts_large.fa
MIXED_GC_TEST = transcripts_mixed_gc.fa

# Default target is help
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "─────────────────────────────────────────────────────────────"
	@echo "Development:"
	@echo "  help                 - Show this help message"
	@echo "  install              - Install the package in development mode"
	@echo "  clean                - Remove generated files and directories"
	@echo ""
	@echo "Testing:"
	@echo "  test                 - Run basic tests"
	@echo "  test-detailed        - Run detailed tests with extensive validation"
	@echo "  test-coverage        - Run tests with coverage report"
	@echo "  test-utils           - Test utility functions only"
	@echo "  test-markov          - Test Markov model only"
	@echo "  test-pwm             - Test Position Weight Matrix only"
	@echo "  test-longorfs        - Test LongOrfs extractor only"
	@echo ""
	@echo "Benchmarking:"
	@echo "  generate-test-data   - Generate synthetic test data for benchmarks"
	@echo "  benchmark-small      - Run benchmarks on small dataset"
	@echo "  benchmark-medium     - Run benchmarks on medium dataset"
	@echo "  benchmark-large      - Run benchmarks on large dataset"
	@echo "  benchmark-mixed-gc   - Run benchmarks on mixed GC content dataset"
	@echo "  benchmark-all        - Run all benchmarks"
	@echo ""
	@echo "Reports:"
	@echo "  report-small         - Generate report for small dataset benchmark"
	@echo "  report-medium        - Generate report for medium dataset benchmark"
	@echo "  report-large         - Generate report for large dataset benchmark"
	@echo "  report-mixed-gc      - Generate report for mixed GC content benchmark"
	@echo "  report-all           - Generate all reports"
	@echo "─────────────────────────────────────────────────────────────"

# Install package in development mode
.PHONY: install
install:
	$(PYTHON) -m pip install -e .

# Run all tests
.PHONY: test
test:
	@echo "Running basic utility tests..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_utils.py
	@echo "\nRunning algorithm compatibility tests..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_algorithms.py

# Run detailed tests with verbose output
.PHONY: test-detailed
test-detailed:
	@echo "\n==============================================" 
	@echo "Running detailed tests with extensive validation" 
	@echo "==============================================" 
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_detailed.py

# Run all tests with coverage report
.PHONY: test-coverage
test-coverage:
	$(PYTHON) -m pytest --cov=transdecoder --cov-report=term-missing --cov-report=html -xvs $(TEST_DIR)

# Run components tests individually
.PHONY: test-utils
test-utils:
	@echo "Testing utility functions..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_utils.py

.PHONY: test-markov
test-markov:
	@echo "Testing Markov model implementation..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_detailed.py::TestMarkovModel

.PHONY: test-pwm
test-pwm:
	@echo "Testing Position Weight Matrix implementation..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_detailed.py::TestPWM

.PHONY: test-longorfs
test-longorfs:
	@echo "Testing LongOrfs extraction implementation..."
	$(PYTHON) -m pytest -xvs $(TEST_DIR)/test_detailed.py::TestLongOrfsExtractor

# Generate synthetic test data
.PHONY: generate-test-data
generate-test-data:
	mkdir -p $(INPUT_DATA_DIR)
	$(PYTHON) $(TEST_DIR)/create_test_data.py --output_dir $(INPUT_DATA_DIR) --create_benchmarks

# Run benchmarks on small dataset
.PHONY: benchmark-small
benchmark-small: $(INPUT_DATA_DIR)/$(SMALL_TEST)
	mkdir -p $(BENCHMARK_DIR)/small
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(SMALL_TEST) \
		--output_dir $(BENCHMARK_DIR)/small \
		--min_protein_length 50 \
		--runs 3 \
		--generate_report

# Run benchmarks on medium dataset
.PHONY: benchmark-medium
benchmark-medium: $(INPUT_DATA_DIR)/$(MEDIUM_TEST)
	mkdir -p $(BENCHMARK_DIR)/medium
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(MEDIUM_TEST) \
		--output_dir $(BENCHMARK_DIR)/medium \
		--min_protein_length 50 \
		--runs 3 \
		--generate_report

# Run benchmarks on large dataset
.PHONY: benchmark-large
benchmark-large: $(INPUT_DATA_DIR)/$(LARGE_TEST)
	mkdir -p $(BENCHMARK_DIR)/large
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(LARGE_TEST) \
		--output_dir $(BENCHMARK_DIR)/large \
		--min_protein_length 50 \
		--runs 3 \
		--generate_report

# Run benchmarks on mixed GC content dataset
.PHONY: benchmark-mixed-gc
benchmark-mixed-gc: $(INPUT_DATA_DIR)/$(MIXED_GC_TEST)
	mkdir -p $(BENCHMARK_DIR)/mixed_gc
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(MIXED_GC_TEST) \
		--output_dir $(BENCHMARK_DIR)/mixed_gc \
		--min_protein_length 50 \
		--runs 3 \
		--generate_report

# Generate report for small dataset benchmark
.PHONY: report-small
report-small:
	mkdir -p $(BENCHMARK_DIR)/small/report
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(SMALL_TEST) \
		--generate_report \
		--report_input $(BENCHMARK_DIR)/small/benchmark_results.csv \
		--report_output $(BENCHMARK_DIR)/small/report

# Generate report for medium dataset benchmark
.PHONY: report-medium
report-medium:
	mkdir -p $(BENCHMARK_DIR)/medium/report
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(MEDIUM_TEST) \
		--generate_report \
		--report_input $(BENCHMARK_DIR)/medium/benchmark_results.csv \
		--report_output $(BENCHMARK_DIR)/medium/report

# Generate report for large dataset benchmark
.PHONY: report-large
report-large:
	mkdir -p $(BENCHMARK_DIR)/large/report
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(LARGE_TEST) \
		--generate_report \
		--report_input $(BENCHMARK_DIR)/large/benchmark_results.csv \
		--report_output $(BENCHMARK_DIR)/large/report

# Generate report for mixed GC content benchmark
.PHONY: report-mixed-gc
report-mixed-gc:
	mkdir -p $(BENCHMARK_DIR)/mixed_gc/report
	$(PYTHON) $(TEST_DIR)/benchmark.py \
		--input $(INPUT_DATA_DIR)/$(MIXED_GC_TEST) \
		--generate_report \
		--report_input $(BENCHMARK_DIR)/mixed_gc/benchmark_results.csv \
		--report_output $(BENCHMARK_DIR)/mixed_gc/report

# Rule to ensure test data files exist
$(INPUT_DATA_DIR)/%.fa:
	mkdir -p $(INPUT_DATA_DIR)
	$(PYTHON) $(TEST_DIR)/create_test_data.py --output_dir $(INPUT_DATA_DIR) --create_benchmarks

# Clean up generated files
.PHONY: clean
clean:
	rm -rf $(BENCHMARK_DIR)
	rm -rf $(INPUT_DATA_DIR)
	rm -rf __pycache__
	rm -rf *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

# Run all benchmarks
.PHONY: benchmark-all
benchmark-all: benchmark-small benchmark-medium benchmark-large benchmark-mixed-gc

# Generate all reports
.PHONY: report-all
report-all: report-small report-medium report-large report-mixed-gc