.PHONY: help install install-dev test test-unit test-integ lint format type-check clean docs

# Variables
PYTHON := python3
PIP := pip3
PROJECT := simplex-dg-solver
VENV := venv
SRC := src
TESTS := tests

# Colors for output
RESET := \033[0m
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m

# ============================================================================
# HELP TARGET
# ============================================================================
help:
	@echo "$(BOLD)Simplex-DG Solver - Development Tasks$(RESET)"
	@echo "========================================"
	@echo ""
	@echo "$(BOLD)Setup & Installation$(RESET)"
	@echo "  make install        Install in editable mode (no dev tools)"
	@echo "  make install-dev    Install with dev tools + pre-commit hooks"
	@echo ""
	@echo "$(BOLD)Testing$(RESET)"
	@echo "  make test           Run all tests (unit + integration)"
	@echo "  make test-unit      Run unit tests only (with coverage)"
	@echo "  make test-integ     Run integration tests only"
	@echo ""
	@echo "$(BOLD)Code Quality$(RESET)"
	@echo "  make lint           Run all linters (flake8, mypy, pylint)"
	@echo "  make format         Auto-format code (black, isort)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make pre-commit     Run pre-commit checks on all files"
	@echo ""
	@echo "$(BOLD)Maintenance$(RESET)"
	@echo "  make clean          Remove build artifacts and caches"
	@echo "  make clean-all      Clean + remove venv"
	@echo "  make docs           Show documentation location"
	@echo ""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================
install:
	@echo "$(GREEN)Installing $(PROJECT) in editable mode...$(RESET)"
	$(PIP) install -e .

install-dev: install
	@echo "$(GREEN)Installing dev tools...$(RESET)"
	$(PIP) install -e ".[dev,jupyter]"
	@echo "$(GREEN)Setting up pre-commit hooks...$(RESET)"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"
	@echo ""
	@echo "$(BOLD)Quick start:$(RESET)"
	@echo "  make test          # Run tests"
	@echo "  make lint          # Check code quality"
	@echo "  make format        # Auto-format code"

# ============================================================================
# TESTING
# ============================================================================
test:
	@echo "$(GREEN)Running all tests...$(RESET)"
	pytest $(TESTS)/ -v --tb=short

test-unit:
	@echo "$(GREEN)Running unit tests with coverage...$(RESET)"
	pytest $(TESTS)/unit/ -v --tb=short --cov=$(SRC) --cov-report=html
	@echo ""
	@echo "$(YELLOW)Coverage report generated: htmlcov/index.html$(RESET)"

test-integ:
	@echo "$(GREEN)Running integration tests...$(RESET)"
	pytest $(TESTS)/integration/ -v --tb=short

# ============================================================================
# CODE QUALITY
# ============================================================================
lint:
	@echo "$(GREEN)Running linters...$(RESET)"
	@echo "  Checking with black..."
	black --check $(SRC)/ $(TESTS)/
	@echo "  Checking imports with isort..."
	isort --check-only $(SRC)/ $(TESTS)/
	@echo "  Linting with flake8..."
	flake8 $(SRC)/ $(TESTS)/
	@echo "  Type checking with mypy..."
	mypy $(SRC)/
	@echo "$(GREEN)✓ All linters passed!$(RESET)"

format:
	@echo "$(GREEN)Auto-formatting code...$(RESET)"
	black $(SRC)/ $(TESTS)/
	isort $(SRC)/ $(TESTS)/
	@echo "$(GREEN)✓ Code formatted!$(RESET)"

type-check:
	@echo "$(GREEN)Running mypy type checker...$(RESET)"
	mypy $(SRC)/ --show-error-codes

pre-commit:
	@echo "$(GREEN)Running pre-commit on all files...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed!$(RESET)"

# ============================================================================
# MAINTENANCE
# ============================================================================
clean:
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage .eggs/
	@echo "$(GREEN)✓ Cleaned!$(RESET)"

clean-all: clean
	@echo "$(YELLOW)Removing virtual environment...$(RESET)"
	rm -rf $(VENV)/
	@echo "$(GREEN)✓ Full cleanup complete!$(RESET)"

# ============================================================================
# DOCUMENTATION
# ============================================================================
docs:
	@echo "$(BOLD)Project Documentation$(RESET)"
	@echo "======================="
	@echo ""
	@echo "$(BOLD)Comprehensive Reference:$(RESET)"
	@echo "  .github/copilot-instructions.md"
	@echo ""
	@echo "$(BOLD)API Functions:$(RESET)"
	@echo "  See § API Function Reference in .github/copilot-instructions.md"
	@echo ""
	@echo "$(BOLD)Development Workflow:$(RESET)"
	@echo "  See § Development Workflow in .github/copilot-instructions.md"
	@echo ""
	@echo "$(BOLD)Contributing:$(RESET)"
	@echo "  1. Fork the repository"
	@echo "  2. make install-dev"
	@echo "  3. make test && make lint"
	@echo "  4. Submit PR"
	@echo ""

# ============================================================================
# DEFAULT TARGET
# ============================================================================
.DEFAULT_GOAL := help
