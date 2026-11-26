set shell := ["bash", "-euo", "pipefail", "-c"]

# List all available recipes
default:
    @just --list

# Install Python dependencies
requirements:
    uv sync --extra dev --extra plotting --extra docs

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Lint using ruff (use `just format` to do formatting)
lint:
    uv run --extra dev ruff format --check
    uv run --extra dev ruff check

# Format source code with ruff
format:
    uv run --extra dev ruff check --fix
    uv run --extra dev ruff format

# Format and lint using ruff
style: format lint

# Run tests
test:
    uv run --extra dev --extra plotting python -m pytest tests

# Run library tests without slow or example workflows
test-fast:
    uv run --extra dev --extra plotting python -m pytest tests -m 'not slow and not examples'

# Build distributable package artifacts
build:
    uv build

# Check formatting, linting, tests, and package build
check: lint test build

# Regenerate committed example and tutorial docs assets
docs-assets:
    uv run --extra docs python -m topographer.examples._docs

# Build the public website after refreshing committed documentation assets
website: docs-assets
    uv run --extra docs great-docs build
