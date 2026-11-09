.PHONY: setup test lint format run-all

# Create .venv and install the package (editable) plus pytest and ruff.
setup:
	uv sync

# Run the unit tests. No network needed: every test uses synthetic data.
test:
	uv run pytest

# Check formatting and lint rules without changing any file.
lint:
	uv run ruff format --check .
	uv run ruff check .

# Auto-format and auto-fix what ruff can fix.
format:
	uv run ruff format .
	uv run ruff check --fix .

# Re-run every experiment. Figures and tables are written to results/.
# The first run downloads prices from Yahoo Finance; later runs use data/cache/.
run-all:
	uv run python experiments/run_rmt.py
	uv run python experiments/run_avellaneda_stoikov.py
	uv run python experiments/run_kalman_pairs.py
	uv run python experiments/run_almgren_chriss.py
