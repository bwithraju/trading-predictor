# Contributing

Thank you for your interest in contributing to Trading Predictor!

## Development Setup

```bash
git clone https://github.com/your-org/trading-predictor
cd trading-predictor
bash scripts/setup.sh
source venv/bin/activate
```

## Branching Strategy

- `main` — stable production branch
- `copilot/**` or `feature/**` — feature branches
- Open a pull request against `main`

## Running Tests

```bash
pytest tests/ -v
```

All PRs must pass the test suite before merging.

## Code Style

- Formatter: **black** (`black .`)
- Linter: **flake8** (`flake8 .`)
- Type hints encouraged throughout

Line length is **100** characters (see `.flake8`).

## Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat: add crypto backtest endpoint
fix: handle empty OHLCV response from yfinance
docs: update API reference
```

## Adding a New Endpoint

1. Add Pydantic request/response models in `src/api/endpoints.py`.
2. Implement the route handler.
3. Write tests in `tests/`.
4. Update `docs/API.md`.

## Reporting Issues

Open a GitHub Issue with:
- Python version (`python --version`)
- Steps to reproduce
- Expected vs. actual behaviour
- Relevant log output
