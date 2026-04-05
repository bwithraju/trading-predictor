# Troubleshooting

## API won't start

**Symptom**: `uvicorn main:app` fails immediately.

**Check**:
```bash
python -m pytest tests/ -v   # confirm tests pass
python -c "import main"      # check for import errors
```

Common cause: missing `.env` file. Run `cp .env.example .env`.

---

## `ModuleNotFoundError`

Run `pip install -r requirements.txt` inside your virtual environment.

---

## `yfinance` returns empty data

- Symbol may be wrong (use uppercase, e.g. `AAPL` not `aapl`).
- `period` too short — use `1y` or `2y`.
- yfinance rate-limited — wait a minute and retry.

---

## Prediction returns `NO_SIGNAL`

This is expected behaviour when:
- Model confidence < `CONFIDENCE_THRESHOLD` (default 70%).
- Fewer than `MIN_INDICATOR_ALIGNMENT` indicators agree.
- The model has not been trained yet — call `POST /train` first.

---

## Database errors

```bash
# Reset the database
rm trading_predictor.db
uvicorn main:app --reload   # re-creates tables on startup
```

---

## Docker: port already in use

```bash
lsof -i :8000   # find the process
kill <PID>
docker compose up
```

---

## Tests fail with network errors

Tests mock all external API calls via `tests/conftest.py`. If you see network errors, ensure `conftest.py` is present and that you are running pytest from the project root.
