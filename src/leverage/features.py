"""Feature engineering for the leverage prediction model.

Computes technical indicators used as model inputs:
  - Volatility features: ATR, Bollinger Band width, historical volatility,
    volatility percentile, spike detection
  - Trend features: ADX, slope, consistency ratio, multi-timeframe agreement
  - Momentum features: RSI, momentum slope, acceleration
  - Risk features: current drawdown, max drawdown history, Sharpe ratio
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Low-level indicator helpers
# ---------------------------------------------------------------------------


def _true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    """Compute per-bar True Range."""
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    return np.maximum(hl, np.maximum(hc, lc))


def calculate_atr(close: np.ndarray, high: np.ndarray, low: np.ndarray, period: int = 14) -> float:
    """Average True Range (last bar).

    Args:
        close: 1-D array of close prices (oldest first).
        high: 1-D array of high prices.
        low: 1-D array of low prices.
        period: Smoothing period.

    Returns:
        ATR value as float, or *np.nan* when there is insufficient data.
    """
    if len(close) < period + 1:
        return float("nan")

    prev_close = close[:-1]
    tr = _true_range(high[1:], low[1:], prev_close)

    # Wilder's smoothing
    atr = float(np.mean(tr[:period]))
    for value in tr[period:]:
        atr = (atr * (period - 1) + value) / period
    return atr


def calculate_historical_volatility(close: np.ndarray, period: int = 20) -> float:
    """Annualised historical volatility (standard deviation of log-returns).

    Args:
        close: 1-D array of close prices.
        period: Rolling window for std calculation.

    Returns:
        Volatility as a decimal (e.g. 0.25 = 25 %), or *np.nan*.
    """
    if len(close) < period + 1:
        return float("nan")

    log_returns = np.diff(np.log(close))
    if len(log_returns) < period:
        return float("nan")

    return float(np.std(log_returns[-period:]) * np.sqrt(252))


def calculate_bollinger_band_width(close: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
    """Bollinger Band width as a fraction of the middle band.

    Args:
        close: 1-D array of close prices.
        period: Moving average window.
        num_std: Number of standard deviations for the bands.

    Returns:
        Band width ratio, or *np.nan*.
    """
    if len(close) < period:
        return float("nan")

    window = close[-period:]
    ma = float(np.mean(window))
    std = float(np.std(window))
    if ma == 0:
        return float("nan")
    return (num_std * 2 * std) / ma


def calculate_volatility_percentile(close: np.ndarray, period: int = 20, lookback: int = 100) -> float:
    """Percentile rank of current volatility vs. lookback period.

    Returns a value in [0, 1].
    """
    if len(close) < lookback + period:
        return float("nan")

    recent_vols: list[float] = []
    for i in range(lookback):
        start = len(close) - lookback - period + i
        end = len(close) - lookback + i + period
        v = calculate_historical_volatility(close[start:end], period)
        if not np.isnan(v):
            recent_vols.append(v)

    if not recent_vols:
        return float("nan")

    current_vol = calculate_historical_volatility(close[-(period + 1):], period)
    if np.isnan(current_vol):
        return float("nan")

    return float(np.mean(np.array(recent_vols) <= current_vol))


def detect_volatility_spike(close: np.ndarray, period: int = 20, spike_threshold: float = 2.0) -> bool:
    """Return *True* if current volatility is *spike_threshold* × its recent average."""
    if len(close) < period * 2 + 1:
        return False

    current_vol = calculate_historical_volatility(close[-(period + 1):], period)
    avg_vol = calculate_historical_volatility(close[-(period * 2 + 1):], period * 2)
    if np.isnan(current_vol) or np.isnan(avg_vol) or avg_vol == 0:
        return False
    return bool(current_vol / avg_vol >= spike_threshold)


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Average Directional Index (last bar).

    Returns a value in [0, 100]; higher = stronger trend.
    """
    if len(close) < period * 2 + 1:
        return float("nan")

    prev_high = high[:-1]
    prev_low = low[:-1]
    prev_close = close[:-1]
    curr_high = high[1:]
    curr_low = low[1:]

    dm_plus = np.where(
        (curr_high - prev_high) > (prev_low - curr_low),
        np.maximum(curr_high - prev_high, 0),
        0,
    ).astype(float)
    dm_minus = np.where(
        (prev_low - curr_low) > (curr_high - prev_high),
        np.maximum(prev_low - curr_low, 0),
        0,
    ).astype(float)

    tr = _true_range(curr_high, curr_low, prev_close)

    # Wilder smooth
    def wilder_smooth(arr: np.ndarray, n: int) -> np.ndarray:
        result = np.empty_like(arr)
        result[:n] = np.nan
        result[n] = np.mean(arr[:n])
        for idx in range(n + 1, len(arr)):
            result[idx] = result[idx - 1] - result[idx - 1] / n + arr[idx]
        return result

    atr_smooth = wilder_smooth(tr, period)
    dmp_smooth = wilder_smooth(dm_plus, period)
    dmm_smooth = wilder_smooth(dm_minus, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_plus = np.where(atr_smooth > 0, 100 * dmp_smooth / atr_smooth, 0.0)
        di_minus = np.where(atr_smooth > 0, 100 * dmm_smooth / atr_smooth, 0.0)
        dx = np.where(
            (di_plus + di_minus) > 0,
            100 * np.abs(di_plus - di_minus) / (di_plus + di_minus),
            0.0,
        )

    valid_dx = dx[~np.isnan(dx)]
    if len(valid_dx) < period:
        return float("nan")

    adx = float(np.mean(valid_dx[-period:]))
    return adx


def calculate_trend_slope(close: np.ndarray, period: int = 20) -> float:
    """Linear regression slope of close prices over *period* bars, normalised by price.

    Positive = uptrend, negative = downtrend.
    """
    if len(close) < period:
        return float("nan")

    window = close[-period:]
    x = np.arange(period, dtype=float)
    slope, _ = np.polyfit(x, window, 1)
    return float(slope / window[0]) if window[0] != 0 else float("nan")


def calculate_trend_consistency(close: np.ndarray, period: int = 20) -> float:
    """Fraction of bars in *period* that move in the dominant direction.

    Returns a value in [0, 1].
    """
    if len(close) < period + 1:
        return float("nan")

    returns = np.diff(close[-period - 1:])
    pos = float(np.sum(returns > 0))
    neg = float(np.sum(returns < 0))
    total = pos + neg
    if total == 0:
        return 0.5
    return float(max(pos, neg) / total)


def calculate_trend_direction(close: np.ndarray, short_period: int = 20, long_period: int = 50) -> int:
    """Simple dual-MA trend direction: +1 up, -1 down, 0 sideways."""
    if len(close) < long_period:
        return 0

    short_ma = float(np.mean(close[-short_period:]))
    long_ma = float(np.mean(close[-long_period:]))
    diff_pct = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
    if diff_pct > 0.01:
        return 1
    if diff_pct < -0.01:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Momentum indicators
# ---------------------------------------------------------------------------


def calculate_rsi(close: np.ndarray, period: int = 14) -> float:
    """Relative Strength Index (last bar)."""
    if len(close) < period + 1:
        return float("nan")

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def calculate_momentum_slope(close: np.ndarray, period: int = 10) -> float:
    """First derivative of close price (momentum slope), normalised."""
    if len(close) < period + 1:
        return float("nan")

    recent = close[-period:]
    slope = calculate_trend_slope(recent, period)
    return slope


def calculate_momentum_acceleration(close: np.ndarray, period: int = 10) -> float:
    """Second derivative (acceleration) of close price, normalised."""
    if len(close) < period * 2:
        return float("nan")

    slope_current = calculate_trend_slope(close[-period:], period)
    slope_prev = calculate_trend_slope(close[-period * 2: -period], period)
    if np.isnan(slope_current) or np.isnan(slope_prev):
        return float("nan")
    return float(slope_current - slope_prev)


# ---------------------------------------------------------------------------
# Risk / drawdown metrics
# ---------------------------------------------------------------------------


def calculate_current_drawdown(close: np.ndarray) -> float:
    """Current drawdown from peak as a positive fraction (e.g. 0.10 = 10 %)."""
    if len(close) == 0:
        return float("nan")

    peak = float(np.max(close))
    current = float(close[-1])
    if peak == 0:
        return 0.0
    return float((peak - current) / peak)


def calculate_max_drawdown(close: np.ndarray) -> float:
    """Maximum historical drawdown as a positive fraction."""
    if len(close) < 2:
        return 0.0

    peak = close[0]
    max_dd = 0.0
    for price in close[1:]:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def calculate_sharpe_ratio(close: np.ndarray, period: int = 252, risk_free_rate: float = 0.04) -> float:
    """Annualised Sharpe ratio from price series."""
    if len(close) < 2:
        return float("nan")

    log_returns = np.diff(np.log(close))
    mean_return = float(np.mean(log_returns)) * period
    std_return = float(np.std(log_returns)) * np.sqrt(period)
    if std_return == 0:
        return float("nan")
    return float((mean_return - risk_free_rate) / std_return)


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------


class LeverageFeatures:
    """Compute all leverage-model features from OHLCV data.

    Example usage::

        features = LeverageFeatures()
        feature_dict = features.compute(close, high, low)
        X = features.to_array(feature_dict)
    """

    FEATURE_NAMES = [
        # Volatility
        "atr_normalized",
        "bb_width",
        "hist_vol_20",
        "hist_vol_50",
        "hist_vol_100",
        "volatility_percentile",
        "volatility_spike",
        "volatility_ratio",
        # Trend
        "adx",
        "trend_slope",
        "trend_consistency",
        "trend_direction",
        # Momentum
        "rsi",
        "momentum_slope",
        "momentum_acceleration",
        # Risk
        "current_drawdown",
        "max_drawdown",
        "sharpe_ratio",
    ]

    def compute(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute all features from OHLCV arrays.

        Args:
            close: Close prices (oldest first).
            high: High prices.
            low: Low prices.
            volume: Optional volume array (unused for now, reserved).

        Returns:
            Dictionary of feature name → value.
        """
        close = np.asarray(close, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        atr = calculate_atr(close, high, low, period=14)
        atr_normalized = (atr / close[-1]) if (not np.isnan(atr) and close[-1] != 0) else float("nan")

        hist_vol_20 = calculate_historical_volatility(close, period=20)
        hist_vol_50 = calculate_historical_volatility(close, period=50)
        hist_vol_100 = calculate_historical_volatility(close, period=100)

        # Volatility ratio: current (20-period) vs longer baseline (100-period)
        if not np.isnan(hist_vol_20) and not np.isnan(hist_vol_100) and hist_vol_100 > 0:
            volatility_ratio = hist_vol_20 / hist_vol_100
        else:
            volatility_ratio = float("nan")

        return {
            "atr_normalized": atr_normalized,
            "bb_width": calculate_bollinger_band_width(close, period=20),
            "hist_vol_20": hist_vol_20,
            "hist_vol_50": hist_vol_50,
            "hist_vol_100": hist_vol_100,
            "volatility_percentile": calculate_volatility_percentile(close, period=20, lookback=100),
            "volatility_spike": float(detect_volatility_spike(close, period=20)),
            "volatility_ratio": volatility_ratio,
            "adx": calculate_adx(high, low, close, period=14),
            "trend_slope": calculate_trend_slope(close, period=20),
            "trend_consistency": calculate_trend_consistency(close, period=20),
            "trend_direction": float(calculate_trend_direction(close)),
            "rsi": calculate_rsi(close, period=14),
            "momentum_slope": calculate_momentum_slope(close, period=10),
            "momentum_acceleration": calculate_momentum_acceleration(close, period=10),
            "current_drawdown": calculate_current_drawdown(close),
            "max_drawdown": calculate_max_drawdown(close),
            "sharpe_ratio": calculate_sharpe_ratio(close),
        }

    def to_array(self, feature_dict: dict[str, float]) -> np.ndarray:
        """Convert a feature dict to a 1-D numpy array in canonical order."""
        return np.array([feature_dict.get(name, np.nan) for name in self.FEATURE_NAMES], dtype=float)

    def fill_missing(self, feature_dict: dict[str, float], strategy: str = "zero") -> dict[str, float]:
        """Replace NaN values with a fill value.

        Args:
            feature_dict: Feature dictionary potentially containing NaN.
            strategy: ``"zero"`` replaces NaN with 0; ``"mean"`` with 0.5 for
                bounded features, 0 for others.

        Returns:
            New dictionary with NaN values replaced.
        """
        fill_value = 0.0
        return {k: (fill_value if np.isnan(v) else v) for k, v in feature_dict.items()}
