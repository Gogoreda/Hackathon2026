import numpy as np
import pandas as pd


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _macd(close):
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal


def _bollinger(close, window=20):
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + (2 * std)
    lower = middle - (2 * std)
    width = (upper - lower) / middle.replace(0, np.nan)
    position = (close - lower) / (upper - lower).replace(0, np.nan)
    return width, position


def _atr(ohlcv, window=14):
    high_low = ohlcv["High"] - ohlcv["Low"]
    high_close = (ohlcv["High"] - ohlcv["Close"].shift()).abs()
    low_close = (ohlcv["Low"] - ohlcv["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _ensure_ohlcv(ohlcv):
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in expected if col not in ohlcv.columns]

    if missing:
        return pd.DataFrame()

    return ohlcv[expected].copy()


def build_feature_frame(ohlcv, macro_returns=None):
    ohlcv = _ensure_ohlcv(ohlcv)

    if ohlcv.empty:
        return pd.DataFrame()

    features = pd.DataFrame(index=ohlcv.index)
    close = ohlcv["Close"]
    volume = ohlcv["Volume"]

    features["close"] = close
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["return_10d"] = close.pct_change(10)
    features["return_21d"] = close.pct_change(21)
    features["realized_vol_5d"] = features["return_1d"].rolling(5).std()
    features["realized_vol_10d"] = features["return_1d"].rolling(10).std()
    features["realized_vol_21d"] = features["return_1d"].rolling(21).std()
    features["volume_change_5d"] = volume.pct_change(5)
    features["volume_zscore_21d"] = (
        (volume - volume.rolling(21).mean())
        / volume.rolling(21).std().replace(0, np.nan)
    )
    features["rsi_14"] = _rsi(close)

    macd, signal, histogram = _macd(close)
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_histogram"] = histogram

    bollinger_width, bollinger_position = _bollinger(close)
    features["bollinger_width"] = bollinger_width
    features["bollinger_position"] = bollinger_position
    features["atr_14"] = _atr(ohlcv)
    features["atr_14_pct"] = features["atr_14"] / close.replace(0, np.nan)

    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_earnings_season"] = features["month"].isin([1, 4, 7, 10]).astype(int)

    if macro_returns is not None and not macro_returns.empty:
        macro_features = macro_returns.reindex(features.index).ffill()
        features = features.join(macro_features)

    return features.replace([np.inf, -np.inf], np.nan)


def add_future_targets(features, horizon_days):
    features = features.copy()
    features["target_return"] = (
        features["close"].shift(-horizon_days) / features["close"] - 1
    )
    features["target_positive"] = (features["target_return"] > 0).astype(int)
    return features
