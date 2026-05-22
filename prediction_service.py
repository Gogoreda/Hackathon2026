import math
import zlib

import numpy as np
import pandas as pd
import yfinance as yf

from ml_service import fit_predict_ml_return


TRADING_DAYS = 252
FACTOR_TICKERS = ["SPY", "QQQ", "IWM"]
MACRO_TICKERS = ["^VIX", "^TNX"]
FACTOR_NAMES = {
    "SPY": "market_beta",
    "QQQ": "growth_beta",
    "IWM": "small_cap_beta",
}


def horizon_to_days(horizon):
    mapping = {
        "1 week": 5,
        "1 month": 21,
        "3 months": 63,
        "6 months": 126,
        "1 year": 252,
    }

    return mapping.get(horizon, 252)


def classify_outlook(expected_return):
    if expected_return is None or pd.isna(expected_return):
        return "Insufficient data"
    if expected_return >= 0.15:
        return "Strong positive"
    if expected_return >= 0.03:
        return "Positive"
    if expected_return > -0.03:
        return "Neutral"
    if expected_return > -0.15:
        return "Negative"
    return "Strong negative"


def explain_prediction(row):
    if row.get("expected_return") is None or pd.isna(row.get("expected_return")):
        return "Not enough historical price data to estimate a predictive scenario."

    probability = row.get("probability_positive")
    volatility = row.get("annualized_volatility")
    expected_return = row.get("expected_return")
    model = row.get("model", "predictive model")

    reasons = []

    if expected_return >= 0.03:
        reasons.append("ML model points to a positive base scenario")
    elif expected_return <= -0.03:
        reasons.append("ML model points to a negative base scenario")
    else:
        reasons.append("ML model points to a near-flat base scenario")

    if probability is not None and not pd.isna(probability):
        if probability >= 0.60:
            reasons.append("most Monte Carlo paths ended with a positive return")
        elif probability <= 0.40:
            reasons.append("most Monte Carlo paths did not end with a positive return")
        else:
            reasons.append("Monte Carlo outcomes are mixed")

    if volatility is not None and not pd.isna(volatility):
        if volatility >= 0.45:
            reasons.append("GARCH estimates elevated forward volatility")
        elif volatility <= 0.20:
            reasons.append("GARCH estimates lower forward volatility")

    return f"Prediction based on {model}: " + ", ".join(reasons) + "."


def _get_price_history(tickers, period="3y"):
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs("Close", axis=1, level=1, drop_level=True)
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers if isinstance(tickers, list) else [tickers]

    return prices.dropna(how="all")


def _get_downloaded_history(tickers, period="5y"):
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False
        )
    except Exception:
        return pd.DataFrame()

    return data if isinstance(data, pd.DataFrame) else pd.DataFrame()


def _extract_ticker_ohlcv(history, ticker):
    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(1):
            return pd.DataFrame()

        return history.xs(ticker, axis=1, level=1, drop_level=True).dropna()

    return history.dropna()


def _extract_close_prices(history):
    if history.empty:
        return pd.DataFrame()

    if isinstance(history.columns, pd.MultiIndex):
        if "Close" in history.columns.get_level_values(0):
            return history["Close"].dropna(how="all")
        if "Close" in history.columns.get_level_values(1):
            return history.xs("Close", axis=1, level=1, drop_level=True).dropna(how="all")

    if "Close" in history.columns:
        return history[["Close"]].dropna()

    return pd.DataFrame()


def _estimate_factor_model(asset_returns, factor_returns):
    aligned = pd.concat(
        [asset_returns.rename("asset"), factor_returns],
        axis=1
    ).dropna()

    if len(aligned) < 90:
        return None

    y = aligned["asset"].to_numpy()
    x = aligned[factor_returns.columns].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])

    try:
        coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    except Exception:
        return None

    residuals = y - x @ coefficients
    recent_factor_returns = factor_returns.tail(TRADING_DAYS).mean().fillna(0)
    expected_daily_return = float(
        coefficients[0] + np.dot(coefficients[1:], recent_factor_returns.to_numpy())
    )

    result = {
        "factor_alpha_daily": float(coefficients[0]),
        "factor_expected_daily_return": expected_daily_return,
        "factor_residual_volatility": float(np.std(residuals, ddof=1)),
    }

    for ticker, beta in zip(factor_returns.columns, coefficients[1:]):
        result[FACTOR_NAMES.get(ticker, f"{ticker.lower()}_beta")] = float(beta)

    return result


def _garch_negative_log_likelihood(returns, alpha, beta):
    variance = float(np.var(returns, ddof=1))

    if variance <= 0 or alpha + beta >= 0.995:
        return np.inf

    omega = variance * (1 - alpha - beta)
    conditional_variance = variance
    nll = 0.0

    for value in returns:
        conditional_variance = (
            omega
            + alpha * (value ** 2)
            + beta * conditional_variance
        )

        if conditional_variance <= 0 or not math.isfinite(conditional_variance):
            return np.inf

        nll += 0.5 * (
            math.log(2 * math.pi)
            + math.log(conditional_variance)
            + (value ** 2) / conditional_variance
        )

    return nll


def _estimate_garch(returns):
    clean_returns = returns.dropna().to_numpy()

    if len(clean_returns) < 90:
        return None

    clean_returns = clean_returns - np.mean(clean_returns)
    unconditional_variance = float(np.var(clean_returns, ddof=1))

    if unconditional_variance <= 0 or not math.isfinite(unconditional_variance):
        return None

    best = None

    for alpha in np.arange(0.03, 0.16, 0.02):
        for beta in np.arange(0.70, 0.96, 0.03):
            if alpha + beta >= 0.995:
                continue

            nll = _garch_negative_log_likelihood(clean_returns, alpha, beta)

            if best is None or nll < best["nll"]:
                best = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "nll": float(nll),
                }

    if best is None:
        return None

    alpha = best["alpha"]
    beta = best["beta"]
    omega = unconditional_variance * (1 - alpha - beta)
    conditional_variance = unconditional_variance

    for value in clean_returns:
        conditional_variance = (
            omega
            + alpha * (value ** 2)
            + beta * conditional_variance
        )

    return {
        "garch_omega": float(omega),
        "garch_alpha": alpha,
        "garch_beta": beta,
        "last_conditional_variance": float(conditional_variance),
        "unconditional_variance": unconditional_variance,
    }


def _forecast_garch_variance(garch, horizon_days):
    variances = []
    variance = garch["last_conditional_variance"]
    omega = garch["garch_omega"]
    alpha = garch["garch_alpha"]
    beta = garch["garch_beta"]
    persistence = alpha + beta

    for _ in range(horizon_days):
        variance = omega + persistence * variance
        variances.append(max(variance, 0))

    return np.array(variances)


def _fallback_prediction(ticker, horizon_days, volatility):
    return {
        "ticker": ticker.upper() if ticker else ticker,
        "model": "Insufficient data",
        "horizon_days": horizon_days,
        "expected_price": None,
        "expected_return": None,
        "bear_case_return": None,
        "base_case_return": None,
        "bull_case_return": None,
        "probability_positive": None,
        "annualized_volatility": volatility,
        "factor_alpha_daily": None,
        "factor_expected_daily_return": None,
        "market_beta": None,
        "growth_beta": None,
        "small_cap_beta": None,
        "garch_alpha": None,
        "garch_beta": None,
        "garch_omega": None,
        "ml_model": None,
        "ml_expected_return": None,
        "ml_probability_positive": None,
        "ml_residual_volatility": None,
        "ml_validation_mae": None,
        "ml_directional_accuracy": None,
        "ml_training_rows": None,
        "ml_feature_count": None,
    }


def simulate_ticker(
    ticker,
    prices,
    factor_returns,
    ohlcv,
    macro_returns,
    horizon_days,
    simulations=1000
):
    if not ticker or ticker not in prices.columns:
        return None

    ticker_prices = prices[ticker].dropna()

    if len(ticker_prices) < 90:
        return None

    asset_returns = ticker_prices.pct_change().dropna()
    factor_model = _estimate_factor_model(asset_returns, factor_returns)
    garch = _estimate_garch(asset_returns)
    ml_result = fit_predict_ml_return(
        ohlcv=ohlcv,
        macro_returns=macro_returns,
        horizon_days=horizon_days
    )

    if factor_model is None or garch is None or ml_result is None:
        return None

    current_price = float(ticker_prices.iloc[-1])
    expected_horizon_return = ml_result["ml_expected_return"]
    expected_daily_return = (1 + expected_horizon_return) ** (1 / horizon_days) - 1
    variance_forecast = _forecast_garch_variance(garch, horizon_days)
    daily_volatility_forecast = np.sqrt(variance_forecast)

    if not np.all(np.isfinite(daily_volatility_forecast)):
        return None

    seed = zlib.crc32(str(ticker).upper().encode("utf-8"))
    rng = np.random.default_rng(seed=seed)
    shocks = rng.normal(0, 1, size=(simulations, horizon_days))
    random_returns = expected_daily_return + shocks * daily_volatility_forecast

    price_paths = current_price * np.cumprod(1 + random_returns, axis=1)
    final_prices = price_paths[:, -1]
    scenario_returns = (final_prices / current_price) - 1

    expected_price = float(np.mean(final_prices))
    expected_return = float((expected_price / current_price) - 1)

    return {
        "ticker": ticker.upper(),
        "model": f"{ml_result['ml_model']} + GARCH + Monte Carlo",
        "horizon_days": horizon_days,
        "expected_price": expected_price,
        "expected_return": expected_return,
        "bear_case_return": float(np.percentile(scenario_returns, 10)),
        "base_case_return": float(np.percentile(scenario_returns, 50)),
        "bull_case_return": float(np.percentile(scenario_returns, 90)),
        "probability_positive": float(np.mean(scenario_returns > 0)),
        "annualized_volatility": float(
            math.sqrt(float(np.mean(variance_forecast)) * TRADING_DAYS)
        ),
        "factor_alpha_daily": factor_model["factor_alpha_daily"],
        "factor_expected_daily_return": expected_daily_return,
        "market_beta": factor_model.get("market_beta"),
        "growth_beta": factor_model.get("growth_beta"),
        "small_cap_beta": factor_model.get("small_cap_beta"),
        "garch_alpha": garch["garch_alpha"],
        "garch_beta": garch["garch_beta"],
        "garch_omega": garch["garch_omega"],
        "ml_model": ml_result["ml_model"],
        "ml_expected_return": ml_result["ml_expected_return"],
        "ml_probability_positive": ml_result["ml_probability_positive"],
        "ml_residual_volatility": ml_result["ml_residual_volatility"],
        "ml_validation_mae": ml_result["ml_validation_mae"],
        "ml_directional_accuracy": ml_result["ml_directional_accuracy"],
        "ml_training_rows": ml_result["ml_training_rows"],
        "ml_feature_count": ml_result["ml_feature_count"],
    }


def add_predictive_analysis(df, horizon="1 year", simulations=1000):
    df = df.copy()
    horizon_days = horizon_to_days(horizon)
    tickers = [
        str(ticker).upper()
        for ticker in df["ticker"].dropna().tolist()
    ]
    all_tickers = list(dict.fromkeys(tickers + FACTOR_TICKERS + MACRO_TICKERS))
    history = _get_downloaded_history(all_tickers)
    prices = _extract_close_prices(history)
    predictions = []

    if prices.empty:
        for _, row in df.iterrows():
            predictions.append(
                _fallback_prediction(
                    row.get("ticker"),
                    horizon_days,
                    row.get("volatility")
                )
            )
    else:
        returns = prices.pct_change().dropna()
        available_factors = [
            ticker for ticker in FACTOR_TICKERS
            if ticker in returns.columns
        ]
        factor_returns = returns[available_factors].dropna()
        macro_columns = [
            ticker for ticker in FACTOR_TICKERS + MACRO_TICKERS
            if ticker in returns.columns
        ]
        macro_returns = returns[macro_columns].rename(columns={
            "SPY": "macro_spy_return",
            "QQQ": "macro_qqq_return",
            "IWM": "macro_iwm_return",
            "^VIX": "macro_vix_change",
            "^TNX": "macro_tnx_change",
        }).dropna(how="all")

        for _, row in df.iterrows():
            ticker = str(row.get("ticker")).upper()
            ohlcv = _extract_ticker_ohlcv(history, ticker)
            prediction = simulate_ticker(
                ticker,
                prices=prices,
                factor_returns=factor_returns,
                ohlcv=ohlcv,
                macro_returns=macro_returns,
                horizon_days=horizon_days,
                simulations=simulations
            )

            if prediction is None:
                prediction = _fallback_prediction(
                    ticker,
                    horizon_days,
                    row.get("volatility")
                )

            predictions.append(prediction)

    prediction_df = pd.DataFrame(predictions)
    merged = df.merge(
        prediction_df,
        on="ticker",
        how="left"
    )

    merged["outlook"] = merged["expected_return"].apply(classify_outlook)
    merged["prediction_explanation"] = merged.apply(
        lambda row: explain_prediction(row.to_dict()),
        axis=1
    )

    return merged
