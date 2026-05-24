import yfinance as yf
import pandas as pd


def _first_available(info, keys):
    for key in keys:
        value = info.get(key)

        if value is not None:
            return value

    return None


def _latest_close(hist):
    if hist.empty or "Close" not in hist:
        return None

    close = hist["Close"].dropna()

    if close.empty:
        return None

    return float(close.iloc[-1])


def get_stock_data(tickers):
    results = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            current_price = _first_available(
                info,
                [
                    "currentPrice",
                    "regularMarketPrice",
                    "navPrice",
                    "previousClose",
                ]
            )

            if current_price is None:
                current_price = _latest_close(hist)

            if hist.empty:
                volatility = None
            else:
                hist["daily_return"] = hist["Close"].pct_change()
                volatility = hist["daily_return"].std() * (252 ** 0.5)

            results.append({
                "ticker": ticker.upper(),
                "company_name": (
                    info.get("longName")
                    or info.get("shortName")
                    or ticker.upper()
                ),
                "asset_type": info.get("quoteType", "N/A"),
                "sector": (
                    info.get("sector")
                    or info.get("category")
                    or info.get("fundFamily")
                    or "N/A"
                ),
                "current_price": current_price,
                "market_cap": info.get("marketCap") or info.get("totalAssets"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "debt_to_equity": info.get("debtToEquity"),
                "volatility": volatility,
            })

        except Exception as e:
            results.append({
                "ticker": ticker.upper(),
                "company_name": "Error",
                "asset_type": "N/A",
                "sector": "N/A",
                "current_price": None,
                "market_cap": None,
                "pe_ratio": None,
                "forward_pe": None,
                "profit_margin": None,
                "revenue_growth": None,
                "debt_to_equity": None,
                "volatility": None,
                "error": str(e),
            })

    return pd.DataFrame(results)
