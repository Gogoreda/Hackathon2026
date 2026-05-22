from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def _get_nested_value(data, keys, default=None):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    return current if current is not None else default


def _parse_timestamp(value):
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
    except Exception:
        return None


def _parse_news_item(ticker, item):
    content = item.get("content", {}) if isinstance(item, dict) else {}

    title = (
        item.get("title")
        or _get_nested_value(content, ["title"])
        or _get_nested_value(content, ["displayName"])
    )

    publisher = (
        item.get("publisher")
        or _get_nested_value(content, ["provider", "displayName"])
        or _get_nested_value(content, ["provider", "name"])
    )

    link = (
        item.get("link")
        or _get_nested_value(content, ["canonicalUrl", "url"])
        or _get_nested_value(content, ["clickThroughUrl", "url"])
    )

    summary = (
        item.get("summary")
        or _get_nested_value(content, ["summary"])
        or _get_nested_value(content, ["description"])
    )

    published_at = (
        item.get("providerPublishTime")
        or _get_nested_value(content, ["pubDate"])
    )

    if isinstance(published_at, str):
        published_date = published_at[:10]
    else:
        published_date = _parse_timestamp(published_at)

    return {
        "ticker": ticker.upper(),
        "published_date": published_date,
        "publisher": publisher,
        "title": title,
        "summary": summary,
        "url": link,
    }


def get_company_news(tickers, limit_per_ticker=5):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news or []
        except Exception as exc:
            rows.append({
                "ticker": ticker.upper(),
                "published_date": None,
                "publisher": None,
                "title": f"News error: {exc}",
                "summary": None,
                "url": None,
            })
            continue

        for item in news_items[:limit_per_ticker]:
            parsed = _parse_news_item(ticker, item)
            if parsed["title"]:
                rows.append(parsed)

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "published_date",
            "publisher",
            "title",
            "summary",
            "url",
        ]
    )


def _pick_statement(stock):
    for attribute in [
        "quarterly_income_stmt",
        "quarterly_financials",
        "quarterly_earnings",
    ]:
        try:
            statement = getattr(stock, attribute)
        except Exception:
            continue

        if isinstance(statement, pd.DataFrame) and not statement.empty:
            return statement

    return pd.DataFrame()


def _get_statement_value(statement, quarter, possible_rows):
    for row_name in possible_rows:
        if row_name in statement.index:
            value = statement.loc[row_name, quarter]
            if pd.notna(value):
                return value

    return None


def get_quarterly_results(tickers, quarters=3):
    rows = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            statement = _pick_statement(stock)
        except Exception as exc:
            rows.append({
                "ticker": ticker.upper(),
                "quarter": None,
                "revenue": None,
                "gross_profit": None,
                "operating_income": None,
                "net_income": None,
                "error": str(exc),
            })
            continue

        if statement.empty:
            rows.append({
                "ticker": ticker.upper(),
                "quarter": None,
                "revenue": None,
                "gross_profit": None,
                "operating_income": None,
                "net_income": None,
                "error": "No quarterly statement data available",
            })
            continue

        quarter_columns = list(statement.columns[:quarters])

        for quarter in quarter_columns:
            quarter_label = (
                quarter.date().isoformat()
                if hasattr(quarter, "date")
                else str(quarter)
            )

            rows.append({
                "ticker": ticker.upper(),
                "quarter": quarter_label,
                "revenue": _get_statement_value(
                    statement,
                    quarter,
                    ["Total Revenue", "Revenue"]
                ),
                "gross_profit": _get_statement_value(
                    statement,
                    quarter,
                    ["Gross Profit"]
                ),
                "operating_income": _get_statement_value(
                    statement,
                    quarter,
                    ["Operating Income", "Operating Income or Loss"]
                ),
                "net_income": _get_statement_value(
                    statement,
                    quarter,
                    ["Net Income", "Net Income Common Stockholders", "Earnings"]
                ),
                "error": None,
            })

    return pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "quarter",
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "error",
        ]
    )
