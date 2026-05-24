import yfinance as yf


VALID_QUOTE_TYPES = {
    "EQUITY",
    "ETF",
    "MUTUALFUND",
    "INDEX",
}


def resolve_direct_ticker(query):
    symbol = query.strip().upper()

    if not symbol or " " in symbol:
        return None

    try:
        stock = yf.Ticker(symbol)
        info = stock.info or {}
        quote_type = info.get("quoteType")
        hist = stock.history(period="5d")

        if quote_type in VALID_QUOTE_TYPES and not hist.empty:
            return symbol

    except Exception:
        return None

    return None


def search_companies(query, max_results=5):
    try:
        search = yf.Search(query, max_results=max_results)
        quotes = search.quotes or []

        results = []

        for item in quotes:
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or item.get("name")
            exchange = item.get("exchange")
            quote_type = item.get("quoteType")

            if symbol and quote_type in VALID_QUOTE_TYPES:
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "quote_type": quote_type
                })

        return results

    except Exception as e:
        return [{
            "symbol": None,
            "name": f"Search error: {str(e)}",
            "exchange": None,
            "quote_type": None
        }]


def resolve_company_to_ticker(company_name):
    direct_ticker = resolve_direct_ticker(company_name)

    if direct_ticker:
        return direct_ticker

    results = search_companies(company_name, max_results=5)

    valid_results = [
        r for r in results
        if r.get("symbol") is not None
    ]

    if not valid_results:
        return None

    return valid_results[0]["symbol"]


def resolve_companies_to_tickers(company_names):
    resolved = []

    for company in company_names:
        ticker = resolve_company_to_ticker(company.strip())

        resolved.append({
            "input": company,
            "ticker": ticker
        })

    return resolved
