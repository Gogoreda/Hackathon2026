def generate_markdown_report(
    df,
    financial_summary,
    sentiment_summary,
    prediction_summary,
    final_report,
    news_df=None,
    quarterly_df=None
):
    markdown = "# AlphaLens Predictive Investment Research Brief\n\n"

    markdown += "## Company Comparison\n\n"

    columns = [
        "ticker",
        "company_name",
        "sector",
        "current_price",
        "pe_ratio",
        "revenue_growth",
        "volatility",
        "model",
        "ml_model",
        "expected_price",
        "expected_return",
        "ml_expected_return",
        "probability_positive",
        "ml_probability_positive",
        "ml_validation_mae",
        "ml_directional_accuracy",
        "annualized_volatility",
        "market_beta",
        "growth_beta",
        "small_cap_beta",
        "garch_alpha",
        "garch_beta",
        "outlook",
    ]

    available_columns = [col for col in columns if col in df.columns]

    markdown += df[available_columns].to_markdown(index=False)

    if news_df is not None and not news_df.empty:
        markdown += "\n\n## Latest Yahoo Finance News\n\n"
        news_columns = [
            "ticker",
            "published_date",
            "publisher",
            "title",
            "url",
        ]
        available_news_columns = [
            col for col in news_columns if col in news_df.columns
        ]
        markdown += news_df[available_news_columns].to_markdown(index=False)

    if quarterly_df is not None and not quarterly_df.empty:
        markdown += "\n\n## Latest Quarterly Results\n\n"
        quarterly_columns = [
            "ticker",
            "quarter",
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
        ]
        available_quarterly_columns = [
            col for col in quarterly_columns if col in quarterly_df.columns
        ]
        markdown += quarterly_df[available_quarterly_columns].to_markdown(index=False)

    markdown += "\n\n## Financial Agent Summary\n\n"
    markdown += financial_summary

    markdown += "\n\n## News & Sentiment Agent Summary\n\n"
    markdown += sentiment_summary

    markdown += "\n\n## Predictive Analysis Agent Summary\n\n"
    markdown += prediction_summary

    markdown += "\n\n## Final Predictive Brief\n\n"
    markdown += final_report

    markdown += "\n\n---\n"
    markdown += "Disclaimer: These are educational predictive scenarios only and not financial advice."

    return markdown
