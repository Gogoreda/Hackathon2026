import streamlit as st
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

from agents import (
    financial_agent,
    prediction_agent,
    report_agent,
    sentiment_agent,
)
from database_service import (
    get_or_create_session_id,
    initialize_database,
    is_database_configured,
    log_prediction_requests,
    log_usage,
)
from finance_service import get_stock_data
from market_context_service import get_company_news, get_quarterly_results
from pdf_service import generate_pdf_report
from prediction_service import add_predictive_analysis
from search_service import resolve_companies_to_tickers

load_dotenv()

APP_VERSION = "local-ml-db-v1"


def has_value(value):
    return value is not None and value == value


st.set_page_config(
    page_title="AlphaLens",
    layout="wide"
)

st.title("AlphaLens")
st.subheader("Multi-Agent Predictive Investment Research Copilot")

st.warning(
    "Educational prototype only. Predictive scenarios are not financial advice "
    "or buy/sell recommendations."
)

with st.sidebar:
    st.header("Input")

    user_session_id = get_or_create_session_id(st.session_state)

    companies_input = st.text_input(
        "Companies or stock tickers",
        value="Nvidia, AMD, Palantir, Amazon"
    )

    horizon = st.selectbox(
        "Prediction horizon",
        ["1 week", "1 month", "3 months", "6 months", "1 year"],
        index=4
    )

    simulations = st.slider(
        "Monte Carlo simulations",
        min_value=250,
        max_value=3000,
        value=1000,
        step=250
    )

    run_button = st.button("Generate Predictive Brief")

    st.divider()
    st.header("Database")

    if is_database_configured():
        st.caption("MySQL logging enabled.")

        if st.button("Initialize local database"):
            success, message = initialize_database()

            if success:
                st.success(message)
            else:
                st.warning(message)
    else:
        st.caption("MySQL logging disabled. Add MYSQL_* values to .env to enable it.")

if run_button:
    companies = [
        company.strip()
        for company in companies_input.split(",")
        if company.strip()
    ]

    if not companies:
        st.error("Please enter at least one company or ticker.")
        st.stop()

    with st.spinner("Search Agent is resolving company names into stock tickers..."):
        resolved_companies = resolve_companies_to_tickers(companies)

    st.header("0. Search Agent - Resolved Companies")

    st.dataframe(
        resolved_companies,
        use_container_width=True,
        hide_index=True
    )

    tickers = [
        item["ticker"]
        for item in resolved_companies
        if item["ticker"]
    ]

    if not tickers:
        st.error("No valid tickers found. Try using known company names or stock symbols.")
        st.stop()

    with st.spinner("Financial Data Agent is collecting public market data..."):
        df = get_stock_data(tickers)

    with st.spinner("Market Context Agent is collecting Yahoo Finance news and quarterly results..."):
        news_df = get_company_news(tickers)
        quarterly_df = get_quarterly_results(tickers, quarters=3)

    with st.spinner("Prediction Engine is fitting ML, GARCH volatility, and Monte Carlo scenarios..."):
        df = add_predictive_analysis(
            df,
            horizon=horizon,
            simulations=simulations
        )

    try:
        log_usage(
            tickers=tickers,
            horizon=horizon,
            simulations=simulations,
            user_session_id=user_session_id,
            app_version=APP_VERSION
        )
        prediction_log_ids = log_prediction_requests(
            df=df,
            horizon=horizon,
            simulations=simulations,
            user_session_id=user_session_id
        )
    except Exception as exc:
        prediction_log_ids = []
        st.warning(f"Database logging failed: {exc}")

    st.success("Agents completed the first analysis.")

    if prediction_log_ids:
        st.caption(f"Saved {len(prediction_log_ids)} prediction request(s) to MySQL.")

    st.header("1. Company Comparison")

    metric_cols = st.columns(len(df))

    for col, (_, row) in zip(metric_cols, df.iterrows()):
        with col:
            price = row.get("current_price")
            price_text = f"${price:.2f}" if has_value(price) else "N/A"
            expected_price = row.get("expected_price")
            expected_price_text = (
                f"${expected_price:.2f}"
                if has_value(expected_price)
                else "N/A"
            )
            expected_return = row.get("expected_return")
            expected_return_text = (
                f"{expected_return * 100:.1f}%"
                if has_value(expected_return)
                else "N/A"
            )

            st.caption(row.get("ticker", "N/A"))

            current_col, predicted_col = st.columns(2)

            with current_col:
                st.metric(
                    label="Current price",
                    value=price_text
                )

            with predicted_col:
                st.metric(
                    label=f"Predicted price ({horizon})",
                    value=expected_price_text,
                    delta=expected_return_text
                )

            st.caption(row.get("company_name", ""))
            st.caption(f"Sector: {row.get('sector', 'N/A')}")
            st.caption(f"P/E: {row.get('pe_ratio', 'N/A')}")
            st.caption(f"Outlook: {row.get('outlook', 'N/A')}")

    st.subheader("Market Cap Comparison")

    fig_market_cap = px.bar(
        df,
        x="ticker",
        y="market_cap",
        text="market_cap",
        title="Market Cap by Company"
    )

    fig_market_cap.update_traces(
        texttemplate="%{y:.2s}",
        textposition="outside"
    )

    fig_market_cap.update_layout(
        yaxis_title="Market Cap",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_market_cap, use_container_width=True)

    st.subheader("Key Financial Metrics")

    metrics_columns = [
        "ticker",
        "pe_ratio",
        "revenue_growth",
        "profit_margin",
        "volatility",
        "debt_to_equity",
    ]

    available_metrics_columns = [
        col for col in metrics_columns
        if col in df.columns
    ]

    metrics_df = df[available_metrics_columns].copy()

    percentage_columns = [
        "revenue_growth",
        "profit_margin",
        "volatility"
    ]

    for col in percentage_columns:
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
            metrics_df[col] = metrics_df[col] * 100

    metrics_df = metrics_df.rename(columns={
        "ticker": "Ticker",
        "pe_ratio": "P/E Ratio",
        "revenue_growth": "Revenue Growth (%)",
        "profit_margin": "Profit Margin (%)",
        "volatility": "Volatility (%)",
        "debt_to_equity": "Debt to Equity (%)",
    })

    metrics_long = metrics_df.melt(
        id_vars="Ticker",
        var_name="Metric",
        value_name="Value"
    )

    metrics_long["Value"] = pd.to_numeric(
        metrics_long["Value"],
        errors="coerce"
    )
    metrics_long = metrics_long.dropna(subset=["Value"])
    metrics_long["Value"] = metrics_long["Value"].round(2)

    fig_metrics = px.bar(
        metrics_long,
        x="Ticker",
        y="Value",
        color="Metric",
        barmode="group",
        title="Financial Metrics by Company"
    )

    fig_metrics.update_traces(
        textposition="outside"
    )

    fig_metrics.update_layout(
        yaxis_title="Metric Value",
        xaxis_title="Company",
        legend_title="Metric"
    )

    st.plotly_chart(fig_metrics, use_container_width=True)

    with st.expander("View raw company data"):
        display_columns = [
            "ticker",
            "company_name",
            "sector",
            "current_price",
            "market_cap",
            "pe_ratio",
            "revenue_growth",
            "profit_margin",
            "debt_to_equity",
            "volatility",
            "model",
            "expected_price",
            "expected_return",
            "probability_positive",
            "annualized_volatility",
            "market_beta",
            "growth_beta",
            "small_cap_beta",
            "garch_alpha",
            "garch_beta",
            "outlook",
        ]

        available_columns = [
            col for col in display_columns
            if col in df.columns
        ]

        st.dataframe(
            df[available_columns],
            use_container_width=True,
            hide_index=True
        )

    st.header("2. News and Quarterly Results")

    st.subheader("Latest Yahoo Finance News")

    if news_df.empty:
        st.info("No Yahoo Finance news items were available for these tickers.")
    else:
        news_display_columns = [
            "ticker",
            "published_date",
            "publisher",
            "title",
            "url",
        ]

        st.dataframe(
            news_df[news_display_columns],
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Latest 3 Quarterly Results")

    if quarterly_df.empty:
        st.info("No quarterly results were available for these tickers.")
    else:
        quarter_display = quarterly_df.copy()

        numeric_columns = [
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
        ]

        for col in numeric_columns:
            if col in quarter_display.columns:
                quarter_display[col] = quarter_display[col].apply(
                    lambda value: f"{value:,.0f}" if has_value(value) else "N/A"
                )

        st.dataframe(
            quarter_display,
            use_container_width=True,
            hide_index=True
        )

    st.header("3. Predictive Outlook")

    st.subheader("Model Pipeline")

    pipeline_columns = [
        "ticker",
        "model",
        "ml_model",
        "ml_training_rows",
        "ml_feature_count",
        "ml_validation_mae",
        "ml_directional_accuracy",
        "market_beta",
        "growth_beta",
        "small_cap_beta",
        "factor_expected_daily_return",
        "annualized_volatility",
        "garch_alpha",
        "garch_beta",
    ]

    pipeline_df = df[
        [col for col in pipeline_columns if col in df.columns]
    ].copy()

    percentage_pipeline_columns = [
        "factor_expected_daily_return",
        "annualized_volatility",
    ]

    for col in percentage_pipeline_columns:
        if col in pipeline_df.columns:
            pipeline_df[col] = pd.to_numeric(pipeline_df[col], errors="coerce")
            pipeline_df[col] = pipeline_df[col] * 100

    pipeline_df = pipeline_df.rename(columns={
        "ticker": "Ticker",
        "model": "Model",
        "ml_model": "ML Model",
        "ml_training_rows": "ML Training Rows",
        "ml_feature_count": "ML Features",
        "ml_validation_mae": "Walk-Forward MAE",
        "ml_directional_accuracy": "Walk-Forward Direction Accuracy",
        "market_beta": "Market Beta",
        "growth_beta": "Growth Beta",
        "small_cap_beta": "Small Cap Beta",
        "factor_expected_daily_return": "Factor Expected Daily Return (%)",
        "annualized_volatility": "GARCH Annualized Volatility (%)",
        "garch_alpha": "GARCH Alpha",
        "garch_beta": "GARCH Beta",
    })

    st.dataframe(
        pipeline_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    prediction_cols = st.columns(len(df))

    for col, (_, row) in zip(prediction_cols, df.iterrows()):
        with col:
            expected_return = row.get("expected_return")
            expected_return_text = (
                f"{expected_return * 100:.1f}%"
                if has_value(expected_return)
                else "N/A"
            )

            st.metric(
                label=f"{row.get('ticker', 'N/A')} Expected Return",
                value=expected_return_text,
                delta=row.get("outlook", "N/A")
            )
            st.caption(row.get("prediction_explanation", ""))

    st.subheader("Expected Return")

    fig_expected_return = px.bar(
        df.sort_values("expected_return", ascending=False),
        x="ticker",
        y="expected_return",
        color="outlook",
        title=f"Expected Return by Company - {horizon}",
        text="expected_return"
    )

    fig_expected_return.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside"
    )

    fig_expected_return.update_layout(
        yaxis_title="Expected Return",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_expected_return, use_container_width=True)

    st.subheader("Scenario Range")

    scenario_columns = [
        "ticker",
        "bear_case_return",
        "base_case_return",
        "bull_case_return",
        "probability_positive",
    ]

    scenario_df = df[scenario_columns].copy()

    for col in scenario_columns[1:]:
        scenario_df[col] = pd.to_numeric(scenario_df[col], errors="coerce")
        scenario_df[col] = scenario_df[col] * 100

    scenario_df = scenario_df.rename(columns={
        "ticker": "Ticker",
        "bear_case_return": "Bear Case (%)",
        "base_case_return": "Base Case (%)",
        "bull_case_return": "Bull Case (%)",
        "probability_positive": "Probability Positive (%)",
    })

    st.dataframe(
        scenario_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    scenario_long = scenario_df.melt(
        id_vars="Ticker",
        value_vars=[
            "Bear Case (%)",
            "Base Case (%)",
            "Bull Case (%)"
        ],
        var_name="Scenario",
        value_name="Projected Return (%)"
    )

    fig_scenarios = px.bar(
        scenario_long,
        x="Ticker",
        y="Projected Return (%)",
        color="Scenario",
        barmode="group",
        title=f"Monte Carlo Scenario Range - {horizon}"
    )

    fig_scenarios.update_layout(
        yaxis_title="Projected Return (%)",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_scenarios, use_container_width=True)

    fig_probability = px.bar(
        df,
        x="ticker",
        y="probability_positive",
        color="outlook",
        title="Estimated Probability of Positive Return",
        text="probability_positive",
        range_y=[0, 1]
    )

    fig_probability.update_traces(
        texttemplate="%{y:.1%}",
        textposition="outside"
    )

    fig_probability.update_layout(
        yaxis_title="Probability",
        xaxis_title="Company"
    )

    st.plotly_chart(fig_probability, use_container_width=True)

    st.header("4. Final Predictive Brief")

    with st.spinner("AI Agents are generating the final predictive research brief..."):
        financial_summary = financial_agent(df, quarterly_df=quarterly_df)
        sentiment_summary = sentiment_agent(tickers, news_df=news_df)
        prediction_summary = prediction_agent(df)

        final_report = report_agent(
            df=df,
            financial_summary=financial_summary,
            sentiment_summary=sentiment_summary,
            prediction_summary=prediction_summary,
            horizon=horizon,
            news_df=news_df,
            quarterly_df=quarterly_df
        )

    st.markdown(final_report)

    pdf_filename = "alphalens_report.pdf"

    generate_pdf_report(
        filename=pdf_filename,
        df=df,
        financial_summary=financial_summary,
        sentiment_summary=sentiment_summary,
        prediction_summary=prediction_summary,
        final_report=final_report,
        news_df=news_df,
        quarterly_df=quarterly_df
    )

    with open(pdf_filename, "rb") as pdf_file:
        st.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name="alphalens_report.pdf",
            mime="application/pdf"
        )

else:
    st.info("Enter company names or tickers in the sidebar and click Generate Predictive Brief.")
