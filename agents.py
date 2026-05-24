import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

client = Groq(api_key=api_key) if api_key else None

DEFAULT_AGENT_MODEL = os.getenv("GROQ_AGENT_MODEL", "llama-3.1-8b-instant")
DEFAULT_REPORT_MODEL = os.getenv("GROQ_REPORT_MODEL", "llama-3.3-70b-versatile")


def call_llm(prompt, model=None):
    if client is None:
        return """
LLM not configured.

Create a .env file with:
GROQ_API_KEY=your_api_key_here
"""

    response = client.chat.completions.create(
        model=model or DEFAULT_AGENT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial research assistant. "
                    "Do not give financial advice. "
                    "Do not provide buy or sell recommendations. "
                    "Provide educational, explainable analysis only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def financial_agent(df, quarterly_df=None):
    quarterly_text = (
        quarterly_df.to_string(index=False)
        if quarterly_df is not None and not quarterly_df.empty
        else "No quarterly results were available."
    )

    prompt = f"""
You are the Financial Data Agent.

Analyze these public financial and predictive metrics:

{df.to_string(index=False)}

Also consider these latest quarterly results:

{quarterly_text}

Return:
1. Key strengths
2. Key weaknesses
3. Revenue and net income trends from the latest quarters
4. Important predictive observations
5. Data limitations

Do not provide investment advice.
"""
    return call_llm(prompt, model=DEFAULT_AGENT_MODEL)


def sentiment_agent(tickers, news_df=None):
    news_text = (
        news_df.to_string(index=False)
        if news_df is not None and not news_df.empty
        else "No Yahoo Finance news items were available."
    )

    prompt = f"""
You are the News and Sentiment Agent.

Analyze these Yahoo Finance news items for these tickers:
{", ".join(tickers)}

News items:
{news_text}

Use only the provided news items. Do not invent specific news headlines.

Return:
1. Positive catalysts mentioned or implied by the news
2. Negative catalysts mentioned or implied by the news
3. Market risks that could invalidate predictive scenarios
4. What investors should verify before relying on the news

Do not provide buy/sell recommendations.
"""
    return call_llm(prompt, model=DEFAULT_AGENT_MODEL)


def prediction_agent(df):
    prompt = f"""
You are the Predictive Analysis Agent.

Analyze this predictive scenario data:

{df[[
    "ticker",
    "model",
    "ml_model",
    "expected_return",
    "ml_expected_return",
    "bear_case_return",
    "base_case_return",
    "bull_case_return",
    "probability_positive",
    "ml_probability_positive",
    "annualized_volatility",
    "ml_validation_mae",
    "ml_directional_accuracy",
    "market_beta",
    "growth_beta",
    "small_cap_beta",
    "factor_expected_daily_return",
    "garch_alpha",
    "garch_beta",
    "outlook",
    "prediction_explanation"
]].to_string(index=False)}

Return:
1. Which companies have stronger predictive outlooks
2. How ML expected return compares with Monte Carlo output
3. How walk-forward validation affects confidence
4. How factor exposure and GARCH volatility affect the scenario range
5. Scenario ranges and uncertainty
6. What could make the scenarios wrong

Do not provide investment advice.
"""
    return call_llm(prompt, model=DEFAULT_AGENT_MODEL)


def report_agent(
    df,
    financial_summary,
    sentiment_summary,
    prediction_summary,
    horizon,
    news_df=None,
    quarterly_df=None
):
    news_text = (
        news_df.to_string(index=False)
        if news_df is not None and not news_df.empty
        else "No Yahoo Finance news items were available."
    )
    quarterly_text = (
        quarterly_df.to_string(index=False)
        if quarterly_df is not None and not quarterly_df.empty
        else "No quarterly results were available."
    )

    prompt = f"""
You are the Report Generator Agent.

Create a final predictive investment research brief.

Prediction horizon:
- {horizon}

Company data:
{df.to_string(index=False)}

Latest Yahoo Finance news:
{news_text}

Latest quarterly results:
{quarterly_text}

Financial Agent output:
{financial_summary}

News and Sentiment Agent output:
{sentiment_summary}

Predictive Analysis Agent output:
{prediction_summary}

Generate:
1. Executive Summary
2. Predictive Company Comparison
3. News and Quarterly Results Context
4. Scenario Analysis
5. Main Uncertainties
6. Questions the investor should answer before relying on these projections

Important:
- Educational only
- No buy/sell recommendation
- Explain that the pipeline combines ML expected returns, factor exposure, GARCH volatility, and Monte Carlo scenarios
- Explain that Monte Carlo scenarios are not guarantees
- Distinguish historical price signals, news context, and quarterly fundamentals
- Be clear and structured
"""
    return call_llm(prompt, model=DEFAULT_REPORT_MODEL)
