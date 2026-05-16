import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

load_dotenv()

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=api_key) if api_key else None


def call_llm(prompt):
    if client is None:
        return """
LLM not configured.

Create a .env file with:
GROQ_API_KEY=your_api_key_here
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
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


def financial_agent(df):
    prompt = f"""
You are the Financial Data Agent.

Analyze these public financial metrics:

{df.to_string(index=False)}

Return:
1. Key strengths
2. Key weaknesses
3. Important observations

Do not provide investment advice.
"""
    return call_llm(prompt)


def sentiment_agent(tickers):
    prompt = f"""
You are the News and Sentiment Agent.

Analyze possible market sentiment drivers for these tickers:
{", ".join(tickers)}

Do not invent specific news headlines.

Return:
1. Positive catalysts
2. Negative catalysts
3. Market risks
4. What investors should verify in recent public news

Do not provide buy/sell recommendations.
"""
    return call_llm(prompt)


def risk_agent(df):
    prompt = f"""
You are the Risk Analysis Agent.

Analyze this risk data:

{df[["ticker", "risk_score", "risk_level", "risk_explanation"]].to_string(index=False)}

Return:
1. Which companies are riskier
2. Why they are riskier
3. Portfolio concentration risks
4. Risk mitigation questions

Do not provide investment advice.
"""
    return call_llm(prompt)


def report_agent(
    df,
    financial_summary,
    sentiment_summary,
    risk_summary,
    profile,
    horizon
):
    prompt = f"""
You are the Report Generator Agent.

Create a final investment research brief.

Investor profile:
- Risk profile: {profile}
- Investment horizon: {horizon}

Company data:
{df.to_string(index=False)}

Financial Agent output:
{financial_summary}

News and Sentiment Agent output:
{sentiment_summary}

Risk Agent output:
{risk_summary}

Generate:
1. Executive Summary
2. Company Comparison
3. Main Risks
4. Portfolio Fit
5. Questions the investor should answer before investing

Important:
- Educational only
- No buy/sell recommendation
- Be clear and structured
"""
    return call_llm(prompt)