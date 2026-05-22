from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus.tables import Table, TableStyle
from reportlab.lib import colors


def generate_pdf_report(
    filename,
    df,
    financial_summary,
    sentiment_summary,
    prediction_summary,
    final_report,
    news_df=None,
    quarterly_df=None
):

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()
    elements = []

    title = Paragraph(
        "AlphaLens Predictive Investment Research Brief",
        styles["Title"]
    )

    elements.append(title)
    elements.append(Spacer(1, 20))

    # Table
    elements.append(
        Paragraph("Company Comparison", styles["Heading2"])
    )

    table_df = df[
        [
            "ticker",
            "model",
            "ml_model",
            "current_price",
            "expected_price",
            "expected_return",
            "probability_positive",
            "ml_validation_mae",
            "ml_directional_accuracy",
            "annualized_volatility",
            "outlook"
        ]
    ].copy()

    table_data = [table_df.columns.tolist()]

    for _, row in table_df.iterrows():
        table_data.append(row.tolist())

    table = Table(table_data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ])
    )

    elements.append(table)
    elements.append(Spacer(1, 20))

    if news_df is not None and not news_df.empty:
        elements.append(
            Paragraph("Latest Yahoo Finance News", styles["Heading2"])
        )

        news_table_df = news_df[
            [
                "ticker",
                "published_date",
                "publisher",
                "title",
            ]
        ].head(15).copy()

        news_table_data = [news_table_df.columns.tolist()]

        for _, row in news_table_df.iterrows():
            news_table_data.append(row.tolist())

        news_table = Table(news_table_data)
        news_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ])
        )

        elements.append(news_table)
        elements.append(Spacer(1, 20))

    if quarterly_df is not None and not quarterly_df.empty:
        elements.append(
            Paragraph("Latest Quarterly Results", styles["Heading2"])
        )

        quarterly_table_df = quarterly_df[
            [
                "ticker",
                "quarter",
                "revenue",
                "gross_profit",
                "operating_income",
                "net_income",
            ]
        ].copy()

        quarterly_table_data = [quarterly_table_df.columns.tolist()]

        for _, row in quarterly_table_df.iterrows():
            quarterly_table_data.append(row.tolist())

        quarterly_table = Table(quarterly_table_data)
        quarterly_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ])
        )

        elements.append(quarterly_table)
        elements.append(Spacer(1, 20))

    # Sections
    sections = [
        ("Financial Agent Summary", financial_summary),
        ("News & Sentiment Agent Summary", sentiment_summary),
        ("Predictive Analysis Agent Summary", prediction_summary),
        ("Final Predictive Brief", final_report),
    ]

    for title_text, content in sections:

        elements.append(
            Paragraph(title_text, styles["Heading2"])
        )

        cleaned_content = content.replace("\n", "<br/>")

        elements.append(
            Paragraph(cleaned_content, styles["BodyText"])
        )

        elements.append(Spacer(1, 20))

    disclaimer = Paragraph(
        "Disclaimer: Educational predictive scenarios only. Not financial advice.",
        styles["Italic"]
    )

    elements.append(disclaimer)

    doc.build(elements)
