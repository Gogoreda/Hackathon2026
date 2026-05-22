import json
import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv

from prediction_service import horizon_to_days


load_dotenv()


def _mysql_connector():
    try:
        import mysql.connector
    except Exception:
        return None

    return mysql.connector


def is_database_configured():
    required = [
        "MYSQL_HOST",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
    ]
    return all(os.getenv(key) for key in required)


def get_connection():
    return _connect(use_database=True)


def _connect(use_database=True):
    mysql = _mysql_connector()

    if mysql is None or not is_database_configured():
        return None

    config = {
        "host": os.getenv("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
    }

    if use_database:
        config["database"] = os.getenv("MYSQL_DATABASE")

    return mysql.connect(**config)


def get_or_create_session_id(session_state):
    if "user_session_id" not in session_state:
        session_state.user_session_id = str(uuid.uuid4())

    return session_state.user_session_id


def _safe_float(value):
    try:
        if value is None or value != value:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if value is None or value != value:
            return None
        return int(value)
    except Exception:
        return None


def _json_dumps(data):
    return json.dumps(data, default=str)


def initialize_database():
    connection = _connect(use_database=False)

    if connection is None:
        return False, "Database is not configured."

    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")

    with open(schema_path, "r", encoding="utf-8") as schema_file:
        schema = schema_file.read()

    cursor = connection.cursor()

    for statement in schema.split(";"):
        statement = statement.strip()

        if statement:
            cursor.execute(statement)

    connection.commit()
    cursor.close()
    connection.close()

    return True, "Database schema initialized."


def log_usage(tickers, horizon, simulations, user_session_id=None, app_version=None):
    connection = get_connection()

    if connection is None:
        return None

    query = """
        INSERT INTO usage_logs (
            tickers,
            horizon,
            horizon_days,
            simulations,
            user_session_id,
            app_version
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    values = (
        ",".join(tickers),
        horizon,
        horizon_to_days(horizon),
        simulations,
        user_session_id,
        app_version,
    )

    cursor = connection.cursor()
    cursor.execute(query, values)
    connection.commit()
    inserted_id = cursor.lastrowid
    cursor.close()
    connection.close()

    return inserted_id


def log_prediction_requests(df, horizon, simulations, user_session_id=None):
    connection = get_connection()

    if connection is None:
        return []

    query = """
        INSERT INTO prediction_requests (
            ticker,
            horizon,
            horizon_days,
            current_price,
            predicted_price,
            predicted_return,
            probability_positive,
            bear_case_return,
            base_case_return,
            bull_case_return,
            model_name,
            ml_model,
            ml_validation_mae,
            ml_directional_accuracy,
            garch_alpha,
            garch_beta,
            market_beta,
            growth_beta,
            small_cap_beta,
            outlook,
            features_snapshot,
            market_context_snapshot
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s
        )
    """

    inserted_ids = []
    cursor = connection.cursor()

    for _, row in df.iterrows():
        feature_snapshot = {
            "ml_expected_return": _safe_float(row.get("ml_expected_return")),
            "ml_probability_positive": _safe_float(row.get("ml_probability_positive")),
            "ml_residual_volatility": _safe_float(row.get("ml_residual_volatility")),
            "ml_training_rows": _safe_int(row.get("ml_training_rows")),
            "ml_feature_count": _safe_int(row.get("ml_feature_count")),
            "annualized_volatility": _safe_float(row.get("annualized_volatility")),
            "factor_expected_daily_return": _safe_float(
                row.get("factor_expected_daily_return")
            ),
        }

        market_context_snapshot = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "simulations": simulations,
            "company_name": row.get("company_name"),
            "sector": row.get("sector"),
            "pe_ratio": _safe_float(row.get("pe_ratio")),
            "revenue_growth": _safe_float(row.get("revenue_growth")),
            "profit_margin": _safe_float(row.get("profit_margin")),
            "debt_to_equity": _safe_float(row.get("debt_to_equity")),
        }

        values = (
            row.get("ticker"),
            horizon,
            _safe_int(row.get("horizon_days")) or horizon_to_days(horizon),
            _safe_float(row.get("current_price")),
            _safe_float(row.get("expected_price")),
            _safe_float(row.get("expected_return")),
            _safe_float(row.get("probability_positive")),
            _safe_float(row.get("bear_case_return")),
            _safe_float(row.get("base_case_return")),
            _safe_float(row.get("bull_case_return")),
            row.get("model"),
            row.get("ml_model"),
            _safe_float(row.get("ml_validation_mae")),
            _safe_float(row.get("ml_directional_accuracy")),
            _safe_float(row.get("garch_alpha")),
            _safe_float(row.get("garch_beta")),
            _safe_float(row.get("market_beta")),
            _safe_float(row.get("growth_beta")),
            _safe_float(row.get("small_cap_beta")),
            row.get("outlook"),
            _json_dumps(feature_snapshot),
            _json_dumps(market_context_snapshot),
        )

        cursor.execute(query, values)
        inserted_ids.append(cursor.lastrowid)

    connection.commit()
    cursor.close()
    connection.close()

    return inserted_ids
