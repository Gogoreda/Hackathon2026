CREATE DATABASE IF NOT EXISTS alphalens;

USE alphalens;

CREATE TABLE IF NOT EXISTS usage_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tickers TEXT NOT NULL,
    horizon VARCHAR(32) NOT NULL,
    horizon_days INT NOT NULL,
    simulations INT NOT NULL,
    user_session_id VARCHAR(128),
    app_version VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS prediction_requests (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ticker VARCHAR(32) NOT NULL,
    horizon VARCHAR(32) NOT NULL,
    horizon_days INT NOT NULL,
    current_price DOUBLE,
    predicted_price DOUBLE,
    predicted_return DOUBLE,
    probability_positive DOUBLE,
    bear_case_return DOUBLE,
    base_case_return DOUBLE,
    bull_case_return DOUBLE,
    model_name VARCHAR(255),
    ml_model VARCHAR(255),
    ml_validation_mae DOUBLE,
    ml_directional_accuracy DOUBLE,
    garch_alpha DOUBLE,
    garch_beta DOUBLE,
    market_beta DOUBLE,
    growth_beta DOUBLE,
    small_cap_beta DOUBLE,
    outlook VARCHAR(64),
    features_snapshot JSON,
    market_context_snapshot JSON,
    resolved BOOLEAN DEFAULT FALSE,
    INDEX idx_prediction_ticker_created (ticker, created_at),
    INDEX idx_prediction_resolution (resolved, created_at)
);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    prediction_request_id BIGINT NOT NULL,
    resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_price DOUBLE,
    actual_return DOUBLE,
    prediction_error DOUBLE,
    direction_correct BOOLEAN,
    FOREIGN KEY (prediction_request_id)
        REFERENCES prediction_requests(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_versions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type VARCHAR(255) NOT NULL,
    feature_set_version VARCHAR(64),
    training_universe TEXT,
    validation_start DATE,
    validation_end DATE,
    sharpe DOUBLE,
    mae DOUBLE,
    directional_accuracy DOUBLE,
    max_drawdown DOUBLE,
    artifact_path TEXT,
    approved BOOLEAN DEFAULT FALSE,
    notes TEXT
);
