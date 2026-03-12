-- 数据库初始化脚本
-- PostgreSQL

-- 创建数据库
-- CREATE DATABASE forex_data;

-- K线数据表
CREATE TABLE IF NOT EXISTS ohlc_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_timeframe
    ON ohlc_data(symbol, timeframe, timestamp);

-- 交易记录表
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    open_price REAL NOT NULL,
    close_price REAL NOT NULL,
    direction VARCHAR(10) NOT NULL,  -- LONG/SHORT
    pnl REAL NOT NULL,               -- 盈亏（点数）
    pnl_ratio REAL NOT NULL,         -- 盈亏比

    -- 开仓时技术指标
    open_ma_cross BOOLEAN,
    open_rsi_overbought BOOLEAN,
    open_rsi_oversold BOOLEAN,
    open_macd_signal VARCHAR(10),
    open_bollinger_position REAL,

    -- 平仓时技术指标
    close_ma_cross BOOLEAN,
    close_rsi_overbought BOOLEAN,
    close_rsi_oversold BOOLEAN,
    close_macd_signal VARCHAR(10),
    close_bollinger_position REAL,

    -- 持仓期间统计
    max_floating_profit REAL,
    max_floating_loss REAL,
    max_drawdown REAL,
    holding_bars INTEGER,

    -- 辅助周期指标（用于规则过滤）
    h1_ma_trend BOOLEAN,
    d1_rsi_14 REAL,
    d1_atr_ratio REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 交易记录索引
CREATE INDEX IF NOT EXISTS idx_trades_symbol
    ON trades(symbol);

CREATE INDEX IF NOT EXISTS idx_trades_open_time
    ON trades(open_time);

-- 训练数据表（可选，用于存储生成的特征数据）
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    label INTEGER NOT NULL,  -- 0: SELL, 1: HOLD, 2: BUY

    -- 主周期特征
    ma5 REAL, ma10 REAL, ma20 REAL, ma60 REAL,
    ema12 REAL, ema26 REAL,
    macd REAL, macd_signal REAL, macd_hist REAL,
    rsi_6 REAL, rsi_12 REAL, rsi_14 REAL, rsi_24 REAL,
    stoch_k REAL, stoch_d REAL, kdj_j REAL,
    bb_position REAL,
    atr_14 REAL, atr_ratio REAL,
    adx REAL, adx_pos REAL, adx_neg REAL,
    ma5_above_ma20 INTEGER,
    rsi_overbought INTEGER, rsi_oversold INTEGER,
    macd_bullish INTEGER,
    price_position REAL, volatility REAL,

    -- 辅助周期特征
    h1_ma_trend INTEGER,
    d1_rsi_14 REAL, d1_atr_ratio REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_data_symbol_timestamp
    ON training_data(symbol, timestamp);

CREATE INDEX IF NOT EXISTS idx_training_data_label
    ON training_data(label);
