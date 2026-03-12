"""数据库模块 - PostgreSQL操作"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """数据库配置"""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database = os.getenv("DB_NAME", "forex_data")
        self.user = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "postgres")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def psycopg2_params(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }


class Database:
    """PostgreSQL数据库操作类"""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._connection = None

    def connect(self):
        """建立数据库连接"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(**self.config.psycopg2_params)
        return self._connection

    def close(self):
        """关闭数据库连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    def execute(self, query: str, params: tuple = None):
        """执行SQL语句"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor
        except Exception as e:
            conn.rollback()
            raise e

    def execute_many(self, query: str, data: List[tuple]):
        """批量执行SQL语句"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            execute_values(cursor, query, data)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()

    def fetch_all(self, query: str, params: tuple = None) -> List[tuple]:
        """查询所有记录"""
        cursor = self.execute(query, params)
        result = cursor.fetchall()
        cursor.close()
        return result

    def fetch_one(self, query: str, params: tuple = None) -> Optional[tuple]:
        """查询单条记录"""
        cursor = self.execute(query, params)
        result = cursor.fetchone()
        cursor.close()
        return result

    def query_to_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        """查询并返回DataFrame"""
        conn = self.connect()
        return pd.read_sql(query, conn, params=params)

    def insert_ohlc(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]],
    ):
        """批量插入K线数据"""
        query = """
            INSERT INTO ohlc_data (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, timeframe, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
        """
        values = [
            (
                symbol,
                timeframe,
                row["time"],
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row.get("volume", 0),
            )
            for row in data
        ]
        self.execute_many(query, values)

    def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """获取K线数据"""
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlc_data
            WHERE symbol = %s AND timeframe = %s
        """
        params = [symbol, timeframe]

        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)

        query += " ORDER BY timestamp ASC LIMIT %s"
        params.append(limit)

        return self.query_to_dataframe(query, tuple(params))

    def insert_trade(self, trade_data: Dict[str, Any]) -> int:
        """插入交易记录，返回ID"""
        query = """
            INSERT INTO trades (
                symbol, open_time, close_time, open_price, close_price,
                direction, pnl, pnl_ratio,
                open_ma_cross, open_rsi_overbought, open_rsi_oversold,
                open_macd_signal, open_bollinger_position,
                close_ma_cross, close_rsi_overbought, close_rsi_oversold,
                close_macd_signal, close_bollinger_position,
                max_floating_profit, max_floating_loss, max_drawdown, holding_bars,
                h1_ma_trend, d1_rsi_14, d1_atr_ratio
            ) VALUES (
                %(symbol)s, %(open_time)s, %(close_time)s, %(open_price)s, %(close_price)s,
                %(direction)s, %(pnl)s, %(pnl_ratio)s,
                %(open_ma_cross)s, %(open_rsi_overbought)s, %(open_rsi_oversold)s,
                %(open_macd_signal)s, %(open_bollinger_position)s,
                %(close_ma_cross)s, %(close_rsi_overbought)s, %(close_rsi_oversold)s,
                %(close_macd_signal)s, %(close_bollinger_position)s,
                %(max_floating_profit)s, %(max_floating_loss)s, %(max_drawdown)s, %(holding_bars)s,
                %(h1_ma_trend)s, %(d1_rsi_14)s, %(d1_atr_ratio)s
            )
            RETURNING id
        """
        cursor = self.execute(query, trade_data)
        result = cursor.fetchone()
        cursor.close()
        return result[0] if result else None

    def get_trades(self, symbol: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """获取交易记录"""
        query = "SELECT * FROM trades ORDER BY open_time DESC LIMIT %s"
        params = [limit]
        if symbol:
            query = "SELECT * FROM trades WHERE symbol = %s ORDER BY open_time DESC LIMIT %s"
            params = [symbol, limit]
        return self.query_to_dataframe(query, tuple(params))

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """获取最新K线时间戳"""
        query = """
            SELECT MAX(timestamp) FROM ohlc_data
            WHERE symbol = %s AND timeframe = %s
        """
        result = self.fetch_one(query, (symbol, timeframe))
        return result[0] if result and result[0] else None


def init_database(config: Optional[DatabaseConfig] = None) -> Database:
    """初始化数据库连接"""
    db = Database(config)
    db.connect()
    return db
