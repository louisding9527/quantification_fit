"""MT5数据获取模块"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入MT5，如果失败则提供模拟数据
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available, using mock data")


class TimeFrame(Enum):
    """时间周期枚举"""

    M1 = "1M"
    M5 = "5M"
    M15 = "15M"
    M30 = "30M"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"

    @property
    def mt5_value(self):
        """MT5对应的时间周期值"""
        mapping = {
            "1M": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
            "5M": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
            "15M": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
            "30M": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
            "1H": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 16385,
            "4H": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 16388,
            "1D": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 16408,
            "1W": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 32769,
        }
        return mapping.get(self.value, 16385)


class MT5Fetcher:
    """MetaTrader 5数据获取器"""

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        """连接MT5"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
            return False

        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False

            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                mt5.shutdown()
                return False

            self.connected = True
            logger.info(f"Connected to MT5, account: {account_info.login}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False

    def disconnect(self):
        """断开MT5连接"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")

    def get_symbols(self) -> List[str]:
        """获取可用交易品种"""
        if not MT5_AVAILABLE or not self.connected:
            return ["EURUSD", "GBPUSD", "USDJPY"]

        symbols = mt5.symbols_get()
        return [s.name for s in symbols if s.visible]

    def get_ohlc(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        num_bars: int = 1000,
    ) -> pd.DataFrame:
        """获取K线数据

        Args:
            symbol: 交易品种
            timeframe: 时间周期
            start_time: 开始时间
            end_time: 结束时间
            num_bars: K线数量

        Returns:
            DataFrame包含 time, open, high, low, close, volume
        """
        if not MT5_AVAILABLE or not self.connected:
            return self._generate_mock_data(symbol, timeframe, num_bars)

        # 转换时间
        if start_time:
            start_time = int(start_time.timestamp())
        else:
            start_time = int((datetime.now() - timedelta(days=30)).timestamp())

        if end_time:
            end_time = int(end_time.timestamp())

        # 获取数据
        rates = mt5.copy_rates_from_pos(
            symbol,
            timeframe.mt5_value,
            0,
            num_bars,
        )

        if rates is None or len(rates) == 0:
            logger.warning(f"No data returned for {symbol} {timeframe.value}")
            return pd.DataFrame()

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "time", "open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]]

        # 过滤时间范围
        if start_time:
            df = df[df["time"] >= pd.to_datetime(start_time, unit="s")]
        if end_time:
            df = df[df["time"] <= pd.to_datetime(end_time, unit="s")]

        return df.reset_index(drop=True)

    def get_ohlc_range(
        self,
        symbol: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """获取指定时间范围的K线数据"""
        if not MT5_AVAILABLE or not self.connected:
            days = (end_time - start_time).days
            num_bars = days * 24 if "H" in timeframe.value else days
            return self._generate_mock_data(symbol, timeframe, num_bars)

        rates = mt5.copy_rates_range(
            symbol,
            timeframe.mt5_value,
            start_time,
            end_time,
        )

        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"time": "time", "open": "open", "high": "high", "low": "low", "close": "close", "tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]]

        return df.reset_index(drop=True)

    def _generate_mock_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        num_bars: int,
    ) -> pd.DataFrame:
        """生成模拟数据用于测试"""
        logger.info(f"Generating mock data for {symbol} {timeframe.value}")

        # 基础价格
        base_price = 1.1000 if "EUR" in symbol else 1.2500 if "GBP" in symbol else 150.0

        # 生成时间序列
        hours_per_bar = int(timeframe.value.replace("H", "").replace("D", "24"))
        start_date = datetime.now() - timedelta(hours=hours_per_bar * num_bars)
        times = [start_date + timedelta(hours=hours_per_bar * i) for i in range(num_bars)]

        # 生成价格数据（随机游走）
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.001, num_bars)
        prices = base_price * np.exp(np.cumsum(returns))

        # 生成OHLC
        data = []
        for i, (t, close) in enumerate(zip(times, prices)):
            volatility = abs(np.random.normal(0, 0.0005))
            high = close + abs(np.random.normal(0, volatility))
            low = close - abs(np.random.normal(0, volatility))
            open_price = np.random.uniform(low, high)

            data.append({
                "time": t,
                "open": round(open_price, 5),
                "high": round(high, 5),
                "low": round(low, 5),
                "close": round(close, 5),
                "volume": int(np.random.randint(1000, 10000)),
            })

        return pd.DataFrame(data)


def fetch_multi_timeframe_data(
    symbol: str,
    timeframes: List[TimeFrame],
    num_bars: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """获取多周期K线数据

    Args:
        symbol: 交易品种
        timeframes: 时间周期列表
        num_bars: 每个周期获取的K线数量

    Returns:
        字典，key为时间周期，value为DataFrame
    """
    fetcher = MT5Fetcher()

    if MT5_AVAILABLE:
        if not fetcher.connect():
            logger.warning("Failed to connect to MT5, using mock data")
        else:
            result = {}
            for tf in timeframes:
                df = fetcher.get_ohlc(symbol, tf, num_bars=num_bars)
                result[tf.value] = df
            fetcher.disconnect()
            return result
    else:
        result = {}
        for tf in timeframes:
            df = fetcher._generate_mock_data(symbol, tf, num_bars)
            result[tf.value] = df
        return result
