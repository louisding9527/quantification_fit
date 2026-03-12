"""技术指标计算模块"""

from typing import Optional
import pandas as pd
import numpy as np

# 尝试导入pandas_ta
try:
    import pandas_ta as ta

    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False


class IndicatorCalculator:
    """技术指标计算器"""

    def __init__(self, df: pd.DataFrame):
        """初始化计算器

        Args:
            df: 包含OHLC数据的DataFrame，必须包含 open, high, low, close 列
        """
        self.df = df.copy()
        self._ensure_columns()

    def _ensure_columns(self):
        """确保必要的列存在"""
        required = ["open", "high", "low", "close"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def calculate_all(self) -> pd.DataFrame:
        """计算所有常用指标"""
        if PANDAS_TA_AVAILABLE:
            return self._calculate_with_pandas_ta()
        else:
            return self._calculate_manual()

    def _calculate_with_pandas_ta(self) -> pd.DataFrame:
        """使用pandas_ta计算指标"""
        df = self.df.copy()

        # 趋势指标
        # MA
        df["ma5"] = ta.sma(df["close"], length=5)
        df["ma10"] = ta.sma(df["close"], length=10)
        df["ma20"] = ta.sma(df["close"], length=20)
        df["ma60"] = ta.sma(df["close"], length=60)

        # EMA
        df["ema12"] = ta.ema(df["close"], length=12)
        df["ema26"] = ta.ema(df["close"], length=26)

        # MACD
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]
        df["macd_hist"] = macd["MACDh_12_26_9"]

        # ADX
        adx = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx["ADX_14"]
        df["adx_pos"] = adx["DMP_14"]
        df["adx_neg"] = adx["DMN_14"]

        # 震荡指标
        # RSI
        df["rsi_6"] = ta.rsi(df["close"], length=6)
        df["rsi_12"] = ta.rsi(df["close"], length=12)
        df["rsi_14"] = ta.rsi(df["close"], length=14)
        df["rsi_24"] = ta.rsi(df["close"], length=24)

        # Stochastic
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]

        # KDJ (基于Stochastic)
        df["kdj_k"] = df["stoch_k"]
        df["kdj_d"] = df["stoch_d"]
        df["kdj_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]

        # 波动指标
        # Bollinger Bands
        bb = ta.bbands(df["close"], length=20, std=2)
        df["bb_upper"] = bb["BBU_20_2.0"]
        df["bb_middle"] = bb["BBM_20_2.0"]
        df["bb_lower"] = bb["BBL_20_2.0"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["atr_ratio"] = df["atr_14"] / df["close"] * 100  # ATR占比

        # Keltner Channel
        kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=2)
        df["kc_upper"] = kc["KCU_20_2"]
        df["kc_middle"] = kc["KCM_20_2"]
        df["kc_lower"] = kc["KCL_20_2"]

        # 量价指标
        if "volume" in df.columns:
            df["obv"] = ta.obv(df["close"], df["volume"])
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])

        # 补充手动计算的指标
        df = self._add_manual_indicators(df)

        return df

    def _calculate_manual(self) -> pd.DataFrame:
        """手动计算指标（不使用pandas_ta）"""
        df = self.df.copy()

        # MA
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()

        # EMA
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["macd"] = df["ema12"] - df["ema26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # RSI
        for period in [6, 12, 14, 24]:
            df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # ATR
        df["atr_14"] = self._calculate_atr(df["high"], df["low"], df["close"], 14)
        df["atr_ratio"] = df["atr_14"] / df["close"] * 100

        # Stochastic
        low_14 = df["low"].rolling(window=14).min()
        high_14 = df["high"].rolling(window=14).max()
        df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()
        df["kdj_k"] = df["stoch_k"]
        df["kdj_d"] = df["stoch_d"]
        df["kdj_j"] = 3 * df["stoch_k"] - 2 * df["stoch_d"]

        # ADX
        df["adx"] = self._calculate_adx(df["high"], df["low"], df["close"], 14)
        df["adx_pos"] = df["adx"]  # 简化
        df["adx_neg"] = df["adx"]  # 简化

        # OBV
        if "volume" in df.columns:
            df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

        df = self._add_manual_indicators(df)

        return df

    def _add_manual_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加手动计算的辅助指标"""
        # MA交叉信号
        df["ma5_above_ma20"] = (df["ma5"] > df["ma20"]).astype(int)
        df["ma_cross"] = df["ma5_above_ma20"].diff()

        # RSI超买超卖
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_neutral"] = ((df["rsi_14"] >= 30) & (df["rsi_14"] <= 70)).astype(int)

        # MACD信号
        df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
        df["macd_cross"] = df["macd_bullish"].diff()

        # 价格位置
        df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

        # 波动率
        df["volatility"] = df["close"].pct_change().rolling(window=20).std()

        return df

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """计算RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算ADX（简化版）"""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = IndicatorCalculator._calculate_atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """便捷函数：计算技术指标

    Args:
        df: 包含OHLC数据的DataFrame

    Returns:
        包含所有技术指标的DataFrame
    """
    calculator = IndicatorCalculator(df)
    return calculator.calculate_all()


def get_feature_columns() -> list:
    """获取特征列名列表（用于模型训练）"""
    return [
        # MA
        "ma5", "ma10", "ma20", "ma60",
        # EMA
        "ema12", "ema26",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # RSI
        "rsi_6", "rsi_12", "rsi_14", "rsi_24",
        # Stochastic
        "stoch_k", "stoch_d", "kdj_j",
        # Bollinger
        "bb_position",
        # ATR
        "atr_14", "atr_ratio",
        # ADX
        "adx", "adx_pos", "adx_neg",
        # 辅助指标
        "ma5_above_ma20", "rsi_overbought", "rsi_oversold",
        "macd_bullish", "price_position", "volatility",
    ]
