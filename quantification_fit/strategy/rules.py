"""交易规则定义模块"""

from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class TradeDirection(Enum):
    """交易方向"""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class SignalType(Enum):
    """信号类型"""

    BUY = 0  # 买入
    HOLD = 1  # 持有
    SELL = 2  # 卖出


@dataclass
class TradeConfig:
    """交易配置"""

    # 止盈止损（点数）
    take_profit: float = 100
    stop_loss: float = 50

    # RSI阈值
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    # MA周期
    fast_ma: int = 5
    slow_ma: int = 20


class TradingRules:
    """交易规则类"""

    def __init__(self, config: Optional[TradeConfig] = None):
        self.config = config or TradeConfig()

    def generate_signal(self, df: pd.DataFrame, idx: int) -> TradeDirection:
        """生成交易信号

        Args:
            df: 包含技术指标的DataFrame
            idx: 当前索引

        Returns:
            TradeDirection: LONG/SHORT/NONE
        """
        if idx < 2:
            return TradeDirection.NONE

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        # 检查必要字段
        required_cols = ["ma5", "ma20", "rsi_14", "macd", "macd_signal"]
        if not all(col in df.columns for col in required_cols):
            return TradeDirection.NONE

        # MA金叉/死叉
        ma5_above_ma20_now = row["ma5"] > row["ma20"]
        ma5_above_ma20_prev = prev_row["ma5"] > prev_row["ma20"]

        ma_cross_bullish = ma5_above_ma20_now and not ma5_above_ma20_prev  # 金叉
        ma_cross_bearish = not ma5_above_ma20_now and ma5_above_ma20_prev  # 死叉

        # RSI条件
        rsi = row["rsi_14"]
        rsi_valid = self.config.rsi_oversold < rsi < self.config.rsi_overbought

        # MACD条件
        macd_bullish = row["macd"] > row["macd_signal"]
        macd_bearish = row["macd"] < row["macd_signal"]

        # 多头开仓: MA金叉 + RSI在合理区间
        if ma_cross_bullish and rsi_valid and macd_bullish:
            return TradeDirection.LONG

        # 空头开仓: MA死叉 + RSI在合理区间
        if ma_cross_bearish and rsi_valid and macd_bearish:
            return TradeDirection.SHORT

        return TradeDirection.NONE

    def should_close(
        self,
        df: pd.DataFrame,
        idx: int,
        direction: TradeDirection,
        entry_price: float,
    ) -> Tuple[bool, str]:
        """判断是否应该平仓

        Args:
            df: 包含技术指标的DataFrame
            idx: 当前索引
            direction: 当前持仓方向
            entry_price: 开仓价格

        Returns:
            (是否平仓, 平仓原因)
        """
        row = df.iloc[idx]

        current_price = row["close"]
        price_change = current_price - entry_price

        # 止盈
        if direction == TradeDirection.LONG:
            if price_change >= self.config.take_profit:
                return True, "take_profit"
            if price_change <= -self.config.stop_loss:
                return True, "stop_loss"
        elif direction == TradeDirection.SHORT:
            if price_change <= -self.config.take_profit:
                return True, "take_profit"
            if price_change >= self.config.stop_loss:
                return True, "stop_loss"

        # RSI超买/超卖
        rsi = row.get("rsi_14", 50)
        if direction == TradeDirection.LONG and rsi > self.config.rsi_overbought:
            return True, "rsi_overbought"
        if direction == TradeDirection.SHORT and rsi < self.config.rsi_oversold:
            return True, "rsi_oversold"

        # MA反向交叉
        if idx >= 2:
            prev_row = df.iloc[idx - 1]
            ma5_above_now = row["ma5"] > row["ma20"]
            ma5_above_prev = prev_row["ma5"] > prev_row["ma20"]

            if direction == TradeDirection.LONG and not ma5_above_now and ma5_above_prev:
                return True, "ma_death_cross"
            if direction == TradeDirection.SHORT and ma5_above_now and not ma5_above_prev:
                return True, "ma_golden_cross"

        return False, ""

    def get_open_features(self, df: pd.DataFrame, idx: int) -> dict:
        """获取开仓时的特征"""
        row = df.iloc[idx]

        features = {
            "open_ma_cross": row.get("ma5", 0) > row.get("ma20", 0),
            "open_rsi_overbought": row.get("rsi_overbought", 0) == 1,
            "open_rsi_oversold": row.get("rsi_oversold", 0) == 1,
            "open_macd_signal": "bullish" if row.get("macd_bullish", 0) == 1 else "bearish",
            "open_bollinger_position": row.get("bb_position", 0.5),
        }

        return features

    def get_close_features(self, df: pd.DataFrame, idx: int) -> dict:
        """获取平仓时的特征"""
        row = df.iloc[idx]

        features = {
            "close_ma_cross": row.get("ma5", 0) > row.get("ma20", 0),
            "close_rsi_overbought": row.get("rsi_overbought", 0) == 1,
            "close_rsi_oversold": row.get("rsi_oversold", 0) == 1,
            "close_macd_signal": "bullish" if row.get("macd_bullish", 0) == 1 else "bearish",
            "close_bollinger_position": row.get("bb_position", 0.5),
        }

        return features


def df_to_signal(df: pd.DataFrame, idx: int) -> SignalType:
    """DataFrame索引转换为信号类型（三分类）

    用于模型训练标签

    Args:
        df: 包含技术指标的DataFrame
        idx: 当前索引

    Returns:
        SignalType: BUY/HOLD/SELL
    """
    if idx < 1:
        return SignalType.HOLD

    row = df.iloc[idx]
    prev_row = df.iloc[idx - 1]

    # MA交叉
    ma_cross_up = (row["ma5"] > row["ma20"]) and (prev_row["ma5"] <= prev_row["ma20"])
    ma_cross_down = (row["ma5"] < row["ma20"]) and (prev_row["ma5"] >= prev_row["ma20"])

    # RSI
    rsi = row.get("rsi_14", 50)

    # MACD
    macd_bullish = row["macd"] > row["macd_signal"]
    macd_bearish = row["macd"] < row["macd_signal"]

    # 买入信号: 金叉 + RSI未超买 + MACD多头
    if ma_cross_up and rsi < 65 and macd_bullish:
        return SignalType.BUY

    # 卖出信号: 死叉 + RSI未超卖 + MACD空头
    if ma_cross_down and rsi > 35 and macd_bearish:
        return SignalType.SELL

    return SignalType.HOLD
