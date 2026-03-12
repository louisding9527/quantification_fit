"""特征生成与标注模块"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from ..data.indicators import calculate_indicators, get_feature_columns
from ..strategy.rules import TradingRules, TradeDirection, TradeConfig


@dataclass
class TradeRecord:
    """交易记录"""

    symbol: str
    open_time: datetime
    close_time: datetime
    open_price: float
    close_price: float
    direction: str
    pnl: float  # 点数
    pnl_ratio: float  # 盈亏比

    # 开仓特征
    open_ma_cross: bool
    open_rsi_overbought: bool
    open_rsi_oversold: bool
    open_macd_signal: str
    open_bollinger_position: float

    # 平仓特征
    close_ma_cross: bool
    close_rsi_overbought: bool
    close_rsi_oversold: bool
    close_macd_signal: str
    close_bollinger_position: float

    # 持仓统计
    max_floating_profit: float
    max_floating_loss: float
    max_drawdown: float
    holding_bars: int

    # 辅助周期指标
    h1_ma_trend: Optional[bool] = None
    d1_rsi_14: Optional[float] = None
    d1_atr_ratio: Optional[float] = None


class LabelGenerator:
    """标注生成器 - 根据交易规则生成训练数据"""

    def __init__(
        self,
        rules: Optional[TradingRules] = None,
        symbol: str = "EURUSD",
    ):
        self.rules = rules or TradingRules()
        self.symbol = symbol

    def generate_trades(
        self,
        df: pd.DataFrame,
        h1_df: Optional[pd.DataFrame] = None,
        d1_df: Optional[pd.DataFrame] = None,
    ) -> List[TradeRecord]:
        """生成交易记录

        Args:
            df: 主周期K线数据（含技术指标）
            h1_df: H1周期数据（可选）
            d1_df: D1周期数据（可选）

        Returns:
            交易记录列表
        """
        # 计算技术指标
        if "ma5" not in df.columns:
            df = calculate_indicators(df)

        # 预处理辅助周期数据
        h1_indicators = None
        d1_indicators = None

        if h1_df is not None and len(h1_df) > 0:
            if "ma5" not in h1_df.columns:
                h1_indicators = calculate_indicators(h1_df)
            else:
                h1_indicators = h1_df

        if d1_df is not None and len(d1_df) > 0:
            if "ma5" not in d1_df.columns:
                d1_indicators = calculate_indicators(d1_df)
            else:
                d1_indicators = d1_df

        trades = []
        current_position = None
        entry_price = 0.0
        entry_time = None
        max_profit = 0.0
        max_loss = 0.0

        for idx in range(len(df)):
            row = df.iloc[idx]

            # 如果有持仓，检查是否平仓
            if current_position is not None:
                should_close, close_reason = self.rules.should_close(
                    df, idx, current_position, entry_price
                )

                if should_close:
                    close_price = row["close"]
                    pnl = self._calculate_pnl(
                        entry_price, close_price, current_position
                    )

                    # 获取平仓特征
                    close_features = self.rules.get_close_features(df, idx)

                    # 获取辅助周期指标
                    aux_features = self._get_aux_features(
                        idx, row["time"], h1_indicators, d1_indicators
                    )

                    trade = TradeRecord(
                        symbol=self.symbol,
                        open_time=entry_time,
                        close_time=row["time"],
                        open_price=entry_price,
                        close_price=close_price,
                        direction=current_position.value,
                        pnl=pnl,
                        pnl_ratio=pnl / self.rules.config.stop_loss
                        if self.rules.config.stop_loss > 0
                        else 0,
                        open_ma_cross=self._get_last_feature("open_ma_cross", idx, df),
                        open_rsi_overbought=self._get_last_feature(
                            "open_rsi_overbought", idx, df
                        ),
                        open_rsi_oversold=self._get_last_feature(
                            "open_rsi_oversold", idx, df
                        ),
                        open_macd_signal=self._get_last_feature(
                            "open_macd_signal", idx, df
                        ),
                        open_bollinger_position=self._get_last_feature(
                            "open_bollinger_position", idx, df
                        ),
                        close_ma_cross=close_features["close_ma_cross"],
                        close_rsi_overbought=close_features["close_rsi_overbought"],
                        close_rsi_oversold=close_features["close_rsi_oversold"],
                        close_macd_signal=close_features["close_macd_signal"],
                        close_bollinger_position=close_features["close_bollinger_position"],
                        max_floating_profit=max_profit,
                        max_floating_loss=max_loss,
                        max_drawdown=max(max_loss, abs(max_profit)),
                        holding_bars=idx - self._entry_idx,
                        **aux_features,
                    )
                    trades.append(trade)

                    # 清空持仓
                    current_position = None
                    entry_price = 0.0
                    entry_time = None
                    max_profit = 0.0
                    max_loss = 0.0
                else:
                    # 更新浮动盈亏
                    floating = self._calculate_pnl(
                        entry_price, row["close"], current_position
                    )
                    if floating > max_profit:
                        max_profit = floating
                    if floating < max_loss:
                        max_loss = floating

            # 如果没有持仓，检查是否开仓
            if current_position is None:
                signal = self.rules.generate_signal(df, idx)

                if signal != TradeDirection.NONE:
                    current_position = signal
                    entry_price = row["close"]
                    entry_time = row["time"]
                    self._entry_idx = idx
                    max_profit = 0.0
                    max_loss = 0.0

        return trades

    def _calculate_pnl(
        self, entry: float, current: float, direction: TradeDirection
    ) -> float:
        """计算盈亏（点数）"""
        if direction == TradeDirection.LONG:
            return (current - entry) * 10000  # 假设EURUSD，1点=0.0001
        else:
            return (entry - current) * 10000

    def _get_last_feature(self, feature_name: str, idx: int, df: pd.DataFrame):
        """获取开仓时的特征值"""
        if hasattr(self, "_entry_features"):
            return self._entry_features.get(feature_name, 0)
        return df.iloc[idx].get(feature_name, 0)

    def _get_aux_features(
        self,
        idx: int,
        current_time: pd.Timestamp,
        h1_df: Optional[pd.DataFrame],
        d1_df: Optional[pd.DataFrame],
    ) -> dict:
        """获取辅助周期指标"""
        features = {}

        # H1趋势
        if h1_df is not None and len(h1_df) > 0:
            # 找到最近对应时间的H1数据
            h1_idx = h1_df[h1_df["time"] <= current_time].index
            if len(h1_idx) > 0:
                last_h1 = h1_df.iloc[h1_idx[-1]]
                features["h1_ma_trend"] = last_h1.get("ma5", 0) > last_h1.get("ma20", 0)

        # D1指标
        if d1_df is not None and len(d1_df) > 0:
            d1_idx = d1_df[d1_df["time"] <= current_time].index
            if len(d1_idx) > 0:
                last_d1 = d1_df.iloc[d1_idx[-1]]
                features["d1_rsi_14"] = last_d1.get("rsi_14", 50)
                features["d1_atr_ratio"] = last_d1.get("atr_ratio", 0)

        return features

    def trades_to_dataframe(self, trades: List[TradeRecord]) -> pd.DataFrame:
        """将交易记录转换为DataFrame"""
        if not trades:
            return pd.DataFrame()

        data = []
        for trade in trades:
            record = asdict(trade)
            record["open_time"] = trade.open_time
            record["close_time"] = trade.close_time
            data.append(record)

        return pd.DataFrame(data)


class FeatureGenerator:
    """特征生成器 - 用于模型训练"""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns or get_feature_columns()

    def generate_labels(
        self,
        df: pd.DataFrame,
        look_ahead: int = 10,
    ) -> pd.DataFrame:
        """生成训练标签

        Args:
            df: 包含技术指标的DataFrame
            look_ahead: 向前看的K线数

        Returns:
            包含特征的DataFrame
        """
        if "ma5" not in df.columns:
            df = calculate_indicators(df)

        df = df.copy()

        # 未来收益
        df["future_return"] = df["close"].shift(-look_ahead) / df["close"] - 1

        # 未来最高价（用于计算最大浮盈）
        df["future_high"] = df["high"].shift(-1).rolling(window=look_ahead).max()
        df["future_low"] = df["low"].shift(-1).rolling(window=look_ahead).min()

        # 未来最低价（用于计算最大回撤）
        df["max_floating_profit"] = (df["future_high"] - df["close"]) / df["close"] * 10000
        df["max_floating_loss"] = (df["close"] - df["future_low"]) / df["close"] * 10000

        # 标签: 基于未来收益
        df["label"] = pd.cut(
            df["future_return"],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[2, 1, 0],  # SELL, HOLD, BUY
        )

        return df

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据

        Args:
            df: 包含特征的DataFrame
            drop_na: 是否删除NaN

        Returns:
            (特征DataFrame, 标签Series)
        """
        # 选择特征列
        X = df[self.feature_columns].copy()

        # 标签列
        y = df["label"].copy()

        # 处理NaN
        if drop_na:
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]

        return X, y

    def get_multi_timeframe_features(
        self,
        h4_df: pd.DataFrame,
        h1_df: Optional[pd.DataFrame] = None,
        d1_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """获取多周期特征

        Args:
            h4_df: H4周期数据
            h1_df: H1周期数据
            d1_df: D1周期数据

        Returns:
            合并后的特征DataFrame
        """
        # 计算H4指标
        if "ma5" not in h4_df.columns:
            h4_df = calculate_indicators(h4_df)

        result = h4_df.copy()

        # 添加H1特征
        if h1_df is not None and len(h1_df) > 0:
            if "ma5" not in h1_df.columns:
                h1_df = calculate_indicators(h1_df)

            # 同步时间（取最近H1数据）
            result["h1_ma5"] = result["time"].map(
                lambda t: h1_df[h1_df["time"] <= t]["ma5"].iloc[-1]
                if len(h1_df[h1_df["time"] <= t]) > 0
                else np.nan
            )
            result["h1_ma20"] = result["time"].map(
                lambda t: h1_df[h1_df["time"] <= t]["ma20"].iloc[-1]
                if len(h1_df[h1_df["time"] <= t]) > 0
                else np.nan
            )
            result["h1_ma_trend"] = (result["h1_ma5"] > result["h1_ma20"]).astype(int)

        # 添加D1特征
        if d1_df is not None and len(d1_df) > 0:
            if "ma5" not in d1_df.columns:
                d1_df = calculate_indicators(d1_df)

            result["d1_rsi_14"] = result["time"].map(
                lambda t: d1_df[d1_df["time"] <= t]["rsi_14"].iloc[-1]
                if len(d1_df[d1_df["time"] <= t]) > 0
                else np.nan
            )
            result["d1_atr_ratio"] = result["time"].map(
                lambda t: d1_df[d1_df["time"] <= t]["atr_ratio"].iloc[-1]
                if len(d1_df[d1_df["time"] <= t]) > 0
                else np.nan
            )

        # 更新特征列
        if "h1_ma_trend" in result.columns:
            self.feature_columns.append("h1_ma_trend")
        if "d1_rsi_14" in result.columns:
            self.feature_columns.append("d1_rsi_14")
        if "d1_atr_ratio" in result.columns:
            self.feature_columns.append("d1_atr_ratio")

        return result
