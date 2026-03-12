"""外汇K线交易标注与神经网络训练 - 主入口"""

import argparse
import logging
import os
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="外汇K线交易标注与神经网络训练"
    )
    parser.add_argument(
        "command",
        choices=["fetch-data", "generate-labels", "train", "export-onnx", "all"],
        help="执行的命令",
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="交易品种 (默认: EURUSD)",
    )
    parser.add_argument(
        "--num-bars",
        type=int,
        default=1000,
        help="获取的K线数量 (默认: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数 (默认: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="输出目录 (默认: output)",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    if args.command == "fetch-data":
        fetch_data(args)
    elif args.command == "generate-labels":
        generate_labels(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "export-onnx":
        export_onnx(args)
    elif args.command == "all":
        run_all(args)


def fetch_data(args):
    """获取数据"""
    from quantification_fit.data.fetcher import fetch_multi_timeframe_data, TimeFrame
    from quantification_fit.data.database import Database
    from quantification_fit.data.indicators import calculate_indicators

    logger.info(f"Fetching {args.symbol} data...")

    # 获取多周期数据
    timeframes = [TimeFrame.H4, TimeFrame.H1, TimeFrame.D1]
    data = fetch_multi_timeframe_data(args.symbol, timeframes, args.num_bars)

    # 计算技术指标
    for tf, df in data.items():
        logger.info(f"{tf}: {len(df)} bars")
        data[tf] = calculate_indicators(df)

    # 尝试保存到数据库（可选）
    try:
        db = Database()
        db.connect()
        for tf, df in data.items():
            records = df.to_dict("records")
            db.insert_ohlc(args.symbol, tf, records)
        db.close()
        logger.info("Data saved to database")
    except Exception as e:
        logger.warning(f"Database not available: {e}")

    # 保存到文件
    for tf, df in data.items():
        df.to_csv(f"{args.output_dir}/{args.symbol}_{tf}.csv", index=False)
        logger.info(f"Saved {args.symbol}_{tf}.csv")

    return data


def generate_labels(args):
    """生成标注"""
    from quantification_fit.data.fetcher import fetch_multi_timeframe_data, TimeFrame
    from quantification_fit.data.indicators import calculate_indicators
    from quantification_fit.features.generator import LabelGenerator, FeatureGenerator

    logger.info(f"Generating labels for {args.symbol}...")

    # 加载数据
    h4_df = None
    h1_df = None
    d1_df = None

    try:
        h4_df = pd.read_csv(f"{args.output_dir}/{args.symbol}_H4.csv")
        h4_df["time"] = pd.to_datetime(h4_df["time"])
    except Exception as e:
        logger.warning(f"Could not load H4 data: {e}")

    if h4_df is None:
        logger.info("Fetching data from MT5...")
        timeframes = [TimeFrame.H4, TimeFrame.H1, TimeFrame.D1]
        data = fetch_multi_timeframe_data(args.symbol, timeframes, args.num_bars)
        h4_df = data.get("H4")
        h1_df = data.get("H1")
        d1_df = data.get("D1")

    if h4_df is None or len(h4_df) == 0:
        logger.error("No data available")
        return

    # 生成交易记录
    generator = LabelGenerator(symbol=args.symbol)
    trades = generator.generate_trades(h4_df, h1_df, d1_df)

    logger.info(f"Generated {len(trades)} trades")

    # 转换为DataFrame并保存
    trades_df = generator.trades_to_dataframe(trades)
    trades_df.to_csv(f"{args.output_dir}/trades.csv", index=False)
    logger.info(f"Saved trades.csv")

    # 生成特征标注（用于模型训练）
    feature_gen = FeatureGenerator()
    h4_with_features = feature_gen.generate_labels(h4_df)
    X, y = feature_gen.prepare_training_data(h4_with_features)

    logger.info(f"Training data shape: X={X.shape}, y={len(y)}")

    # 保存
    X.to_csv(f"{args.output_dir}/features.csv", index=False)
    y.to_csv(f"{args.output_dir}/labels.csv", index=False)
    logger.info("Saved features.csv and labels.csv")


def train_model(args):
    """训练模型"""
    import pandas as pd
    from quantification_fit.models.trainer import train_model

    logger.info("Loading training data...")

    try:
        X = pd.read_csv(f"{args.output_dir}/features.csv")
        y = pd.read_csv(f"{args.output_dir}/labels.csv")["label"]
    except Exception as e:
        logger.error(f"Could not load training data: {e}")
        logger.info("Run 'generate-labels' first")
        return

    logger.info(f"Training data: {X.shape}")

    # 训练
    trainer, history = train_model(
        X, y,
        hidden_dims=[64, 32],
        epochs=args.epochs,
    )

    # 保存模型
    model_path = f"{args.output_dir}/model.pth"
    trainer.save(model_path)

    # 打印结果
    logger.info(f"Training completed. Best val accuracy: {max(history['val_acc']):.4f}")


def export_onnx(args):
    """导出ONNX模型"""
    from quantification_fit.models.trainer import Trainer
    from quantification_fit.models.exporter import export_to_onnx

    logger.info("Loading trained model...")

    try:
        trainer = Trainer(input_dim=20)  # 会从保存的模型加载
        trainer.load(f"{args.output_dir}/model.pth")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        logger.info("Run 'train' first")
        return

    # 导出ONNX
    onnx_path = export_to_onnx(trainer, args.output_dir, "trading_model")
    logger.info(f"ONNX model exported to {onnx_path}")


def run_all(args):
    """运行完整流程"""
    logger.info("Running complete pipeline...")

    # 1. 获取数据
    fetch_data(args)

    # 2. 生成标注
    generate_labels(args)

    # 3. 训练模型
    train_model(args)

    # 4. 导出ONNX
    export_onnx(args)

    logger.info("Pipeline completed!")


if __name__ == "__main__":
    import pandas as pd

    main()
