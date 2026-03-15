# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是一个 Python 量化分析项目 (quantification_fit)。

## Common Commands

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 运行项目
python -m quantification_fit

# 运行测试
pytest

# 运行单个测试文件
pytest tests/test_file.py

# 代码格式化
black .

# 代码检查
flake8 .
pylint .
mypy .
```

## Project Structure

推荐目录结构：
```
quantification_fit/
├── quantification_fit/    # 主包
│   ├── data/              # 数据模块
│   │   ├── database.py    # PostgreSQL操作
│   │   ├── fetcher.py     # MT5数据获取
│   │   └── indicators.py  # 技术指标计算
│   ├── strategy/          # 策略模块
│   │   └── rules.py       # 交易规则
│   ├── features/          # 特征模块
│   │   └── generator.py  # 特征生成与标注
│   ├── models/            # 模型模块
│   │   ├── trainer.py     # PyTorch模型训练
│   │   └── exporter.py    # ONNX导出
│   └── mql5/              # MQL5代码
│       └── model_loader.mq5  # ONNX模型加载器
├── scripts/               # 脚本
│   └── init_db.sql        # 数据库初始化SQL
├── tests/                 # 测试文件
├── output/                # 输出目录
├── main.py                # 主入口
├── requirements.txt       # 依赖
└── .env.example           # 环境变量示例
```

## Usage

```bash
# 1. 获取数据
python main.py fetch-data --symbol EURUSD --num-bars 1000

# 2. 生成标注
python main.py generate-labels --symbol EURUSD

# 3. 训练模型
python main.py train --epochs 100

# 4. 导出ONNX
python main.py export-onnx

# 5. 完整流程
python main.py all
```

## Database Setup

```bash
# 创建数据库
psql -U postgres -c "CREATE DATABASE forex_data;"

# 执行SQL脚本
psql -U postgres -d forex_data -f scripts/init_db.sql
```

## Development Notes

- 使用 `pytest` 作为测试框架
- 使用 `black` 进行代码格式化
- 使用 `flake8` 或 `pylint` 进行代码检查
- 类型注解使用 `mypy` 检查
