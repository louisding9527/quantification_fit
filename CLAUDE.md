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
│   ├── __init__.py
│   └── ...
├── tests/                  # 测试文件
├── requirements.txt        # 依赖
├── setup.py               # 或 pyproject.toml
└── CLAUDE.md
```

## Development Notes

- 使用 `pytest` 作为测试框架
- 使用 `black` 进行代码格式化
- 使用 `flake8` 或 `pylint` 进行代码检查
- 类型注解使用 `mypy` 检查
