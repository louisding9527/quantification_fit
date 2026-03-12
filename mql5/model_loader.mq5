//+------------------------------------------------------------------+
//|                                                 model_loader.mq5 |
//|                        外汇交易神经网络ONNX模型加载器              |
//+------------------------------------------------------------------+
#property copyright "Quantification Project"
#property version   "1.00"
#property strict

#include <OnnxLoader.mqh>

//+------------------------------------------------------------------+
//| 模型参数配置                                                      |
//+------------------------------------------------------------------+
input string InpModelName = "trading_model.onnx";  // ONNX模型文件
input double InpConfidence = 0.6;                   // 置信度阈值

// 特征数量 (根据训练时的特征数量调整)
#define NUM_FEATURES 20

//+------------------------------------------------------------------+
//| 全局变量                                                          |
//+------------------------------------------------------------------+
COnnxLoader *onnx;
datetime lastBarTime = 0;
ENUM_TIMEFRAMES currentTimeframe = PERIOD_H4;

//+------------------------------------------------------------------+
//| 专家初始化函数                                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // 创建ONNX加载器
    onnx = new COnnxLoader();

    // 加载模型
    if(!onnx.LoadModel(InpModelName, ONNX_DEFAULT_EXECUTION_PROVIDER))
    {
        Print("Failed to load ONNX model: ", InpModelName);
        return INIT_FAILED;
    }

    // 添加输入张量 (batch_size, NUM_FEATURES)
    if(!onnx.AddInputTensor(ONNX_TYPE_FLOAT, {1, NUM_FEATURES}, "input"))
    {
        Print("Failed to add input tensor");
        return INIT_FAILED;
    }

    // 添加输出张量 (batch_size, 3) - BUY/HOLD/SELL
    if(!onnx.AddOutputTensor(ONNX_TYPE_FLOAT, {1, 3}, "output"))
    {
        Print("Failed to add output tensor");
        return INIT_FAILED;
    }

    Print("ONNX model loaded successfully");
    Print("Input shape: ", onnx.GetInputType(0));
    Print("Output shape: ", onnx.GetOutputType(0));

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| 专家反初始化函数                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(onnx != NULL)
    {
        delete onnx;
        onnx = NULL;
    }
}

//+------------------------------------------------------------------+
//| tick函数                                                         |
//+------------------------------------------------------------------+
void OnTick()
{
    // 只在新K线时计算
    if(lastBarTime == Time[0])
        return;

    lastBarTime = Time[0];

    // 获取特征
    float features[NUM_FEATURES];
    if(!CalculateFeatures(features))
    {
        return;
    }

    // 运行推理
    if(!RunInference(features))
    {
        return;
    }
}

//+------------------------------------------------------------------+
//| 计算特征函数                                                      |
//+------------------------------------------------------------------+
bool CalculateFeatures(float &features[])
{
    // 确保有足够的数据
    if(iRSI(NULL, currentTimeframe, 14, PRICE_CLOSE) < 1)
        return false;

    int idx = 0;

    // 趋势指标 - MA
    features[idx++] = iMA(NULL, currentTimeframe, 5, 0, MODE_SMA, PRICE_CLOSE, 1);
    features[idx++] = iMA(NULL, currentTimeframe, 10, 0, MODE_SMA, PRICE_CLOSE, 1);
    features[idx++] = iMA(NULL, currentTimeframe, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
    features[idx++] = iMA(NULL, currentTimeframe, 60, 0, MODE_SMA, PRICE_CLOSE, 1);

    // EMA
    features[idx++] = iMA(NULL, currentTimeframe, 12, 0, MODE_EMA, PRICE_CLOSE, 1);
    features[idx++] = iMA(NULL, currentTimeframe, 26, 0, MODE_EMA, PRICE_CLOSE, 1);

    // MACD
    double macdMain[], macdSignal[];
    ArraySetAsSeries(macdMain, true);
    ArraySetAsSeries(macdSignal, true);
    iMACD(NULL, currentTimeframe, 12, 26, 9, PRICE_CLOSE).GetData(0, 2, macdMain);
    iMACD(NULL, currentTimeframe, 12, 26, 9, PRICE_CLOSE).GetLine(0).GetData(0, 2, macdSignal);
    features[idx++] = (float)macdMain[0];
    features[idx++] = (float)macdSignal[0];
    features[idx++] = (float)(macdMain[0] - macdSignal[0]);

    // RSI
    features[idx++] = (float)iRSI(NULL, currentTimeframe, 6, PRICE_CLOSE, 1);
    features[idx++] = (float)iRSI(NULL, currentTimeframe, 12, PRICE_CLOSE, 1);
    features[idx++] = (float)iRSI(NULL, currentTimeframe, 14, PRICE_CLOSE, 1);
    features[idx++] = (float)iRSI(NULL, currentTimeframe, 24, PRICE_CLOSE, 1);

    // Stochastic
    double stochK[], stochD[];
    ArraySetAsSeries(stochK, true);
    ArraySetAsSeries(stochD, true);
    iStochastic(NULL, currentTimeframe, 14, 3, 3, MODE_SMA, STO_LOWHIGH).GetMain(stochK);
    iStochastic(NULL, currentTimeframe, 14, 3, 3, MODE_SMA, STO_LOWHIGH).GetSignal(stochD);
    features[idx++] = (float)stochK[0];
    features[idx++] = (float)stochD[0];
    features[idx++] = (float)(3 * stochK[0] - 2 * stochD[0]);

    // Bollinger Bands
    double bbUpper[], bbLower[];
    ArraySetAsSeries(bbUpper, true);
    ArraySetAsSeries(bbLower, true);
    iBands(NULL, currentTimeframe, 20, 2, 0, PRICE_CLOSE).GetUpper(bbUpper);
    iBands(NULL, currentTimeframe, 20, 2, 0, PRICE_CLOSE).GetLower(bbLower);
    float bbPosition = (float)((Close[1] - bbLower[0]) / (bbUpper[0] - bbLower[0] + 0.0001));
    features[idx++] = bbPosition;

    // ATR
    float atr = (float)iATR(NULL, currentTimeframe, 14, 1);
    features[idx++] = atr;
    features[idx++] = atr / Close[1] * 100;

    // ADX
    features[idx++] = (float)iADX(NULL, currentTimeframe, 14, 1);

    // 辅助指标
    float ma5 = iMA(NULL, currentTimeframe, 5, 0, MODE_SMA, PRICE_CLOSE, 1);
    float ma20 = iMA(NULL, currentTimeframe, 20, 0, MODE_SMA, PRICE_CLOSE, 1);
    features[idx++] = ma5 > ma20 ? 1.0f : 0.0f;

    float rsi14 = iRSI(NULL, currentTimeframe, 14, PRICE_CLOSE, 1);
    features[idx++] = rsi14 > 70 ? 1.0f : 0.0f;
    features[idx++] = rsi14 < 30 ? 1.0f : 0.0f;
    features[idx++] = macdMain[0] > macdSignal[0] ? 1.0f : 0.0f;
    features[idx++] = (float)((High[1] - Low[1]) / Close[1]);

    // 确保所有特征都已填充
    if(idx != NUM_FEATURES)
    {
        Print("Feature count mismatch: ", idx, " vs ", NUM_FEATURES);
        return false;
    }

    // 标准化 (需要与训练时相同的均值和标准差)
    // 这里需要用训练时的scaler参数替换
    float means[NUM_FEATURES] = {0};  // 替换为训练时的均值
    float stds[NUM_FEATURES] = {1};   // 替换为训练时的标准差

    for(int i = 0; i < NUM_FEATURES; i++)
    {
        features[i] = (features[i] - means[i]) / stds[i];
    }

    return true;
}

//+------------------------------------------------------------------+
//| 运行推理函数                                                      |
//+------------------------------------------------------------------+
bool RunInference(float &features[])
{
    // 设置输入
    if(!onnx.SetInputTensorData(0, features))
    {
        Print("Failed to set input data");
        return false;
    }

    // 运行推理
    if(!onnx.Run())
    {
        Print("Failed to run inference");
        return false;
    }

    // 获取输出
    float output[];
    if(!onnx.GetOutputTensorData(0, output))
    {
        Print("Failed to get output data");
        return false;
    }

    // 解析输出: output[0]=BUY概率, output[1]=HOLD概率, output[2]=SELL概率
    float buyProb = output[0];
    float holdProb = output[1];
    float sellProb = output[2];

    // 打印概率
    Print("Buy: ", buyProb, ", Hold: ", holdProb, ", Sell: ", sellProb);

    // 信号过滤
    if(buyProb > InpConfidence && buyProb > sellProb && buyProb > holdProb)
    {
        // 买入信号
        if(PositionGetSymbol(0) == "")
        {
            Print("BUY SIGNAL - Opening position");
            // Trade.Buy(1.0, _Symbol, 0, 0, 0, "NN BUY");
        }
    }
    else if(sellProb > InpConfidence && sellProb > buyProb && sellProb > holdProb)
    {
        // 卖出信号
        if(PositionGetSymbol(0) == _Symbol)
        {
            Print("SELL SIGNAL - Closing position");
            // Trade.PositionClose(_Symbol);
        }
    }

    return true;
}

//+------------------------------------------------------------------+
//| 训练数据生成器 (Python参考)                                        |
//+------------------------------------------------------------------+
/*
# 以下是Python端生成训练数据的参考代码

from quantification_fit.data.fetcher import fetch_multi_timeframe_data, TimeFrame
from quantification_fit.data.indicators import calculate_indicators
from quantification_fit.features.generator import FeatureGenerator
import pandas as pd

# 获取数据
data = fetch_multi_timeframe_data("EURUSD", [TimeFrame.H4, TimeFrame.H1, TimeFrame.D1], 5000)

# 计算指标
h4_df = calculate_indicators(data["4H"])

# 生成特征
gen = FeatureGenerator()
df = gen.generate_labels(h4_df, look_ahead=10)

# 准备训练数据
X, y = gen.prepare_training_data(df)

# 训练模型
from quantification_fit.models.trainer import train_model
trainer, history = train_model(X, y, epochs=100)

# 导出ONNX
from quantification_fit.models.exporter import export_to_onnx
export_to_onnx(trainer, "output", "trading_model")
*/
