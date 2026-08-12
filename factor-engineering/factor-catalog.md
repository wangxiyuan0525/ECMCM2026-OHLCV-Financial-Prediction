# 整理版本 分类

## 一、 趋势追踪与平滑因子 (Trend Following & Smoothing)
这一维度的因子旨在过滤掉市场的高频噪音，识别出价格运动的中长期主干方向。深度学习模型非常需要这类特征来明确宏观背景。

### 1. KAMA (考夫曼自适应移动平均线, Kaufman's Adaptive Moving Average)
**计算与逻辑：** 结合了效率比率（ER）的智能均线。在单边趋势中紧跟价格，在震荡市中走平以减少错误信号。  
**公式：**
$$KAMA_t = KAMA_{t-1} + SC_t \times (Close_t - KAMA_{t-1})$$

### 2. HMA (赫尔移动平均线, Hull Moving Average)
**计算与逻辑：** 通过加权移动平均的差值重构，在保持曲线平滑的同时，极大地消除了传统均线的滞后性。  
**公式：**
$$HMA_n = WMA\left(2 \times WMA\left(Close, \frac{n}{2}\right) - WMA(Close, n), \sqrt{n}\right)$$

### 3. TRIX (三重指数平滑平均线, Triple Exponential Average)
**计算与逻辑：** 对收盘价连续进行三次 EMA 平滑，极大地滞后了价格，用于清除高频噪音并捕捉最坚固的趋势。  
**公式：**
$$TRIX = \text{3rd EMA of } (\text{EMA of } (\text{EMA of } Close))$$

### 4. Ichimoku Cloud (一目均衡表指标)
**计算与逻辑：** 通过多周期的最高价和最低价均值来构建支撑/阻力云带。需相对于收盘价进行归一化。  
**公式（以转换线为例）：**
$$Conversion = \frac{High_9 + Low_9}{2}$$

### 5. Supertrend Direction (超级趋势方向)
**计算与逻辑：** 结合 ATR 计算的动态止损/反转线，通常作为二进制特征（**1** 为上升趋势，**0** 为下降趋势）输入模型。  

---

## 二、 动量与超买超卖因子 (Momentum & Oscillators)
此类因子刻画价格变化的速度和多空力量的对比，用于寻找均值回归的节点（超买/超卖）或动量的爆发点。

### 1. RSI 家族 (相对强弱指数及其衍生, RSI / StochRSI / RSI Slope)
**计算与逻辑：** 比较近期涨跌的平均幅度。基础 RSI 反映超买超卖；StochRSI 对其再次标准化以捕捉高频极端波动；RSI Slope (当前 RSI - 昨天 RSI) 捕捉动量变化率。  
**公式：**
$$RSI = 100 - \frac{100}{1 + \frac{EMA(Gain)}{EMA(Loss)}}$$

### 2. MACD (平滑异同移动平均线)
**计算与逻辑：** 最经典的趋势动量指标，通过快慢 EMA 的差值捕捉趋势的加速或减速。  
**公式：**
$$MACD = EMA_{12}(Close) - EMA_{26}(Close)$$

### 3. CCI (商品通道指数, Commodity Channel Index)
**计算与逻辑：** 衡量典型价格偏离其简单移动平均线的程度，专用于寻找异常极值。  
**公式：**
$$CCI = \frac{Typical\ Price - SMA(Typical\ Price)}{0.015 \times Mean\ Deviation}$$

### 4. Williams %R (威廉指标)
**计算与逻辑：** 衡量当前收盘价在过去 n 个周期高低点区间内的相对位置，值越接近 **0** 越超买。  
**公式：**
$$\%R = \frac{High_{max,n} - Close_t}{High_{max,n} - Low_{min,n}} \times (-100)$$

### 5. CMO (钱德动量振荡器, Chande Momentum Oscillator)
**计算与逻辑：** 不经过 EMA 平滑的 RSI 变体，直接对比上涨和下跌的绝对动能。  
**公式：**
$$CMO = \frac{\sum U - \sum D}{\sum U + \sum D} \times 100$$

### 6. APO & PPO (绝对/百分比价格振荡器)
**计算与逻辑：** 衡量两条 EMA 之间的绝对差值（APO）或百分比差值（PPO，适合跨资产比较）。  
**公式：**
$$PPO = \frac{EMA_{fast} - EMA_{slow}}{EMA_{slow}} \times 100$$

---

## 三、 趋势绝对强度因子 (Trend Strength)
不判断价格是涨还是跌，只评估当前市场是否有强烈的单边共识（无论向上还是向下）。这是策略切换的“开关”。

### 1. ADX & ADXR (平均趋向指数及评级)
**计算与逻辑：** 基于最高价和最低价的突破差值（+DI, -DI）构建，平滑后反映单边行情的猛烈程度。ADXR 则是进一步平滑的稳健版本。  
**公式（以 DX 为例，ADX 为 DX 的平滑均值）：**
$$DX = \frac{|+DI - -DI|}{+DI + -DI} \times 100$$

### 2. AROON 振荡器 (Aroon Oscillator)
**计算与逻辑：** 测量周期内自最高价和最低价出现以来流逝的时间。距离极值越近（时间越短），说明该方向力量越强。  
**公式：**
$$Aroon\ Oscillator = AroonUp - AroonDown$$

---

## 四、 波动率与极值边界因子 (Volatility & Boundaries)
刻画资产的风险暴露水平和价格跳跃极限，对处理加密货币和股票的极端行情（尾部风险）至关重要。

### 1. ATR & Normalized ATR (真实波动幅度及归一化)
**计算与逻辑：** 考虑了跳空缺口的波动极限。为了消除绝对价格影响，通常将 ATR 除以收盘价得到 Normalized ATR。  
**公式：**
$$TR = \max(High - Low, |High - Close_{prev}|, |Low - Close_{prev}|)$$

### 2. Bollinger Bands 衍生 (布林带宽度与 %B)
**计算与逻辑：** Bandwidth 衡量波动的剧烈程度（收窄预示突破）；%B 刻画价格在上下轨之间的相对位置。  
**公式：**
$$Bandwidth = \frac{UpperBand - LowerBand}{MiddleBand}$$

---

## 五、 量价共振因子 (Volume-Price Dynamics)
引入真实的交易量数据，判断价格走势是否得到了真金白银的支持，是甄别“虚假突破”的核心。

### 1. VWAP 衍生 (成交量加权均价溢价/折价)
**计算与逻辑：** 计算当前收盘价相对于日内平均交易成本的偏离度。  
**公式：**
$$VWAP\ Premium = \frac{Close_t - VWAP_t}{VWAP_t}$$

### 2. MFI (资金流量指数, Money Flow Index)
**计算与逻辑：** 带有交易量权重的 RSI。将量价齐升记为正向资金流，量价齐跌记为负向资金流。  
**公式：**
$$MFI = 100 - \frac{100}{1 + \frac{Positive\ Money\ Flow}{Negative\ Money\ Flow}}$$

### 3. OBV (能量潮指标, On-Balance Volume)
**计算与逻辑：** 累积的量价指标，上涨日加成交量，下跌日减成交量，用于判断量价背离。  

### 4. CMF (蔡金资金流量指标, Chaikin Money Flow)
**计算与逻辑：** 考量了收盘价在单根 K 线最高低点之间的位置，结合成交量判断买卖盘压力的持续性。  
**公式：**
$$CMF_V = Volume \times \frac{(Close - Low) - (High - Close)}{High - Low}$$

---

## 六、 基础价格转化与微观结构因子 (Price Transformation & Microstructure)
将原始绝对价格转化为模型更容易消化的相对变化率，以及单根 K 线内部的博弈状态。

### 1. BOP (力量平衡, Balance of Power)
**计算与逻辑：** 刻画单根 K 线的实体占整体振幅的比例，值越接近 **1** 说明买方以绝对优势收盘。  
**公式：**
$$BOP = \frac{Close - Open}{High - Low}$$

### 2. MOM / ROC / Lagged Log Returns (动量/变动率/滞后对数收益率)
**计算与逻辑：** 从绝对动量（MOM）、百分比动量（ROC）到具备统计分布优势的对数收益率，都是衡量周期性回报的基础特征。  
**公式（对数收益率）：**
$$LogReturn_n = \ln\left(\frac{Close_t}{Close_{t-n}}\right)$$

### 3. EMA Returns (均线乖离率)
**计算与逻辑：** 衡量当前价格相较于其均线位置的溢价或折价率，即价格的乖离程度。  
**公式：**
$$EMA\_Return = \frac{Close - EMA_n(Close)}{EMA_n(Close)}$$