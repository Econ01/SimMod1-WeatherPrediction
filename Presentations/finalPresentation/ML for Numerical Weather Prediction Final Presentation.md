---
marp: true
theme: default
paginate: true
footer: 'Machine Learning for Numeric Weather Prediction'
transition: fade
style: |
  @import url('https://fonts.cdnfonts.com/css/sf-pro-display');

  section {
    font-family: 'SF Pro Rounded', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  section.title {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  section.title h1 {
    font-size: 2.5em;
    margin-bottom: 0.5em;
  }

  section.title p {
    font-size: 1.1em;
    color: #ccc;
  }

  header {
    font-weight: bold;
    color: #333;
  }

  footer {
    font-size: 0.7em;
    color: #666;
  }

  div.twocols {
    display: flex;
    gap: 2em;
  }

  div.twocols .col {
    flex: 1;
  }

  div.twocols h2:first-child {
    margin-top: 0 !important;
  }

  div.twocols .col-center {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  div.twocols .col-center h1 {
    text-align: left;
  }

  .img-full-height {
    position: absolute;
    top: 0;
    right: 0;
    width: 65%;
    height: 100%;
    overflow: hidden;
  }

  .img-full-height img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    filter: blur(1px);
  }

  .img-full-height::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 50%;
    height: 100%;
    background: linear-gradient(to right, white, transparent);
    z-index: 1;
  }

  table {
    border-radius: 50px;
    overflow: hidden;
    border-collapse: separate;
    border-spacing: 0;
  }

  table td, table th {
    background: #00000007;
    padding: 0.6rem;
    font-size: 0.9rem;
  }

  table tr:nth-child(even) {
    background: transparent;
  }
math: mathjax
---

<!-- _class: title -->
<!-- _paginate: false -->
<!-- _footer: '' -->
<!-- _backgroundColor: #000 -->
<!-- _color: #fff -->

![bg blur:5px brightness:0.5](images/clouds.jpg)

# Machine Learning for Numeric Weather Prediction

**Group Members:**
Yuqi Fang, Ali Cem Çakmak, Muhammad Fakhar, Diego Garces, Deepak Sorout

---

# Table of Contents

1. Introduction
2. Data Management
3. Methods Overview
4. Linear Regression
5. Random Forest
6. Neural Network
7. Model Comparison
8. Conclusion
9. Questions

---

<!-- header: '1. Introduction' -->

# Problem & Objective

**Goal:** Predict future temperature using machine learning

<div class="twocols">
<div class="col">

**Approach:**
- Compare 3 machine learning methods
- Predict mean daily temperature
- Evaluate on real-world data (2023-2025)

</div>
<div class="col">

**Data Source:**
- European Climate Assessment & Dataset (ECA&D)
- Station: Köln-Bonn, Germany
- Period: 1957-2025 (68 years)
- Variables: 10 weather features

</div>
</div>

---

<!-- header: '2. Data Management' -->
<!-- _class: title -->

# Data Management

---

<!-- header: '3. Data Management' -->

# Data Cleaning

- Removed measurements beyond 2025-09-30
- Forward-filled missing values
- Excluded variables with poor data quality (Sea Level Pressure, Global Radiation)


---

# Data Splitting

<div class="twocols">
<div class="col">

**Year-based split (no data leakage):**
- **Train:** 1957-2017 (60 years)
- **Validation:** 2018-2022 (5 years)
- **Test:** 2023-2025 (current data)

</div>
<div class="col">

**Why year-based splitting?**
- Preserves temporal ordering
- Tests generalization to future, unseen data
- Models cannot "peek" into the future during training

</div>
</div>

---

<!-- header: '3. Methods Overview' -->
<!-- _class: title -->

# Methods Overview

---

<!-- _header: '' -->

<style scoped>
.contribution-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 2em;
  margin: 2em 0;
}

.contribution-box {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50px;
  padding: 1.5em;
  color: white;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  text-align: center;
  display: flex;
  flex-direction: column;
}

.contribution-box h3 {
  margin: 0 0 0.8em 0;
  font-size: 1.3em;
  border-bottom: 2px solid rgba(255, 255, 255, 0.3);
  padding-bottom: 0.5em;
}

.contribution-box p {
  margin: 0.3em 0;
  font-size: 0.95em;
  opacity: 0.95;
}

.box-lr { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.box-rf { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
.box-nn { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
</style>

<div class="contribution-grid">
  <div class="contribution-box box-lr">
    <h3>Linear Regression</h3>
    <p>Diego Garces</p>
    <p>Muhammad Fakhar</p>
  </div>

  <div class="contribution-box box-rf">
    <h3>Random Forest</h3>
    <p>Yuqi Fang</p>
    <p>Deepak Sorout</p>
  </div>

  <div class="contribution-box box-nn">
    <h3>Neural Network</h3>
    <p>Ali Cem Çakmak</p>
  </div>
</div>

---

<!-- header: '4. Linear Regression' -->
<!-- _class: title -->

# Linear Regression

---

<div style="text-align: center;">

# Three Approaches

**Simple Linear Regression:**
$$TG_t = \beta_0 + \beta_1 TG_{t-1} + \epsilon$$

**Multiple Linear Regression:**
$$TG_t = \beta_0 + \beta_1 TN_{t-1} + \beta_2 TX_{t-1} + \beta_3 TG_{t-1} + \epsilon$$

**Rolling Window Linear Regression:**
$$TG_t = \beta_0 + \beta_1 TN_{t-1} + \beta_2 TX_{t-1} + \beta_3 TG_{t-1} + \beta_4 TG_{3d} + \epsilon$$

</div>

---

<style scoped>
.lr-container { display: flex; gap: 1em; align-items: center; }
.lr-left { flex: 0.6; display: flex; align-items: center; justify-content: center; }
.lr-right { flex: 0.4; display: flex; align-items: center; }
.lr-right table { margin: 0; width: 100%; font-size: 0.9em; }
.lr-right td, .lr-right th { padding: 0.4rem 0.5rem; }
</style>

# Results

<div class="lr-container">
<div class="lr-left">

![](images/tg_multi_real_vs_predicho.png)

</div>
<div class="lr-right">

| Experiment | MAE | RMSE | R² |
|------------|-----|------|-----|
| Simple | 1.76°C | 2.26°C | 0.880 |
| Multiple | 1.72°C | 2.22°C | 0.885 |
| Rolling Window | 1.72°C | 2.22°C | 0.884 |

</div>
</div>

**Key Observations:**
- Moving from simple to multiple regression improved all metrics
- More information → provides context about the stability of the weather
- Adding the 3-day average adds more noise than signal

---

<div style="text-align: center;">

# Fitted Models

**Simple Linear Regression:**
$$TG_t = 0.62 + 0.93 TG_{t-1} + \epsilon$$

**Multiple Linear Regression:**
$$TG_t = 0.07 - 0.12 TN_{t-1} + 0.07 TX_{t-1} + 0.96 TG_{t-1} + \epsilon$$

**Rolling Window Linear Regression:**
$$TG_t = 0.02 + 0.15 TN_{t-1} + 0.035 TX_{t-1} + 0.94 TG_{t-1} + 0.093 TG_{3d} + \epsilon$$

*The coefficient for yesterday's mean temperature is the strongest baseline predictor.*

</div>

---

<!-- header: '5. Random Forest' -->
<!-- _class: title -->

# Random Forest

---

<style scoped>
section { font-size: 2em; }
.rf-container { display: flex; gap: 1em; align-items: center; }
.rf-left { flex: 1; }
.rf-left p, .rf-left ul { margin: 0em 0; }
.rf-left ul { padding-left: 1.2em; }
.rf-right { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; }
</style>

<div class="rf-container">
<div class="rf-left">

*"The Wisdom of the Crowds"*

**Process:**
- **Many Trees:** 100 independent decision trees
- **Randomness:** Each tree sees random subset of data & features
- **Averaging:** Final Prediction = Average of all 100 trees

**Why:**
- Reduces overfitting
- Lowers prediction variance
- Good for non-linear relationships

</div>
<div class="rf-right">

![w:480](images/randomforest.webp)

<p style="font-size: 0.6em;">Source: GeeksforGeeks</p>

</div>
</div>

---

<style scoped>
section { font-size: 1.5em; }
h1 { margin-bottom: 0.2em; }
.meth-container { display: flex; gap: 0.8em; }
.meth-left { flex: 0.7; }
.meth-left p, .meth-left ul { margin: 0.2em 0; }
.meth-left ul { padding-left: 1.2em; }
.meth-right { flex: 1.3; font-size: 0.8em; }
.meth-right table { margin: 0; }
.meth-right td, .meth-right th { padding: 0.3rem 0.6rem; }
</style>

# Methodology

<div class="meth-container">
<div class="meth-left">

**Autoregression (temporal dependencies)**
- Past **15 days** of data → predict today (TG)

**Features:**
- 10 Weather variables: TG, TN, TX, RR, SS, HU, FG, FX, CC, SD
- Lag days: **15**
- Total features = 10 × 15 = **150 input columns**

**Model Training:**
- Lagged features fed into **Random Forest Regressor**
- Model learns patterns from historical data

</div>
<div class="meth-right">

| DATE | TG_1 | TG_2 | TG_3 | TG_4 | ... | TG_15 | Target |
|------|------|------|------|------|-----|-------|--------|
| 10-16 | 108 | 102 | 97 | 106 | ... | 56 | 139 |
| 10-17 | 139 | 108 | 102 | 97 | ... | 57 | 139 |
| 10-18 | 139 | 139 | 108 | 102 | ... | 83 | 174 |
| 10-19 | 174 | 139 | 139 | 108 | ... | 58 | 105 |
| 10-20 | 105 | 174 | 139 | 139 | ... | 93 | 77 |
| 10-21 | 77 | 105 | 174 | 139 | ... | 108 | 100 |
| 10-22 | 100 | 77 | 105 | 174 | ... | 106 | 77 |
| 10-23 | 77 | 100 | 77 | 105 | ... | 105 | 84 |

</div>
</div>

---

# Validation & Hyperparameter Tuning

<div class="twocols">
<div class="col">

<span style="color: green;">Train → Test</span> to <span style="color: green;">Train → Validation → Test</span>

|<span style="color: green;">Trees \ Depth</span> | <span style="color: green;">None</span> | <span style="color: green;">10</span> |
|---------------|------|------|
| 50 | 2.0882 | 2.0771 |
| 100 | 2.0740 | **2.0737** |
| 200 | 2.0768 | 2.0777 |

</div>
<div class="col">

🏆 **Best Model:**

**Trees: 100 | Depth: 10** *(RMSE: 2.0737)*

More trees ≠ Better results!

</div>
</div>

---

# Feature Selection Experiment

<div class="twocols">
<div class="col">

<span style="color: green;">**The Hypothesis:**</span>

**Reduce Dimensionality**
- **Action:** Selected Top 30 Features
- **Expectation:** Remove noise → Better Accuracy

</div>
<div class="col">

<span style="color: green;">**What we observed:**</span>

**Performance Drop**
- 🏆 Full Model (150 Feats): → lower MAE, higher R²
- ❌ Reduced Model (30 Feats): → MAE increased, R² decreased

</div>
</div>

*Take-away: Feature selection likely disrupted temporal continuity (breaking the trend) and lost valuable interaction effects (where features contribute jointly)*

---

<div style="text-align: center;">

**MAE: 1.59°C &emsp; RMSE: 2.07°C &emsp; R²: 0.901**

![w:1200](images/randomforest_results.jpeg)

</div>

---

<!-- header: '6. Neural Network' -->
<!-- _class: title -->

# Neural Network

---

# Approach

<div class="twocols">
<div class="col">

**Architecture:**
- Sequence-to-Sequence GRU (Gated Recurrent Unit) with Attention
- Encoder-Decoder structure
- Hidden dimension: 64 units
- Single layer

</div>
<div class="col">

**Task:**
- Input: 15 days of weather history
- Output: 3-day temperature forecast
- Autoregressive prediction
- Benchmarked against baselines

</div>
</div>

---

# Benchmark Models

<div class="twocols">
<div class="col">

## Persistent Model

**Approach:** Tomorrow = Today

$$\hat{y}_{t+1} = y_t$$

- Simplest baseline
- No training required
- Assumes weather stays constant

</div>
<div class="col">

## SARIMA Model

**Approach:** Statistical time series model

- Captures seasonal patterns
- Traditional forecasting method
- Widely used in meteorology

</div>
</div>

---

# Results: Day 1 Forecast

<div class="twocols">
<div class="col">

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| **GRU** | **1.61** | **2.06** | **0.901** |
| Persistent | 1.78 | 2.30 | 0.877 |
| SARIMA | 1.72 | 2.22 | 0.885 |

</div>
<div class="col">

**Key Results:**
- 9.6% improvement over Persistent
- 6.4% improvement over SARIMA
- Explains 90% of variance
- Best performance on all metrics

</div>
</div>

---

# Results: Day 3 Forecast

<div class="twocols">
<div class="col">

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| **GRU** | **2.58** | **3.26** | **0.753** |
| Persistent | 3.04 | 3.86 | 0.652 |
| SARIMA | 2.73 | 3.47 | 0.719 |

</div>
<div class="col">

**Key Results:**
- 15.1% improvement over Persistent
- 5.5% improvement over SARIMA
- Performance degrades gracefully
- Maintains advantage at longer horizon

</div>
</div>

---

# Execution Time

<div class="twocols">
<div class="col">

| Model | Time | vs SARIMA |
|-------|------|-----------|
| **GRU (to best)** | **227.8s** | **42.7% faster** |
| Persistent | 10.2ms | - |
| SARIMA | 397.9s | baseline |

*GRU time includes full training*

</div>
<div class="col">

**Key Observations:**
- GRU trains in ~3.8 minutes
- Nearly twice as fast as SARIMA
- Achieves better accuracy in less time
- Persistent is instantaneous (no training)
- GRU offers best accuracy-to-time ratio

</div>
</div>

---

<div style="text-align: center;">

![w:950](images/temperature_forecast_evaluation.png)

</div>

---

<!-- header: '7. Model Comparison' -->
<!-- _class: title -->

# Model Comparison

---

# Performance Summary

<div class="twocols">
<div class="col">

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| Random Forest | **1.59** | 2.07 | 0.901 |
| **GRU** | 1.61 | **2.06** | **0.902** |
| Linear Regression | 1.73 | 2.23 | 0.885 |
| SARIMA | 1.73 | 2.23 | 0.885 |
| Persistent | 1.78 | 2.30 | 0.878 |

</div>
<div class="col">

**Key Observations:**
- GRU and RF achieve comparable top performance
- Both ML models outperform traditional methods
- 10% improvement over Persistent baseline
- 7% improvement over SARIMA baseline
- All models maintain R² > 0.87

</div>
</div>

---

<div style="text-align: center;">

![w:1000](images/metrics_bar_chart.png)

</div>

---

<div style="text-align: center;">

![w:850](images/scatter_plots_comparison.png)

</div>

---

# All Models: Time Series (Full Range)

<div style="text-align: center;">

![w:1050](images/time_series_comparison.png)

</div>

---

# All Models: Time Series (2025 Detail)

<div style="text-align: center;">

![w:1050](images/time_series_2025.png)

</div>

---

<!-- header: '8. Conclusion' -->

# Conclusion

<div class="twocols">
<div class="col">

**Key Findings:**
- GRU and Random Forest achieved top performance (R² > 0.90)
- All ML methods outperformed presistent by 3-12%
- GRU: Better accuracy-to-time ratio (43% faster than SARIMA)

</div>
<div class="col">

**Future Directions:**
- Multi-station data for spatial patterns
- Extend forecast horizon beyond 3 days
- Ensemble methods combining approaches

</div>
</div>

---

<!-- header: '9. Questions?' -->
<!-- _class: title -->

# Questions?
