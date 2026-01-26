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

![bg blur:5px brightness:0.5](clouds.jpg)

# Machine Learning for Numeric Weather Prediction

**Group Members:**
Yuqi Fang, Ali Cem Çakmak, Muhammad Fakhar, Diego Garces, Deepak Sorout

---

# Table of Contents

1. Introduction
2. Methods Overview
3. Linear Regression
4. Random Forest
5. Neural Network
6. Model Comparison
7. Conclusion
8. Questions

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

<!-- header: '2. Methods Overview' -->
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

<!-- header: '3. Linear Regression' -->
<!-- _class: title -->

# Linear Regression

---

<!-- header: '3. Linear Regression' -->

# [Linear Regression Content]

*This section will be filled by Diego and Muhammad*

---

<!-- header: '4. Random Forest' -->
<!-- _class: title -->

# Random Forest

---

<!-- header: '4. Random Forest' -->

# [Random Forest Content]

*This section will be filled by Yuqi and Deepak*

---

<!-- header: '5. Neural Network' -->
<!-- _class: title -->

# Neural Network

---

<!-- header: '5. Neural Network' -->

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

![w:950](../../Neural-Network/figures/temperature_forecast_evaluation.png)

</div>

---

<!-- header: '6. Model Comparison' -->
<!-- _class: title -->

# Model Comparison

---

<!-- header: '6. Model Comparison' -->

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

![w:1000](../../figures/metrics_bar_chart.png)

</div>

---

<div style="text-align: center;">

![w:850](../../figures/scatter_plots_comparison.png)

</div>

---

# All Models: Time Series (Full Range)

<div style="text-align: center;">

![w:1050](../../figures/time_series_comparison.png)

</div>

---

# All Models: Time Series (2025 Detail)

<div style="text-align: center;">

![w:1050](../../figures/time_series_2025.png)

</div>

---

<!-- header: '7. Conclusion' -->

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

<!-- header: '8. Questions?' -->
<!-- _class: title -->

# Questions?
