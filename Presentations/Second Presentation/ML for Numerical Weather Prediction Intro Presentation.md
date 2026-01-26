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

1. Group Members
2. Problem Description
3. Data Management
4. Linear Regression
5. Random Forest
6. Neural Network
7. Questions

---

<!-- header: '1. Group Members' -->
<!-- _class: title -->

# Group Members

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

<!-- header: '2. Problem Description' -->
<!-- _class: title -->

# Problem Description

---

<!-- header: '2. Problem Description' -->

**Objective:** Predict future weather conditions using machine learning methods

**Challenge:** Single station limitation
- Data from only one location
- Unable to detect approaching weather fronts from surrounding regions
- Impacts accuracy for multi-day forecasts

---

# Data Source

**European Climate Assessment & Dataset (ECA&D)**

- **Station:** Köln-Bonn, Germany
- **Time Period:** 1957-2025 (68 years)
- **Temporal Resolution:** Daily measurements
- **Total Data Points:** Over 24,000 days of observations
- **Variables:** 12 weather variables measured daily

---

<!-- header: '3. Data Management' -->
<!-- _class: title -->

# Data Management

---

<!-- header: '3. Data Management' -->

<div class="twocols">
<div class="col">

## Data Cleaning
- Removed measurements beyond 2025-09-30
- Forward-filled missing values
- Excluded variables with poor data quality (Sea Level Pressure, Global Radiation)

</div>
<div class="col">

## Features Used (9 variables)
- Min/Max/Mean Temperature
- Precipitation & Snow Depth
- Sunshine Duration & Cloud Cover
- Humidity
- Wind Speed & Wind Gust

</div>
</div>

---

# Data Splitting

<div class="twocols">
<div class="col">

**Year-based split (no data leakage):**
- **Train:** 1957-2017 (60 years)
- **Validation:** 2018-2022 (5 years)
- **Test:** 2023-2025 (current data)

**Sample counts vary by method:**
- Linear Regression & Random Forest use daily samples
- Neural Network uses overlapping sequences

</div>
<div class="col">

**Why year-based splitting?**
- Preserves temporal ordering
- Tests generalization to future, unseen data
- Prevents look-ahead bias
- Models cannot "peek" into the future during training

</div>
</div>

---

<!-- header: '6. Neural Network' -->
<!-- _class: title -->

# Neural Network

---

# Sequence Creation

<div class="twocols">
<div class="col">

**Sliding window approach:**
- **Input (X):** 15 consecutive days × 9 features
- **Output (y):** 3 consecutive days of temperature
- **Sliding step:** 1 day (overlapping sequences)

</div>
<div class="col">

**Example:**
- Day 1-15 → Predict Day 16-18
- Day 2-16 → Predict Day 17-19
- Day 3-17 → Predict Day 18-20

</div>
</div>

---

# Architecture Details

**Model:** Sequence-to-Sequence with Attention (GRU)

**Configuration:**
- Hidden dimension: 64 units
- Number of layers: 1

---

# Benchmark Models

<div class="twocols">
<div class="col">

## Persistent Model

**Approach:** Tomorrow's weather = Today's weather

$$\hat{y}_{t+1} = y_t$$

- Simplest forecasting baseline
- Assumes weather remains constant
- Any useful model must outperform this

</div>
<div class="col">

## SARIMA Model

**Approach:** Seasonal AutoRegressive Integrated Moving Average

- Statistical time series model
- Captures seasonal patterns and trends
- Traditional method for weather forecasting
- Widely used in meteorology

</div>
</div>

---

# Results: 1-Day Forecast

<div class="twocols">
<div class="col">


| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|----------|
| **GRU (Ours)** | **1.61** | **2.06** | **0.9011** |
| Persistent | 1.78 | 2.30 | 0.8768 |
| SARIMA | 1.72 | 2.22 | 0.8848 |

</div>
<div class="col">

**Key Observations:**
- GRU outperforms both baselines
- 9.6% improvement over Persistent model
- 6.4% improvement over SARIMA
- Explains 90% of temperature variance

</div>
</div>

---

# Results: 3-Day Forecast

<div class="twocols">
<div class="col">


| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|----------|
| **GRU (Ours)** | **2.58** | **3.26** | **0.7525** |
| Persistent | 3.04 | 3.86 | 0.6524 |
| SARIMA | 2.73 | 3.47 | 0.7191 |

</div>
<div class="col">

**Key Observations:**
- GRU maintains advantage at longer horizons
- 15.1% improvement over Persistent model
- 5.5% improvement over SARIMA
- Performance degrades gracefully with forecast horizon

</div>
</div>

---

# Execution Time

<div class="twocols">
<div class="col">

**Model Training/Computation Time:**

| Model | Time | vs SARIMA |
|-------|------|-----------|
| **GRU (training to best)** | **279.1s** | **12.9% faster** |
| Persistent | 6.9ms | - |
| SARIMA | 320.5s | baseline |

*Note: GRU time includes full training, not just inference*

</div>
<div class="col">

**Key Observations:**
- GRU achieves best performance in ~5 minutes of training
- Persistent model is instantaneous (naive baseline, no training)
- SARIMA takes longer to train than GRU
- GRU offers best accuracy-to-training-time ratio

</div>
</div>

---

<!-- header: '6.1 Forecast Evaluation' -->
<!-- _class: title -->

# Forecast Evaluation

---

<div style="text-align: center;">

![w:1000](temperature_forecast_evaluation.png)

</div>

---

<!-- header: '7. Questions?' -->
<!-- _class: title -->

# Questions?