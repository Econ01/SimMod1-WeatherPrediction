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
    padding: 1rem;
    font-size: 1.2rem;
  }

  table tr:nth-child(even) {
    background: transparent;
  }

  table thead {
    display: none;
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

<!-- _header: '' -->

# Table of Contents

1. Problem Description
2. Baseline Benchmark
3. Challenges
4. Random Forest Method
5. Linear Regression
6. Neural Network

<div class="img-full-height">
<img src="clouds2.jpg" alt="Clouds">
</div>

---

<!-- header: '1. Problem Description' -->

# Problem Description

**Objective:** Predict weather conditions using various machine learning methods

**Data Source:** Köln-Bonn Weather Station

- Historical data from 1950s to present
- Daily measurements of 12 weather variables
- Over 24,000 data points available for training and evaluation

---

# Input Variables

<div style="display: flex; justify-content: center;">

| | | |
|---|---|---|
| Mean Temperature | Minimum Temperature | Maximum Temperature |
| Precipitation | Sea Level Pressure | Sunshine Duration |
| Humidity | Wind Speed | Wind Gust |
| Cloud Cover | Snow Depth | Global Radiation |

</div>

---

<!-- header: '2. Baseline Benchmark' -->

# Baseline Benchmark

## Persistence Method

The simplest forecasting approach: **tomorrow's weather = today's weather**

$$\hat{y}_{t+1} = y_t$$

- Assumes weather patterns remain constant
- Provides a minimum performance threshold
- Any useful model must outperform this baseline

---

# Evaluation Metrics

<div class="twocols">
<div class="col">

## Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Average of absolute differences between predictions and actual values

</div>
<div class="col">

## Root Mean Square Error (RMSE)

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Penalizes larger errors more heavily than MAE

</div>
</div>

---

<!-- header: '3. Challenges' -->

# Challenges

<div class="twocols">
<div class="col">

## Single Station Limitation

- Data from only one location (Köln-Bonn)
- Unable to detect approaching weather fronts from surrounding regions
- Limits accuracy for multi-day forecasts (3-day, 7-day)

</div>
<div class="col">

## Data Quality Issues

- Missing values in historical records require imputation
- Measurement inconsistencies across decades of data

</div>
</div>

---

<!-- header: '4. Random Forest Method' -->
<!-- _class: title -->

<div class="twocols">
<div class="col-center">

<img src="RandomForest.png" alt="Random Forest" style="width: 100%;">

</div>
<div class="col-center">

# Random Forest Method

</div>
</div>

---

# How It Works

<div class="twocols" style="align-items: center;">
<div class="col">

<img src="RFAlgo.png" alt="Random Forest Diagram" style="width: 100%;">

</div>
<div class="col" style="font-size: 1.3rem;">

A **supervised** learning method for regression/classification

We use past weather to predict tomorrow

*E.g. yesterday's temperature, the past week's trend, etc.*

We build many small "if–then" decision trees.

*Each tree is trained on a random subset of the data, and makes a rough prediction like: "If last week was cold, tomorrow is likely cold."*

The forest **averages** all of them to get a more stable prediction.

***Crowd wisdom:** many simple opinions = one accurate forecast.*

</div>
</div>

---

# Advantages & Limitations

<div class="twocols">
<div class="col">

## Advantages

- Captures nonlinear temperature patterns
- Reduces risk of overfitting (multiple decision trees)
- Handles data with missing values
- Provides Feature Importance Scores, aiding in Feature Selection

</div>
<div class="col">

## Limitations

- Weak for long-term prediction
- Computationally expensive, increased run time
- Lack of interpretability, "black-box" model

</div>
</div>

---

<!-- header: '5. Linear Regression' -->
<!-- _class: title -->

<div class="twocols">
<div class="col-center">

<img src="linearRegression.png" alt="Linear Regression" style="width: 100%;">

</div>
<div class="col-center">

# Linear Regression

</div>
</div>

---

# How It Works

<div class="twocols">
<div class="col-center">

- Finds a **linear relationship** between input and output

$$T_{t+1} = w_1 \cdot T_t + w_2 \cdot H_t + w_3 \cdot W_t + \cdots + b$$

- Finds weights for features (The model learns the optimal weights)

- Prediction = Weighted sum of all features

</div>
<div class="col">

<img src="Multiple-linear-regression-Methodology.png" alt="Linear Regression Diagram" style="width: 100%;">

</div>
</div>

---

# Implementation & Data

<div class="twocols">
<div class="col">

## Implementation

- 30 days input → 7 days output
- 7 separate models
- Training on 65 years of data

<img src="data.jpg" alt="Linear Regression Implementation" style="width: 100%; height: 150px; object-fit: cover; border-radius: 20px;">

</div>
<div class="col">

## Data

- 24,929 days (1957-2025)
- 11 weather features
- 80/10/10 split (Train/Val/Test)

## Why Linear Regression?

Baseline for comparison

</div>
</div>

---

<!-- header: '6. Neural Network' -->
<!-- _class: title -->

<div class="twocols">
<div class="col-center">

<img src="nn.png" alt="Neural Network" style="width: 100%;">

</div>
<div class="col-center">

# Neural Network

</div>
</div>

---

# How It Works

## Sequence-to-Sequence (Seq2Seq) Architecture

A neural network designed for **sequence-to-sequence** tasks

- **Encoder:** Reads the input sequence (30 days of weather data)
  - Compresses information into a "context vector"

- **Decoder:** Generates the output sequence (1-day forecast)
  - Uses the context to predict the future day

- **Attention Mechanism:** Allows decoder to focus on relevant past days
  - Not all input days are equally important for each prediction

---

# Proposed Model Architecture

<div class="twocols">
<div class="col-center">

<img src="gru-3.svg" alt="Model Architecture" style="width: 100%;">

</div>
<div class="col">

## Encoder

- **GRU** (Gated Recurrent Unit)
- 2 layers, 256 hidden units

## Decoder

- **Autoregressive** generation
- Predicts one day at a time
- Uses previous prediction as input
- Attention-weighted context

</div>
</div>

---

# Implementation Details

<div class="twocols">
<div class="col">

## Configuration

- 30 days input → 1 day output
- Single target: Mean Temperature
- 11 input features
- ~1.5M trainable parameters

</div>
<div class="col">

## Training

- Year-based split (no data leakage)
  - Train: 1957-2022
  - Validation: 2023
  - Test: 2024-2025
- Early stopping with patience

</div>
</div>

---

# Advantages & Limitations

<div class="twocols">
<div class="col">

## Advantages

- Captures complex non-linear patterns
- Learns long-term temporal dependencies
- Automatic feature extraction
- Scales well with more data

</div>
<div class="col">

## Limitations

- Computationally expensive to train
- Requires large amounts of data
- "Black-box" - hard to interpret
- Prone to overfitting

</div>
</div>

---

<!-- _class: title -->
<!-- _header: '' -->

# Questions?
