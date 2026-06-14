# 📈 BBCA Stock Price Prediction — Bidirectional LSTM

A deep learning project that predicts **PT Bank Central Asia Tbk (BBCA.JK)** stock prices using a Bidirectional LSTM model with automated hyperparameter tuning and walk-forward cross-validation.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| MAPE | 1.83% |
| R² | 0.9628 |
| RMSE | 181.46 IDR |
| MAE | 143.17 IDR |
| CV Std (fold stability) | 0.000063 |
| Directional Accuracy | ~39% |

> **Note:** Directional accuracy of ~39% remained consistent across all experiments (different architectures, targets, and dataset sizes), suggesting this represents the information ceiling of daily BBCA returns predictable from technical indicators alone.

---

## 🏗️ Model Architecture

```
Input (60 timesteps × 13 features)
    ↓
Bidirectional LSTM (256 units, return_sequences=True)
    ↓
Dropout (0.1)
    ↓
LSTM (128 units)
    ↓
Dropout (0.1)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (1 unit) → predicted Close price
```

Hyperparameters were found using **Keras Tuner — Bayesian Optimization** (25 trials).

---

## ✨ Key Features

- **Automated hyperparameter search** — Keras Tuner with Bayesian Optimization across 324 possible combinations
- **Walk-forward cross-validation** — 3 manually balanced folds to prevent data leakage and ensure each fold covers a distinct market regime
- **13 engineered features** — Close, Volume, HL Range, Daily Return, EMA 9/21/50/200, EMA crossover ratios, volatility regime
- **Volatility regime detection** — binary flag for high-volatility periods (top quartile of realized 20-day vol)
- **Monte Carlo forecast** — 100 simulations with Dropout active at inference time, producing a 90% confidence band
- **Comprehensive diagnostics** — Durbin-Watson test, lag scatter plot, residual distribution, directional accuracy

---

## 📁 Project Structure

```
bbca-lstm-prediction/
├── BBCA_LSTM.ipynb   # Main notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Python + model file exclusions
```

---

## 🔧 Features Used

| Feature | Description |
|---------|-------------|
| `Close` | Daily closing price |
| `Volume` | Trading volume |
| `HL_Range` | High - Low (intraday volatility) |
| `Daily_Return` | Daily % price change |
| `EMA_9/21/50/200` | Exponential moving averages |
| `EMA_9_21_ratio` | Short-term momentum signal |
| `EMA_50_200_ratio` | Golden/death cross signal |
| `Close_EMA50_gap` | Mean reversion distance |
| `Realized_Vol_20` | 20-day annualised realized volatility |
| `Vol_Regime` | Binary high-vol flag (top 25% of history) |

---

## 🔁 Training Pipeline

```
1. Download BBCA.JK data (2015–present) via yfinance
2. Feature engineering (EMAs, volatility regime)
3. MinMaxScaler fit on train only (no data leakage)
4. Keras Tuner — Bayesian Optimization (25 trials, middle CV fold)
5. Manual walk-forward CV (3 folds, val_size=400)
6. Best fold saved by lowest val_loss
7. Evaluate on held-out test set (25% of data)
8. Monte Carlo 30-day forecast
```

### Cross-Validation Fold Structure

```
Fold 1: train 2015→2018 (837 rows)  | val 2018→2019 (400 rows)
Fold 2: train 2015→2019 (1237 rows) | val 2019→2021 (400 rows)
Fold 3: train 2015→2021 (1637 rows) | val 2021→2023 (400 rows)
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bbca-lstm-prediction.git
cd bbca-lstm-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `BBCA_LSTM.ipynb` in Jupyter or Google Colab and run all cells in order.

> 💡 **Recommended:** Use Google Colab with T4 GPU for ~10x faster training.

---

## 📈 Key Findings

### What worked
- Extending data from 2019 → 2015 significantly stabilized CV fold variance (std: 0.00185 → 0.000063)
- Manual balanced walk-forward folds outperformed scikit-learn's `TimeSeriesSplit` for this dataset size
- Larger model (256 LSTM units) justified by the larger dataset — confirmed by Keras Tuner convergence
- Volatility regime feature improved crash period awareness

### What didn't work
- `PREDICT_RETURNS = True` (predicting daily % change instead of price level) — model outputted near-constant positive values despite StandardScaler, unable to learn directional signal
- Extending data to 2010 — regime mismatch (BBCA traded at 500–2,000 IDR pre-2015 vs 2,000–10,000 IDR post-2015) caused MAPE to jump from 1.83% to 15.30%
- Return prediction did not improve directional accuracy beyond the ~39% ceiling

### Directional accuracy ceiling
Directional accuracy converged to ~39% regardless of architecture, target variable, or hyperparameters. This is consistent with the **semi-strong form of market efficiency** — daily price movements of a large-cap stock like BBCA are largely unpredictable from historical technical data alone.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. The model's predictions should not be used as sole trading signals. Always combine quantitative models with fundamental analysis and proper risk management.

---

## 🛠️ Built With

- [Keras](https://keras.io/) / [TensorFlow](https://www.tensorflow.org/)
- [Keras Tuner](https://keras.io/keras_tuner/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [scikit-learn](https://scikit-learn.org/)
- [statsmodels](https://www.statsmodels.org/)
