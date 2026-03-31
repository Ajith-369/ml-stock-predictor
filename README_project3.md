# ML Stock Price Predictor

**Tech Stack:** Python, Pandas, NumPy, yfinance, pandas-datareader, Scikit-learn, PyTorch, Matplotlib

---

## What This Project Does

Predicts daily stock returns for 10 stocks using **CAPM and Fama-French 3-Factor Model** as ML features. Compares **Linear Regression vs Neural Network (PyTorch)** performance on a held-out test set.

Key finding: **Linear Regression outperforms Neural Network for 9 out of 10 stocks** — demonstrating that model complexity is not always better on noisy financial data with limited features.

---

## Portfolio — 10 Stocks Across 6 Sectors

| Ticker | Company | Sector |
|--------|---------|--------|
| NVDA | Nvidia | Technology / AI Chips |
| AMD | Advanced Micro Devices | Technology / Chips |
| MSTR | MicroStrategy | Technology / Bitcoin Proxy |
| GS | Goldman Sachs | Finance / Banking |
| MSFT | Microsoft | Technology / Cloud |
| COST | Costco | Retail / Consumer |
| EBAY | eBay | E-Commerce |
| PYPL | PayPal | FinTech / Payments |
| F | Ford | Automotive |
| LMT | Lockheed Martin | Defense |

---

## Key Concepts

| Term | Plain English |
|------|--------------|
| **CAPM** | Expected return = Risk Free Rate + Beta × Market Return |
| **Fama-French** | CAPM + Size factor (SMB) + Value factor (HML) |
| **Beta** | How much a stock moves vs the market. >1 = more volatile |
| **Alpha** | Extra return beyond what the model predicts |
| **Mkt-RF** | Market return minus Risk Free Rate |
| **SMB** | Small Minus Big — small caps tend to outperform large caps |
| **HML** | High Minus Low — cheap stocks tend to outperform expensive ones |
| **RMSE** | Root Mean Square Error — average prediction error. Lower = better |

---

## Methodology

### Data
- **Stock prices:** 10 stocks, Jan 2021 – Mar 2026 (~1,300 trading days)
- **Why 2021?** Excludes COVID crash — abnormal patterns distort model learning
- **Fama-French factors:** Downloaded from Kenneth French's data library via `pandas-datareader`

### Features & Target
- **Features (X):** Mkt-RF, SMB, HML, RF — the 4 Fama-French factors
- **Target (y):** Daily stock return per ticker
- **Why factors, not prices?** Factors explain *why* a stock moves, not just *what* it did

### Train/Test Split (Time-Based)
- **Train:** Jan 2021 → Sep 2025 (~1,190 days)
- **Test:** Oct 2025 → Jan 2026 (~84 days, never seen during training)
- **Critical:** Random split causes data leakage in financial time series — always split by date

### Models
1. **Linear Regression** (Scikit-learn) — one model trained per stock
2. **Neural Network** (PyTorch) — 3-layer feedforward: 4 → 32 → 16 → 1

---

## Results

### NVDA — Model Comparison
| Model | RMSE | Avg Daily Error |
|-------|------|----------------|
| Linear Regression | 0.0146 | 1.46% |
| Neural Network | 0.0215 | 2.15% |
| **Winner** | **Linear Regression** | |

### All 10 Stocks — Linear Regression
| Stock | RMSE | Beta | Alpha |
|-------|------|------|-------|
| COST | 0.0134 | 0.76 | +0.0010 |
| GS | 0.0135 | 1.25 | -0.0001 |
| MSFT | 0.0144 | 1.05 | +0.0008 |
| NVDA | 0.0146 | 1.86 | +0.0017 |
| LMT | 0.0161 | 0.37 | +0.0004 |
| PYPL | 0.0172 | 1.27 | **-0.0018** |
| F | 0.0194 | 1.29 | +0.0009 |
| EBAY | 0.0225 | 0.86 | -0.0001 |
| MSTR | 0.0385 | 2.07 | +0.0013 |
| AMD | 0.0391 | 1.74 | +0.0007 |

### Final Model Comparison
- **Linear Regression wins:** 9/10 stocks
- **Neural Network wins:** 1/10 stocks (COST only)

---

## Interview-Ready Insights

**1. Higher Beta = Harder to predict**
> MSTR and AMD have the highest Beta (>1.7) and the highest RMSE. Volatile stocks move on news and sentiment that factor models cannot capture.

**2. PYPL negative Alpha — consistent signal**
> PYPL showed negative Alpha across both this ML model and Project 1's Sharpe Ratio analysis. It consistently underperforms what its risk exposure should earn.

**3. Linear > Neural Network with limited features**
> With only 4 Fama-French features, the factor-return relationship is approximately linear. Neural networks need many features and large datasets to outperform simpler models. This is Occam's Razor in ML — simpler wins on noisy data.

**4. Data leakage is a real risk**
> Using random train/test split on financial time series leaks future information into training, artificially inflating model performance. Always split by date.

---

## Setup

```bash
pip install yfinance pandas-datareader scikit-learn torch
```

Run the notebook: `ml_stock_predictor_clean.ipynb`

---

## Related Projects

- [Stock Portfolio Analyzer](https://github.com/Ajith-369/stock-portfolio-analyzer) — Python portfolio analysis (Project 1)
- [SQL Financial Dashboard](https://github.com/Ajith-369/sql-financial-dashboard) — PostgreSQL + SQL analysis (Project 2)
