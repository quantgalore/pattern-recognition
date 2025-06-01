# 🧠 Euclidean Distance Pattern Recognition

A walk-forward backtest that applies Euclidean distance to intraday S&P 500 return data to identify optically similar days — and tests whether those patterns have predictive power.

This was part of our exploration into pattern recognition methods *without the astrology*. 📉🔍

Original Post: A Junior Quant's Guide to Pattern Recognition

---

## 🚀 Overview

This script implements a simple but systematic strategy:

- Track intraday returns of SPY up to 3PM
- Use Euclidean distance to compare that day to historical days
- Identify the 10 most similar days
- Use a majority-vote from their final-hour direction to predict today’s close
- Execute a $1,000 long/short based on the prediction
- Repeat daily

The backtest is fully walk-forward and avoids data leakage.

---

## 📂 Files

- `euclidean-distance-trading.py`: Core script that runs the experiment and generates plots.
