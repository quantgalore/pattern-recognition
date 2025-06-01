# -*- coding: utf-8 -*-
"""
Created in 2025

@author: Quant Galore
"""

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from pandas_market_calendars import get_calendar
from datetime import datetime, timedelta
from tslearn.metrics import dtw
from sklearn.metrics import classification_report

polygon_api_key = "KkfCQ7fsZnx0yK4bhX9fD81QplTh0Pf3"
calendar = get_calendar("NYSE")

trading_dates = calendar.schedule(start_date="2020-01-01", end_date= (datetime.today() - timedelta(days=1))).index.strftime("%Y-%m-%d").values

ticker = "SPY"

# =============================================================================
# Initial Data Collection
# =============================================================================

data_list = []

# date = trading_dates[0]
for date in trading_dates:
    
    try:

        underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
        underlying_data.index = pd.to_datetime(underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        
        intraday_data = underlying_data[(underlying_data.index.time >= pd.Timestamp("09:30").time()) & (underlying_data.index.time <= pd.Timestamp("16:00").time())].copy()
        intraday_data["date"] = intraday_data.index.strftime("%Y-%m-%d")
        
        intraday_data["returns"] = round(((intraday_data["c"] - intraday_data["c"].iloc[0]) / intraday_data["c"].iloc[0]) * 100, 2)
        
        data_list.append(intraday_data)
        
    except Exception as error:
        print(error)
        continue

full_dataset = pd.concat(data_list)

# =============================================================================
# Testing + Plotting
# =============================================================================

cutoff_time = "15:00"

dates_covered = np.sort(full_dataset["date"].drop_duplicates().values)

# Backtest over the last 20 days
backtest_dates = dates_covered[-20:]

outcome_data_list = []

# backtest_date = backtest_dates[-5]
for backtest_date in backtest_dates:
    
    try:
            
        historical_data = full_dataset[full_dataset["date"] < backtest_date].copy()
        historical_dates = historical_data["date"].drop_duplicates().values
        
        daily_data = full_dataset[full_dataset["date"] == backtest_date].copy()
        daily_pre_trade_data = daily_data[daily_data.index.time <= pd.Timestamp(cutoff_time).time()].copy()
        
        daily_pre_trade_data["hour"] = daily_pre_trade_data.index.hour
        daily_pre_trade_data["minute"] = daily_pre_trade_data.index.minute
        
        distance_list = []
        
        # historical_date = historical_dates[0]
        for historical_date in historical_dates:
            
            historical_daily_data = historical_data[historical_data["date"] == historical_date].copy()
            historical_pre_trade_data = historical_daily_data[historical_daily_data.index.time <= pd.Timestamp(cutoff_time).time()].copy()
            
            # If the market closed before 3pm that day, don't include it.
            if len(historical_pre_trade_data) < 300: continue
        
            historical_pre_trade_data["hour"] = historical_pre_trade_data.index.hour
            historical_pre_trade_data["minute"] = historical_pre_trade_data.index.minute
        
            combined_data = pd.merge(left=daily_pre_trade_data, right = historical_pre_trade_data, on = ["hour", "minute"])
            
            # Comparing "today's" cutoff data to the cutoff data of a historical date.
            euclidean_distance = np.linalg.norm(combined_data["returns_x"].values - combined_data["returns_y"].values)
            
            historical_post_trade_data = historical_daily_data[historical_daily_data.index.time >= pd.Timestamp(cutoff_time).time()].copy()
            historical_post_trade_return = round(((historical_post_trade_data["c"].iloc[-1] - historical_post_trade_data["c"].iloc[0]) / historical_post_trade_data["c"].iloc[0]) * 100, 2)
            historical_binary_post_trade_return = 1 if historical_post_trade_return > 0 else 0 
    
            distance_data = pd.DataFrame([{"date": historical_date, "euc_distance": euclidean_distance, "forward_return": historical_post_trade_return,
                                           "forward_return_direction": historical_binary_post_trade_return}])
            
            distance_list.append(distance_data)
            
        full_euc_distances = pd.concat(distance_list).sort_values(by="euc_distance", ascending=True)
        
        data_0 = full_dataset[full_dataset["date"] == full_euc_distances["date"].iloc[0]].copy()
        data_1 = full_dataset[full_dataset["date"] == full_euc_distances["date"].iloc[1]].copy()
        data_2 = full_dataset[full_dataset["date"] == full_euc_distances["date"].iloc[2]].copy()
        
        plt.figure(figsize=(10, 6), dpi=800)    
        plt.xticks(rotation=45)
        plt.title("Euclidean Distance - Pattern Matching")
        plt.xlabel("Minutes since start of trading session")
        plt.ylabel("Intraday Return (%)")
        plt.plot(np.arange(0, len(daily_data)), daily_data["returns"])
        plt.plot(np.arange(0, len(daily_data)), data_0["returns"])
        plt.plot(np.arange(0, len(daily_data)), data_1["returns"])
        plt.plot(np.arange(0, len(daily_data)), data_2["returns"])
        plt.axvline(x=331, color='gray', linestyle='--', linewidth=1)
        
        plt.legend(["Current Trading Day", "Closest Historical Pattern",
                    "Runner-Up Pattern Match", "Third-Likeliest Analog",
                    "3:00 PM Cutoff"])
        
        plt.show()
        plt.close()
        
        most_similar = full_euc_distances.copy().head(10)
        
        likelihood_of_up = len(most_similar[most_similar["forward_return_direction"] == 1]) / len(most_similar)
        theo_expected_vol = abs(most_similar["forward_return"]).mean()
        
        daily_forward_data = daily_data[daily_data.index.time >= pd.Timestamp(cutoff_time).time()].copy()
        daily_forward_return = round(((daily_forward_data["c"].iloc[-1] - daily_forward_data["c"].iloc[0]) / daily_forward_data["c"].iloc[0]) * 100, 2)
        daily_forward_return_binary = 1 if daily_forward_return > 0 else 0 
        
        theo_prediction = 1 if likelihood_of_up > 0.5 else 0
        theo_proba = likelihood_of_up if likelihood_of_up > 0.5 else 1 - likelihood_of_up
        
        outcome_data = pd.DataFrame([{"date": backtest_date,"expected_vol": theo_expected_vol,
                                      "pred": theo_prediction, "proba": theo_proba, "1_proba": likelihood_of_up,
                                      "actual_raw": daily_forward_return, "actual_binary": daily_forward_return_binary}])
    
        outcome_data_list.append(outcome_data)
        
    except Exception as error:
        print(error)
        continue
        
full_outcome_data = pd.concat(outcome_data_list)

full_outcome_data["theo_pnl"] = full_outcome_data.apply(lambda x: 1000 * (x["actual_raw"]/100) if x["1_proba"] >= 0.5 else 1000 * (x["actual_raw"]/-100), axis = 1)
full_outcome_data["theo_capital"] = 1000 + full_outcome_data["theo_pnl"].cumsum()

plt.figure(figsize=(10, 6), dpi=800)    
plt.xticks(rotation=45)
plt.title(f"Euclidean Distance - Majority-Rule Predictor")
plt.xlabel("Date")
plt.ylabel("Growth of $1,000")
plt.plot(pd.to_datetime(full_outcome_data["date"]), full_outcome_data["theo_capital"])
plt.legend(["Gross PnL Curve"])
plt.show()
plt.close()

accuracy_rate = len(full_outcome_data[(full_outcome_data["pred"] == full_outcome_data["actual_binary"])]) / len(full_outcome_data)
print(f"Accuracy Rate: {round(accuracy_rate*100, 2)}%")
