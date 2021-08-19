#! /usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv

preddf = read_csv("to_plot_df.csv")
preddf = preddf.set_index("hour_res")
preddf.index = pd.to_datetime(preddf.index)

# Plot the forecast with the actuals
plt.figure(figsize=(16, 8))
plt.plot(preddf.index, preddf.counts.values, label="Actual Counts")
plt.plot(preddf.index, preddf.prediction.values, label="Predited Counts")
plt.legend()
plt.suptitle('August 2019 to August 2020 Forecast vs Actuals')
plt.savefig("plots/prediction_year.png")       #Final Comparison Plotfile


fig, ax = plt.subplots(1)
fig.set_figheight(5)
fig.set_figwidth(18)
plt.plot(preddf.index, preddf.counts.values, label="Actual Counts", linewidth=1.4, color='orange')
plt.plot(preddf.index, preddf.prediction.values, label="Predited Counts", linewidth=1.4, color='brown')
ax.set_xbound(lower=pd.Timestamp('2019-08-15'), upper=pd.Timestamp('2019-09-15'))
ax.set_ylim(0, 60)
plt.legend()
plt.suptitle('August-September 2019 Forecast vs Actuals')
plt.savefig("plots/prediction_month.png")



fig = plt.figure(figsize=(16,16))
ax1 = fig.add_subplot(211)
ax1 = preddf.counts.rolling(24).sum().plot(label="Actual")
ax1 = preddf.prediction.rolling(24).sum().plot(label="Predicted")
ax1.set_xlabel("Day", fontsize=18)
ax1.set_ylabel("Counts", fontsize=18)
ax1.legend(title="Daily", fontsize=14)
ax2 = fig.add_subplot(212)
ax2 = preddf.counts.rolling(7*24).sum().plot(label="Actual")
ax2 = preddf.prediction.rolling(7*24).sum().plot(label="Predicted")
ax2.set_xlabel("Week", fontsize=18)
ax2.set_ylabel("Counts", fontsize=18)
ax2.legend(title="Weekly", fontsize=14)
plt.suptitle("Daily and Weekly Call Volume, Actual vs Prediction.png")
plt.savefig("plots/daily_weekly_comparison.png")
