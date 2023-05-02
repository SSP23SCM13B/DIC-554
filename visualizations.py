import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

bitstamp = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv")
bitstamp['Timestamp'] = [datetime.fromtimestamp(x) for x in bitstamp['Timestamp']]
bitstamp.set_index("Timestamp", inplace=True)

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(bitstamp["Weighted_Price"])
ax.set_title("Bitcoin Weighted Price")
plt.show()
def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()

    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()

    print(df.head())
    print(df.isnull().sum())
fill_missing(bitstamp)
bitstamp_non_indexed = bitstamp.copy()
bitstamp = bitstamp.set_index('Timestamp')

plt.clf()
plt.figure(figsize=(15,12))
plt.suptitle('Lag Plots', fontsize=22)

plt.subplot(3,3,1)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1) #minute lag
plt.title('1-Minute Lag')

plt.subplot(3,3,2)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=60) #hourley lag
plt.title('1-Hour Lag')

plt.subplot(3,3,3)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1440) #Daily lag
plt.title('Daily Lag')

plt.subplot(3,3,4)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=10080) #weekly lag
plt.title('Weekly Lag')

plt.subplot(3,3,5)
pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=43200) #month lag
plt.title('1-Month Lag')

plt.legend()
plt.show()




bi# Make a copy of the bitstamp dataframe
bitstamp2 = bitstamp.copy()

# Concatenate bitstamp and df dataframes
bitstamps = pd.concat([bitstamp2, df])

# Resample the concatenated dataframe by daily frequency and compute the mean
bitstamp_daily = bitstamps.resample("24H").mean()

# Set Timestamp column as index
df = bitstamp_daily.set_index("Timestamp")

# Reset index and keep Timestamp column
df.reset_index(drop=False, inplace=True)

# Define lag features and rolling window sizes
lag_features = ["Open", "High", "Low", "Close","Volume_(BTC)"]
window1 = 3
window2 = 7
window3 = 30

# Compute rolling means for the defined lag features and window sizes
df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)
df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

# Compute rolling standard deviations for the defined lag features and window sizes
df_std_3d = df_rolled_3d.std().shift(1).reset_index()
df_std_7d = df_rolled_7d.std().shift(1).reset_index()
df_std_30d = df_rolled_30d.std().shift(1).reset_index()

# Add lagged mean and standard deviation features to the dataframe for each lag feature and window size
for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

# Replace missing values with mean of the corresponding column
df.fillna(df.mean(), inplace=True)

# Set Timestamp column as index
df.set_index("Timestamp", drop=False, inplace=True)

# Print the head of the resulting dataframe
df.head()