import numpy as np
import pandas as pd

ds_df = pd.read_csv('../../input/paper3/bitstampUSD.csv', names=["timestamp", "rate", "volume"])

ds_df = ds_df.set_index(['timestamp'])

ds_df.index = pd.to_datetime(ds_df.index, unit='s')

date_start = '2013-03-01'
date_end = '2017-04-01'

print(ds_df[date_start, date_end].shape)

exit(0)

l_periods = np.array(['10Min', '15Min', '30Min', '60Min'])

for period in l_periods:
    ds_period_df = pd.concat([ds_df['rate'].resample(period).ohlc(),
                              ds_df['volume'].resample(period).sum()], axis=1)
    ds_period_df = ds_period_df.dropna()
    ds_period_df = ds_period_df[date_start:date_end]
    ds_period_df.to_csv('../../output/paper3/bitcoin_' + period + '.csv', encoding='utf-8')
