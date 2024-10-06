#%%
import yfinance as yf
import pandas as pd
import numpy as np
from pytz import timezone
# %%
NV = yf.Ticker('NVDA')
# %%
print(NV)
# %%
nv_hist = NV.history(period='5y', interval='1mo')
# %%
type(nv_hist)
# %%
nv_hist.head()

# %%
stocks = pd.read_csv('data_branch2_v3.csv')
# %%
companies = list(set(list(stocks['Name'])))
# %%
tickers = {}

for company in companies:
    tickers[company]=(yf.Ticker(company).history(period='8y', interval='1wk'))
#%%
stocks['End'] = stocks['End'] + ' 00:00:00-0400'
#%%
stocks['End'] = pd.to_datetime(stocks['End'])
#%%
for index in stocks['Name'].index:
    x_init = 0
    temp = tickers[stocks['Name'][index]]
    new_index = list(range(0,len(stocks['Name'].index)))

    temp = temp.reset_index().reindex(new_index)

    final = stocks.at[index, 'End']
    initial = stocks.at[index, 'End'] - pd.DateOffset(months=3)

    # tz = timezone('America/New_York')
    # final = tz.localize(final)
    try:
        x_init = 0
        while initial >= temp.at[x_init, 'Date']:
            x_init+=1
        # x_init-=1
    except Exception as e:
        print(e)

    try:
        x_fin = x_init
        while stocks.at[index, "End"] >= temp.at[x_fin, 'Date']:
            x_fin+=1
        # x_fin-=1
    except Exception as e:
        print(e)
    stocks.at[index, 'Volume'] = temp['Volume'][x_init:x_fin].mean()
    stocks.at[index, 'st_dev'] = temp['Open'][x_init:x_fin].std()
    stocks.at[index, 'open'] = temp.at[x_init, 'Open']
    stocks.at[index, 'close'] = temp.at[x_fin, 'Close']
    stocks.at[index, 'delta'] = temp.at[x_fin, 'Close'] - temp.at[x_init, 'Open']
    #print(f"{index}: {temp.at[x_fin, 'Close']} - {temp.at[x_init, 'Open']}")
stocks
# %%
stocks.to_csv('data_branch2_v4.csv', index=True)
# %%
