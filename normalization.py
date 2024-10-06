#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %%
df = pd.read_csv('openBB_Data_v3.csv')
# %%
# for row in df.index:
#     if df.at[row, 'delta'] > 0:
#         df.at[row, 'classification'] = 1
#     else:
#         df.at[row, 'classification'] = 0

# %%
# X = data[['totalRevenue', 'costofGoodsAndServicesSold',
#        'totalAssets', 'totalLiabilities', 'commonStockSharesOutstanding', 
#        'GDP', 'Interest', 'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
#        'Unemployment_Rate', 'Recession_probability', 'Corporate_Profits']]

#%%
#Using matplotlib.pyplot, plot the distribution of the target variable 'finalPrice'
#plt.hist(data['totalRevenue'])
#Use log scaling to normalize the data for totalRevenue and replace the column
df['totalRevenue'] = np.log(df['totalRevenue'])
#%%
plt.hist(df['Asset/share'])
df['Asset/share'] = np.log(df['Asset/share'])
plt.hist(df['Asset/share'])
#%%
plt.hist(df['Liab/Rev'])
df['Liab/Rev'] = np.log(df['Liab/Rev'])
plt.hist(df['Liab/Rev'])
#%%
#plt.hist(df['totalAssets'])
df['totalAssets'] = np.log(df['totalAssets'])
plt.hist(df['totalAssets'])
# %%
#plt.hist(df['totalLiabilities'])
df['totalLiabilities'] = np.log(df['totalLiabilities'])
plt.hist(df['totalLiabilities'])
# %%
# plt.hist(df['commonStockSharesOutstanding'])
df['commonStockSharesOutstanding'] = np.log(df['commonStockSharesOutstanding'])
plt.hist(df['commonStockSharesOutstanding'])
# %%
#plt.hist(df['GDP'])
#Normalize the data for GDP and replace the column
#Use the z-score method
df['GDP'] = (df['GDP'] - df['GDP'].mean())/df['GDP'].std()
plt.hist(df['GDP'])
# %%
#Not going to normalize because the scale is already small
plt.hist(df[0:20]['Interest'])
# df['Interest'] = (df['Interest'] - df['Interest'].mean())/df['Interest'].std()
# plt.hist(df['Interest'])

# %%
temp = 'Inflation'
# plt.hist(df[0:20][temp])
df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])

# %%
temp = 'SP'
#plt.hist(df[0:20][temp])
df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])

# %%
temp = 'Cons_Sent'
# plt.hist(df[0:20][temp])
df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])
# %%
#Not going to normalize because the scale is already small
temp = 'Fin_Stress'
plt.hist(df[0:20][temp])
# df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
# plt.hist(df[temp])
# %%
temp = 'Cash_On_Hand'
#plt.hist(df[0:20][temp])
df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])

# %%
temp = 'Unemployment_Rate'
#plt.hist(df[0:20][temp])
#Use log scaling to normalize the data for Unemployment_Rate and replace the column
df[temp] = np.log(df[temp])
# df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])
# %%
temp = 'Recession_probability'
#plt.hist(df[0:20][temp])
# Have to use the normal distribution because 0 is in the data
df[temp] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])
# %%
temp = 'Corporate_Profits'
plt.hist(df[0:20][temp])
df['Corporate_Profits'] = (df[temp] - df[temp].mean())/df[temp].std()
plt.hist(df[temp])
# %%
df.to_csv('openBB_Data_v4_Normalized.csv')

# %%
