#%%
from openbb_terminal.sdk import openbb
# %%
import pandas as pd
import numpy as np
import time
import math
# %%
# create a dataframe named df that reads from a csv file named 'ALL_DATA.csv'
df = pd.read_csv('ALL_DATA.csv')

# %%
# AMZN = openbb.stocks.fa.income('AMZN', source="AlphaVantage", quarterly=True)
# %%
# AMZN
# %%
df

# %%
#loop through the column 'name' in the dataframe df and count how many times each name appears
#store the results in a dictionary named 'counts'
counts = {}
for name in df['Name']:
    if name in counts:
        counts[name] += 1
    else:
        counts[name] = 1
# %%
# creata a new column of df named 'Revenue' and make all the values Nan
df['Revenue'] = np.nan
#%%
# create a new dataframe named 'df2' that contains the same columns as df
open_income = ['totalRevenue', 'costofGoodsAndServicesSold', 'researchAndDevelopment' ]
open_balance = ['totalAssets', 'totalLiabilities', 'inventory', 'commonStockSharesOutstanding']
df2 = pd.DataFrame(columns=(['Name'] + open_income + open_balance))
#print the values of the keys of counts
# %%
# Loop through the dataframe df and for each row, take the value under column 'End' and match it
# to the correct value for totalRevenue using the openbb function stocks.fa.income
# store the result in the new column 'Revenue'
# to call the openbb function, use the syntax openbb.stocks.fa.income('ticker', source='AlphaVantage', quarterly=True)
# where 'ticker' is the ticker symbol of the company you want to look up
# source='AlphaVantage' means you want to use the AlphaVantage API to get the data
# quarterly=True means you want to get quarterly data
#ensure that every revenue value matches the correct end date from the dataframe df
#use time.sleep(15) to ensure that you don't exceed the API call limit
# to get the value for the correct date, use the syntax AMZN['2020-12-31'].loc['totalRevenue']
# where AMZN is the variable name of the dataframe returned by the openbb function
# and '2020-12-31' is the date you want to look up
# and 'totalRevenue' is the name of the column you want to look up
# convert the value of the end column of df to a string with the format yyyy-mm-dd and remove the excess before using it to index the AMZN dataframe
# store the converted value in a variable named 'end' and use that variable to index the AMZN dataframe, don't change the preexisting dataframe end column
# use the syntax df['Revenue'][i] = AMZN[end].loc['totalRevenue']
# where i is the index of the row you are currently on
x=0
for name in list(counts.keys()):
    try:
        temp = openbb.stocks.fa.income(name, source='AlphaVantage', quarterly=True, limit=20)
        rev_df = temp.loc['totalRevenue'].transpose()
        rev_df = rev_df[~rev_df.index.duplicated()]
        
        COGS_df = temp.loc['costofGoodsAndServicesSold'].transpose()
        COGS_df = COGS_df[~COGS_df.index.duplicated()]
        
        RnD_df = temp.loc['researchAndDevelopment'].transpose()
        RnD_df = RnD_df[~RnD_df.index.duplicated()]

        time.sleep(12)
        temp = openbb.stocks.fa.balance(name, source='AlphaVantage', quarterly=True, limit=20)

        assets_df = temp.loc['totalAssets'].transpose()
        assets_df = assets_df[~assets_df.index.duplicated()]

        liabilities_df = temp.loc['totalLiabilities'].transpose()
        liabilities_df = liabilities_df[~liabilities_df.index.duplicated()]

        inventory_df = temp.loc['inventory'].transpose()
        inventory_df = inventory_df[~inventory_df.index.duplicated()]

        shares_df = temp.loc['commonStockSharesOutstanding'].transpose()
        shares_df = shares_df[~shares_df.index.duplicated()]

        temp2 = pd.concat(
            [rev_df, COGS_df, RnD_df, assets_df, liabilities_df, inventory_df, shares_df], 
            axis=1)
        
        temp2.insert(loc=0, column='Name', value='')
        for i in range(len(temp2)):
            temp2['Name'][i] = name
        df2 = pd.concat([df2, temp2], axis=0)
        time.sleep(12)

        # for i in range(0,20):
        #     try:
        #         end = str(df['End'][x])
        #         end = end[0:10]
        #         df['Revenue'][x] = temp[end].loc['totalRevenue']
        #     except KeyError:
        #         df['Revenue'][i] = np.nan
        #         print("Name", name, "Index", i, "KeyError", sep=" ")
        #     x+=1
        # time.sleep(10)
    except:
        pass

# %%
open_balance = ['totalAssets', 'totalLiabilities', 'inventory', 'commonStockSharesOutstanding']
df_balance = ['Assets', 'Liabilities', 'InventoryNet', 'SharesOutstanding']
#%%
#count the number of empty or NaN values in the 'GrossProfit' column of df
#store the result in a variable named 'empty'
empty = df['Revenue'].isna().sum()
# %%
df2.to_csv('openBB_data_version1.csv')
# %%
