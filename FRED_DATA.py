#%%
import requests as rq
import pandas as pd
import fredapi as fa
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import openpyxl
# %%
key = "4c8ae4eae5408f5c9bafbe39021f2afb"
fred = fa.Fred(key)
# %%
gdp = fred.get_series("GDP", observation_start='01/01/2016', 
                      observation_end='04/10/2023', frequency='q')
# %%
gdp.head(n=30)
# %%
stocks = pd.read_csv('openBB_data_version1.csv', header=0)

# %%
stocks.head()
# %%
stocks['GDP'] = np.zeros(stocks.shape[0])
#rename 'unnamed: 0' to 'End'
stocks.rename(columns={'Unnamed: 0':'End'}, inplace=True)

stocks['End']=pd.to_datetime(stocks['End'], format='%m/%d/%Y', dayfirst=False, yearfirst=False)
#%%
for i in range(0, len(stocks)):
    try:
        x = 0
        while stocks.at[i, "End"] >= gdp.index[x]:
            #print(stocks.at[i, "Date"], gdp.index[x], sep='\t')
            x+=1
        stocks.at[i,"GDP"] = (gdp.get(x-1)-gdp.get(x-2))/gdp.get(x-2)
    except Exception:
        stocks.at[i,"GDP"] = np.nan

# %%
stocks.head()
# %%
interest = fred.get_series("REAINTRATREARAT10Y", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')
# %%
stocks['Interest'] = np.empty(stocks.shape[0])
#%%
for i in range(0, len(stocks)):
    try:
        x = 0
        while stocks.at[i, "End"] >= interest.index[x]:
            x+=1
        stocks.at[i,"Interest"] = (interest.get(x-1) - interest.get(x-2))/interest.get(x-2)
    except Exception:
        stocks.at[i,"Interest"] = np.nan

# %%
inf = fred.get_series("CORESTICKM159SFRBATL", observation_start='01/01/2016', 
                      observation_end='01/01/2024', frequency='q')
# %%

#%%
stocks['Inflation'] = np.empty(stocks.shape[0])
#%%
for i in range(0, len(stocks)):
    x = 0
    try:
        while stocks.at[i, "End"] >= inf.index[x]:
            x+=1
        stocks.at[i,"Inflation"] = (inf.get(x-1) - inf.get(x-2))/inf.get(x-2)
    except IndexError:
        stocks.at[i,"Inflation"] = None
#%%
SP = fred.get_series("SP500", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')

# %%

stocks['SP'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= SP.index[x]:
            x+=1
        stocks.at[i,"SP"] = (SP.get(x-1) - SP.get(x-2))/SP.get(x-2)
    except IndexError:
        stocks.at[i,"SP"] = None

# %%
CS = fred.get_series("UMCSENT", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')
stocks['Cons_Sent'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= CS.index[x]:
            x+=1
        stocks.at[i,'Cons_Sent'] = (CS.get(x-1) - CS.get(x-2))/CS.get(x-2)
    except IndexError:
        stocks.at[i,'Cons_Sent'] = None

# %%
FS = fred.get_series("STLFSI4", observation_start='01/01/2017', 
                      observation_end='04/10/2023', frequency='q')
stocks['Fin_Stress'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= FS.index[x]:
            x+=1
        stocks.at[i,'Fin_Stress'] = (FS.get(x-1) - FS.get(x-2))/FS.get(x-2)
    except IndexError:
        stocks.at[i,'Fin_Stress'] = None
stocks.head()

# %%
FS = fred.get_series("QFRTCASHINFUSNO", observation_start='01/01/2017', 
                      observation_end='04/10/2023', frequency='q')
stocks['Cash_On_Hand'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= FS.index[x]:
            x+=1
        stocks.at[i,'Cash_On_Hand'] = (FS.get(x-1) - FS.get(x-2))/FS.get(x-2)
    except IndexError:
        stocks.at[i,'Cash_On_Hand'] = None
stocks.head()

# %%
FS = fred.get_series("UNRATE", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')
stocks['Unemployment_Rate'] = np.empty(stocks.shape[0])#THIS ONE
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= FS.index[x]:
            x+=1
        stocks.at[i,'Unemployment_Rate'] = (FS.get(x-1) - FS.get(x-2))/FS.get(x-2)
    except IndexError:
        stocks.at[i,'Unemployment_Rate'] = None
stocks.head()

# %%
FS = fred.get_series("RECPROUSM156N", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')
stocks['Recession_probability'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= FS.index[x]:
            x+=1
        stocks.at[i,'Recession_probability'] = (FS.get(x-1) - FS.get(x-2))/FS.get(x-2)
    except IndexError:
        stocks.at[i,'Recession_probability'] = None
stocks.head()


# %%
FS = fred.get_series("A053RC1Q027SBEA", observation_start='01/01/2017', 
                      observation_end='04/03/2023', frequency='q')
stocks['Corporate_Profits'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try: 
        while stocks.at[i, "End"] >= FS.index[x]:
            x+=1
        stocks.at[i,'Corporate_Profits'] = (FS.get(x-1) - FS.get(x-2))/FS.get(x-2)
    except IndexError:
        stocks.at[i,'Corporate_Profits'] = None
stocks.head()
# %%
inf = fred.get_series("IITTRLB", observation_start='01/01/2017', 
                      observation_end='01/01/2024', frequency='a')

stocks['Inc_Taxes'] = np.empty(stocks.shape[0])
for i in range(0, len(stocks)):
    x = 0
    try:
        while stocks.at[i, "End"] >= inf.index[x]:
            x+=1
        stocks.at[i,'Inc_Taxes'] = (inf.get(x-1) - inf.get(x-2))/inf.get(x-2)
    except IndexError:
        stocks.at[i,'Inc_Taxes'] = None
stocks.head()
# %%
#writer = pd.ExcelWriter('SEC_FRED_data.xlsx', engine='openpyxl', mode='w')
stocks.to_csv('SEC_FRED_data.csv', index=False)
# %%
thresh = pd.Timestamp('2018-01-01 00:00:00-04:00', tz='America/New_York')
print(thresh)
for i in range(0,len(stocks.index)):
    time = pd.Timestamp(stocks.at[i, 'End'], tz='America/New_York')
    if time > thresh:
        stocks.at[i, 'Inc_Taxes'] = 0.21
    else:
        stocks.at[i, 'Inc_Taxes'] = 0.35

# %%
stocks.to_csv('filler.csv', index=True)

# %%

#count all the null values in stocks and print out their row column and name
null_columns=stocks.columns[stocks.isnull().any()]
# %%
stocks.to_csv('data_branch2_v2.csv', index=False)

# %%
# save the dataframe stocks into a text file
stocks.to_csv('data_branch2_v2.csv', index=False)