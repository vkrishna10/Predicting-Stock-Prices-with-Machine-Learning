#%%
import pandas as pd
import yfinance as yf
from openbb_terminal.sdk import openbb
import time
import numpy as np
import requests as rq
import json
# %%
df = pd.read_csv('openBB_Data_Version2.csv')
# %%
df
#%%
headers = {'User-Agent': "vkrishnaswamy489@gmail.com"}
tickers_cik = rq.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
tickers_cik =     pd.json_normalize(pd.json_normalize(tickers_cik.json(),\
max_level=0).values[0])
tickers_cik["cik_str"] = tickers_cik["cik_str"].astype(str).str.zfill(10)
tickers_cik.set_index("ticker",inplace=True)
tickers_cik.sort_index()
# %%
#creat a dict of tickers and ciks
cik_dict = {}
for i in range(0, len(tickers_cik.index)):
    cik_dict[tickers_cik.index[i]] = tickers_cik.at[tickers_cik.index[i], 'cik_str']
# %%
# creata a dict where the key value is the ticker and the value is the frequency of the ticker
freq_dict = {}
for i in range(0, len(df.index)):
    if df.at[i, 'Name'] in freq_dict:
        freq_dict[df.at[i, 'Name']] += 1
    else:
        freq_dict[df.at[i, 'Name']] = 1
#%%
response = rq.get('https://data.sec.gov/submissions/CIK0001045810.json', headers=headers)
#The format of the response is in json
#Inside the dictionary 'filing' is dictionary named 'recent'
#Turn the dictionary recent into a pandas dataframe named filings
#The keys of 'recent' are the columns of the dataframe
filings = pd.json_normalize(response.json()['filings']['recent'])

# %%
# All of the values in the dataframe filings are lists
# every column has one list as its value
# Give every index of the list its own row in a new dataframe
# For example if I have a dataframe with one column named 'a' and the value of that column is [1,2,3] and one column named 'b' and the value of that column is [4,5,6]
# Then the new dataframe will have two columns named 'a' and 'b'
# And the rows would be 1,4 2,5 3,6
# The new dataframe will have the same columns as filings
# The new dataframe will have the same number of rows as the sum of the lengths of the lists in filings
# The new dataframe will have the same values as filings, but with each list in filings turned into its own row
individual_filings = pd.DataFrame(columns=filings.columns)
for ind in range(0, len(filings.at[0, 'accessionNumber'])):
    new_row = {}
    for col in filings.columns:
        new_row[col] = filings.at[0, col][ind]
    individual_filings = individual_filings.append(new_row, ignore_index=True)
#%%
reports = pd.DataFrame(columns=filings.columns)
for i in range(0, len(individual_filings.index)):
    if individual_filings.at[i, 'form'] == '10-Q' or individual_filings.at[i, 'form'] == '10-K':
        reports.loc[len(reports)] = (individual_filings.loc[i])

# %%
#change the index of reports to be the 'reportDate' column
reports.set_index('reportDate', inplace=True)

# %%
df_prime = df[0:20]
#loop through df_prime and get the 'filingDate' for every data in the 'Date' column of df_prime
#The 'Date' column of df_prime is formatted mm/dd/yyyy
#The 'filingDate' column of reports is formatted yyyy-mm-dd
#So you will have to convert the 'Date' column of df_prime to yyyy-mm-dd before you can compare it to the 'filingDate' column of reports

for i in df_prime.index:
    #get the date from the 'Date' column of df_prime
    date = df_prime.at[i, 'Date']
    #convert the date to yyyy-mm-dd
    date = pd.to_datetime(date, format='%m/%d/%Y').strftime('%Y-%m-%d')
    #get the 'filingDate' from reports
    filingDate = reports.at[date, 'filingDate']
    #set the 'filingDate' in df_prime
    df_prime.at[i, 'filingDate'] = filingDate

# %%
x=0
#modify freq_dict so that it only contains the first 5 tickers

for name in freq_dict:
    print('Still going')
    
    url_str = 'https://data.sec.gov/submissions/CIK' + cik_dict[name] + '.json'
    response = rq.get(url_str, headers=headers)
    filings = pd.json_normalize(response.json()['filings']['recent'])

    individual_filings = pd.DataFrame(columns=filings.columns)
    for ind in range(0, len(filings.at[0, 'accessionNumber'])):
        new_row = {}
        for col in filings.columns:
            new_row[col] = filings.at[0, col][ind]
        individual_filings = individual_filings.append(new_row, ignore_index=True)

    reports = pd.DataFrame(columns=filings.columns)
    for i in range(0, len(individual_filings.index)):
        if individual_filings.at[i, 'form'] == '10-Q' or individual_filings.at[i, 'form'] == '10-K':
            reports.loc[len(reports)] = (individual_filings.loc[i])
    reports.set_index('reportDate', inplace=True)

    for i in range(0, freq_dict[name]):
        try:
            date = df.at[x, 'Date']
            #convert the date to yyyy-mm-dd
            date = pd.to_datetime(date, format='%m/%d/%Y').strftime('%Y-%m-%d')
            #get the 'filingDate' from reports
            filingDate = reports.at[date, 'filingDate']
            #set the 'filingDate' in df_prime
            df.at[x, 'filingDate'] = filingDate
            x+=1
        except Exception as e:
            #print(f"Error with {name} on {date}. Error: {e}")
            df.at[x, 'filingDate'] = np.nan
            x+=1


# %%
# count the number of nan values in the 'filingDate' column
a = df['filingDate'].isna().sum()
len(df['filingDate'])-a
# %%

#Loop through df by row and get all of the rows with nan values in the 'filingDate' column
# Take those rows out and store them in a new dataframe called df_nan
df_nan = pd.DataFrame(columns=df.columns)
for i in df.index:
    if pd.isna(df.at[i, 'filingDate']):
        df_nan.append(df.loc[i])
        df.drop(i, inplace=True)
#reset the index of df_nan
df.reset_index(inplace=True)

# %%
#write df to a new csv file
df.to_csv('openBB_Data_v3.csv')
# %%
#Using the yfinance library, get the stock price for every ticker in df one day before the 'filingDate' column value
#Store the stock price in a new column named 'initialPrice'
#The 'filingDate' column value is formatted yyyy-mm-dd
#The 'initialPrice' column value will be the stock price on the day before the 'filingDate' column value
#The 'initialPrice' column value will be a float
#The 'initialPrice' column value will be the closing price of the stock on the day before the 'filingDate' column value
x=0
for company in freq_dict:
    
        hist = yf.Ticker(company).history(period='1d', start='2017-01-01', end='2023-04-23', interval='1d')
        hist.index = hist.index.strftime('%Y-%m-%d')
        #make the indicies of hist into datetime objects
        hist.index = pd.to_datetime(hist.index)
        for i in range(0, freq_dict[company]):
            date = df.at[x, 'filingDate']
            date = pd.to_datetime(date, format='%Y-%m-%d').strftime('%Y-%m-%d')
            df.at[x, 'initialPrice'] = hist.at[date, 'Close']
            date_fin = (pd.to_datetime(date) + pd.DateOffset(days=2)).strftime('%Y-%m-%d')
            try:
                #Check if date_fin lands on a weekend, if it does, then add how many days it takes to get to the next monday
                if pd.to_datetime(date_fin).weekday() == 5:
                    date_fin = (pd.to_datetime(date_fin) + pd.DateOffset(days=2)).strftime('%Y-%m-%d')
                elif pd.to_datetime(date_fin).weekday() == 6:
                    date_fin = (pd.to_datetime(date_fin) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
                
                df.at[x, 'finalPrice'] = hist.at[date_fin, 'Close']
                x+=1
            except Exception as e:
                # set the final price to the closing price closest to the date_fin
                # hist[date_fin] will not be defined, so we will have to find the closest date to date_fin
                # that is defined in hist
                # we will do this by finding the difference between date_fin and all of the dates in hist
                # and then finding the minimum difference
                # the date in hist that has the minimum difference will be the date that we will use
                # to get the closing price
                # we will then set the final price to that closing price

                #create a list of the differences between date_fin and all of the dates in hist
                diff_list = []
                for date in hist.index:
                    diff_list.append(abs(pd.to_datetime(date_fin) - date))
                #find the minimum difference
                min_diff = min(diff_list)
                #find the index of the minimum difference
                min_diff_index = diff_list.index(min_diff)
                #find the date in hist that has the minimum difference
                min_diff_date = hist.index[min_diff_index]
                #set the final price to the closing price of the date in hist that has the minimum difference
                df.at[x, 'finalPrice'] = hist.at[min_diff_date, 'Close']


                x+=1
                print(f"Error with {company} on {date}. Error: {e}")

# %%
df.to_csv('openBB_Data_v3.csv')

# %%
