#%%
import pandas as pd
import numpy as np
# %%
df = pd.read_csv('data_branch2_v2.csv', header=0)
# %%
df
# %%
df.at[0, 'totalRevenue']
# %%
#Chnage the numbers from format 1.234B t0 (1.234*10^9)
def convert_to_num(x):
    if x[-1] == 'M':
        return float(x[:-1])*10**6
    elif x[-1] == 'B':
        return float(x[:-1])*10**9
    elif x[-1] == 'T' or x[-1] == 'K':
        return float(x[:-1])*10**12
    else:
        return float(x)
# %%
#Loop through the totalRevenue, costOfGoodsAndServicesSold, researchAndDevelopment, totalAssets, totalLiabilities, inventory, and commonStockSharesOutstanding columns and convert the numbers to the correct format
for i in range(0, len(df)):
    # df.at[i, 'totalRevenue'] = convert_to_num(str(df.at[i, 'totalRevenue']))
    df.at[i, 'costofGoodsAndServicesSold'] = convert_to_num(str(df.at[i, 'costofGoodsAndServicesSold']))
    # df.at[i, 'researchAndDevelopment'] = convert_to_num(str(df.at[i, 'researchAndDevelopment']))
    # df.at[i, 'totalAssets'] = convert_to_num(str(df.at[i, 'totalAssets']))
    # df.at[i, 'totalLiabilities'] = convert_to_num(str(df.at[i, 'totalLiabilities']))
    # df.at[i, 'inventory'] = convert_to_num(str(df.at[i, 'inventory']))
    # df.at[i, 'commonStockSharesOutstanding'] = convert_to_num(str(df.at[i, 'commonStockSharesOutstanding']))
# %%
df.to_csv('data_branch2_v2.csv')
# %%
df = pd.read_csv('data_branch2_v2.csv', header=0)
# %%
df
# drop the column 'Unnamed: 0'
df.drop(columns=['Unnamed: 0'], inplace=True)
# %%
frames = []
start = 0
end = 0
for i in range(0, len(df)-1):
    if(df.at[i, 'Name'] != df.at[i+1, 'Name']):
        end = i
        frames.append(df[start:end+1])
        start = i+1
frames

#%%
#copy the contents of frames into a new list called frames_copy
frames_copy = []
for frame in frames:
    frames_copy.append(frame.copy())

#%%
#Loop through frames and reindex each frame
for frame in frames:
    frame.reset_index(inplace=True)
    frame.drop(columns=['index'], inplace=True)

#%%

for frame in frames:
    for column in frame.columns[2:9]:
        copy = frame.at[0, column]
        for i in range (1, len(frame)):
            temp = (frame.at[i, column] - copy)/copy
            copy = frame.at[i, column]
            frame.at[i, column] = temp
    frame.drop(index=0, inplace=True)

# %%
#Loop through frames and concat the dataframes into one dataframe named df
df = pd.concat(frames)
# %%
#reindex df
df.reset_index(inplace=True)

# %%
df.to_csv('data_branch2_v3.csv')
# %%
