#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
# %%
df = pd.read_csv('data_branch2_v4.csv')
# %%
#Count all NaN values in each column
df.isnull().sum()
# %%
df_2 = pd.read_csv('openBB_Data_v3.csv')
#%%
#create a dict of dataframes for each entry under the 'Name' column
start = 0
end = 0
df2_dict = {}
breaks = []
for i in range(0, len(df_2) - 1):
    if df_2.at[i, 'Name'] != df_2.at[i+1, 'Name']:
        end = i
        df2_dict[df_2.at[i-1, 'Name']] = (df_2[start:end+1])
        start = i+1
        breaks.append(i)
#%%
#Loop through df2_dict and add all values under the 'Name' column to a set, if the set has a length greater than 1, print it out
for key in df2_dict.keys():
    if len(set(df2_dict[key]['Name'])) > 1:
        print(key)
# %%
#reindex df_2 to make the column 'End' the index
for key in df2_dict.keys():
    df2_dict[key].index = df2_dict[key]['Date']

#%%
start = 0
end = 0
df_list = []
for i in range(0, len(df) - 1):
    if df.at[i, 'Name'] != df.at[i+1, 'Name']:
        end = i
        df_list.append(df[start:end+1])
        start = i+1
#%%
for i in range(0, len(df_list)):
    s= set(df_list[i]['Name'])
    if len(s) > 1:
        print(s)
#%%
#Loop through df_list and reindex every dataframe so that it starts at 0
for frame in df_list:
    frame.index = range(0, len(frame))

# %%
for i in range(0, len(df_list)):
    for j in range(0, len(df_list[i])):
        temp = df_list[i].at[j, 'End'][5:7] + "/" + df_list[i].at[j, 'End'][8:10] + "/" + df_list[i].at[j, 'End'][0:4]
        df_list[i].at[j, 'End'] = temp
# %%
# Fix the format of the 'End' column
# RIght now its mm/dd/yyyy, remove all extraneous zeros
# For example the date 07/01/2020 should be 7/1/2020
for frame in df_list:
    for i in range(0, len(frame)):
        if df.at[i, 'End'][0] == '0':
            df.at[i, 'End'] = df.at[i, 'End'][1:]
        if df.at[i, 'End'][3] == '0':
            df.at[i, 'End'] = df.at[i, 'End'][:3] + df.at[i, 'End'][4:]
#%%
#make the End column a datetime object
for i in range(0, len(df_list)):
    for j in range (0, len(df_list[i])):
        df_list[i].at[ j,'End'] = pd.to_datetime(df_list[i].at[ j,'End'])

#make the index of df_2 a datetime object
for key in df2_dict.keys():
    df2_dict[key].index = pd.to_datetime(df2_dict[key].index)

#%%
#Loop through every column and find the missing values

for i in range (0, len(df_list)):

#for i in range(0, 2):

    df_temp = df2_dict[df_list[i].at[0, 'Name']]

    for col in df.columns:
        for j in range(0, len(df_list[i])):

            if pd.isnull(df_list[i][col][j]):
                print("column: ", col ," index: " , str(j))
                try: 
                    final_val = df_temp[col][df_list[i]['End'][j]]
                    initial_val = df_temp[col][df_list[i]['End'][j-1]]

                    df_list[i].at[j,col] = (final_val - initial_val)/initial_val
                except:
                    df_list[i].at[j,col] = np.nan

# %%
#Loop through df_list and concatenate all the dataframes into one dataframe named final_df
final_df = pd.DataFrame()
for frame in df_list:
    final_df = pd.concat([final_df, frame])

# %%
#reindex final_df
final_df.index = range(0, len(final_df))
# %%
final_df.isnull().sum()

# %%
final_df.to_csv('final_df.csv')
# %%
