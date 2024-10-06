#%%
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
# %%
stocks= pd.read_csv('ALL_DATA.csv', header=0)
stocks['End']=pd.to_datetime(stocks['End'], format='mixed', dayfirst=False, yearfirst=False)
# %%
stocks = stocks.drop('Unnamed: 0'	, axis=1)
# %%
Corp_prof = stocks['Corporate_Profits'][0:20]
# %%
Corp_prof = Corp_prof.drop(0, axis=0)
# %%
X= pd.DataFrame(list(range(0,19,1)))
# %%
y= pd.DataFrame(Corp_prof)
# %%
lin = LinearRegression()
lin.fit(X,y)

# %%
new_input = np.array([[-2]])

# Predict the target variable for your specific input
predicted_value = lin.predict(new_input.reshape(1, -1))

print(predicted_value)
# %%
import matplotlib.pyplot as plt

y_pred = lin.predict(X)
plt.plot(X, y_pred, color='red')

plt.scatter(X, y)
plt.xlabel('Index') # set the labels of the x and y axes
plt.ylabel('Corporate Profits')
plt.show()

# %%
Unemp = stocks['Unemployment_Rate'][0:20]
Unemp.isna().sum()
# %%
X= pd.DataFrame(list(range(0,20,1)))
y= pd.DataFrame(Unemp)
# %%
lin = LinearRegression()
lin.fit(X,y)
for i in range(-5,2,1):
    new_input = np.array([[i]])

    # Predict the target variable for your specific input
    predicted_value = lin.predict(new_input.reshape(1, -1))

    print(i, predicted_value, sep = '\t')

# %%
nan_indices = stocks.index[stocks['Unemployment_Rate'].isna()]
print(nan_indices)
# %%


#%%
Rec = stocks['Recession_probability']
print(Rec.isna().sum())
Rec = stocks['Recession_probability'][0:18]
# %%
X= pd.DataFrame(list(range(0,18,1)))
y= pd.DataFrame(Rec)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Recession_probability'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Recession_probability'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Recession_probability'] = None
        print(e)
# %%
stocks['Recession_probability'].to_csv('inter.csv', index=True, mode='a')

# %%


#%%
COH = stocks['Cash_On_Hand']
print(COH.isna().sum())
COH = stocks['Cash_On_Hand'][1:20]
# %%
X= pd.DataFrame(list(range(1,20,1)))
y= pd.DataFrame(COH)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Cash_On_Hand'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Cash_On_Hand'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Cash_On_Hand'] = None
        print(e)
# %%
stocks['Cash_On_Hand'].to_csv('inter.csv', index=True, mode='w')

# %%
FS = stocks['Fin_Stress']
print(FS.isna().sum())
FS = stocks['Fin_Stress'][0:17]
# %%
X= pd.DataFrame(list(range(0,17,1)))
y= pd.DataFrame(FS)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Fin_Stress'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Fin_Stress'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Fin_Stress'] = None
        print(e)
# %%
stocks['Fin_Stress'].to_csv('inter.csv', index=True, mode='w')

# %%

# %%
CS = stocks['Cons_Sent']
print(CS.isna().sum())
CS = stocks['Cons_Sent'][0:17]
# %%
X= pd.DataFrame(list(range(0,17,1)))
y= pd.DataFrame(CS)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Cons_Sent'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Cons_Sent'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Cons_Sent'] = None
        print(e)
# %%
stocks['Cons_Sent'].to_csv('inter.csv', index=True, mode='w')

# %%

#%%
SP = stocks['SP']
print(SP.isna().sum())
SP = stocks['SP'][0:17]
# %%
X= pd.DataFrame(list(range(0,17,1)))
y= pd.DataFrame(SP)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['SP'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'SP'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'SP'] = None
        print(e)
# %%
stocks['SP'].to_csv('inter.csv', index=True, mode='w')
# %%


#%%
Inf = stocks['Inflation']
print(Inf.isna().sum())
Inf = stocks['Inflation'][0:20]
# %%
X= pd.DataFrame(list(range(0,20,1)))
y= pd.DataFrame(Inf)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Inflation'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Inflation'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Inflation'] = None
        print(e)
# %%
stocks['Inflation'].to_csv('inter.csv', index=True, mode='w')
# %%


#%%
Int = stocks['Interest']
print(Int.isna().sum())
Int = stocks['Interest'][0:17]
# %%
X= pd.DataFrame(list(range(0,17,1)))
y= pd.DataFrame(Int)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['Interest'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'Interest'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'Interest'] = None
        print(e)
# %%
stocks['Interest'].to_csv('inter.csv', index=True, mode='w')
# %%

#%%
GDP = stocks['GDP']
print(GDP.isna().sum())
GDP = stocks['GDP'][0:20]
# %%
X = pd.DataFrame(list(range(0,20,1)))
y = pd.DataFrame(GDP)
# %%
lin = LinearRegression()
lin.fit(X,y)
df = pd.DataFrame(columns=['Date', 'Value'])
prev_date = stocks.at[0, 'End']
for i in range(-10,29,1):
    new_row = {
        'Date': prev_date - (i * pd.DateOffset(months=3))
    }

    new_row['Value'] = lin.predict(np.array([[i]]).reshape(1, -1))
    df.loc[len(df)] = new_row
    #prev_date = new_row['Date']
    # print(i, predicted_value, sep = '\t')
df

# %%
nan_indices = stocks.index[stocks['GDP'].isna()]
print(nan_indices)

# %%
for index in nan_indices:
    x = 0
    try: 
        while stocks.at[index, "End"] <= df.at[x, 'Date']:
            x+=1
        stocks.at[index,'GDP'] = df.at[x-1, 'Value']
    except Exception as e:
        stocks.at[index,'GDP'] = None
        print(e)
# %%
stocks['GDP'].to_csv('inter.csv', index=True, mode='w')

# %%
stocks['DividendPayments'] = stocks['DividendPayments'].fillna(0)
# %%
stocks['DividendPayments'].to_csv('inter.csv', index=True, mode='w')
# %%
df = pd.read_csv('openBB_Data_Version2.csv')
# %%
# write code that fills in missing values in the 'GDP' column by running a linear regression on the data
#