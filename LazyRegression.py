#%%
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import sklearn
# %%
data = pd.read_csv('final_df.csv')

# %%
X = data[['totalRevenue', 'totalAssets',
       'totalLiabilities', 'commonStockSharesOutstanding', 'GDP', 'Interest',
       'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
       'Unemployment_Rate', 'Recession_Probability', 'Corporate_Profits']]
# X = data[['totalRevenue', 'stockholderEquity', 'totalLiabilities', 'commonStockSharesOutstanding',
#        'Interest', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
#        'Unemployment_Rate']]

# X = data[['totalRevenue',  
#        'Interest', 'SP', 'Cons_Sent', 'Fin_Stress',
#        'Unemployment_Rate']]
#
# %%
y = data[['% delta']]
# %%
X, y =  shuffle(X, y, random_state=42)
#%%
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# %%
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
                     
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)
#%%
from sklearn.ensemble import ExtraTreesRegressor
tree = ExtraTreesRegressor()

# Run a 10 cross fold validation on tree using dataframes X and y
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree, X, y, cv=10)
scores

#print out the summary statistics for scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#%%
#Do the same thing as the cell above, except for the following models:
#['LGBMRegressor', 'HistGradientBoostingRegressor', 'XGBRegressor']
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

xgb = XGBRegressor()
lgbm = LGBMRegressor()
bag = BaggingRegressor()
hist = HistGradientBoostingRegressor()

# Run a 10 cross fold validation on the models using dataframes X and y
from sklearn.model_selection import cross_val_score
scores_xgb = cross_val_score(xgb, X, y, cv=10)
scores_lgbm = cross_val_score(lgbm, X, y, cv=10)
scores_bag = cross_val_score(bag, X, y, cv=10)
scores_hist = cross_val_score(hist, X, y, cv=10)
#%%
#print out the square root of 2
print("ExtraTreesRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), (scores.std() * 2)/len(scores)))
print("XGBRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_xgb.mean(), scores_xgb.std() * 2)/len(scores))
print("LGBMRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_lgbm.mean(), scores_lgbm.std() * 2))
print("BaggingRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_bag.mean(), scores_bag.std() * 2))
print("HistGradientBoostingRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_hist.mean(), scores_hist.std() * 2))

#%%
#Run a 10 cross fold validation for the following models: LGBMRegressor, SGDRegressor, RandomForestRegressor.  for adjusted r-squared and RMSE
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

sgd = SGDRegressor()
rf = RandomForestRegressor()
lgbm = LGBMRegressor()

# Run a 10 cross fold validation on the models using dataframes X and y
from sklearn.model_selection import cross_val_score
scores_sgd = cross_val_score(sgd, X, y, cv=10)
scores_rf = cross_val_score(rf, X, y, cv=10)
scores_lgbm = cross_val_score(lgbm, X, y, cv=10)

#print out the summary statistics for scores
print("SGDRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_sgd.mean(), scores_sgd.std() * 2))
print("RandomForestRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))
print("LGBMRegressor Accuracy: %0.2f (+/- %0.2f)" % (scores_lgbm.mean(), scores_lgbm.std() * 2))

#%%
# Plot the relation between '% delta' and 'totalRevenue' using a scatter plot, draw a line of best fit
# and print out the r-squared value
import seaborn as sns
sns.regplot(x='Asset/share', y='finalPrice', data=data)
plt.show()


#%%
# from sklearn.ensemble import BaggingRegressor
# bag = BaggingRegressor()
# bag.fit(X_train, y_train)
# y_pred = bag.predict(X_test)
# import sklearn.metrics as metrics
# r2 = metrics.r2_score(y_test,y_pred)
# r2
#%%
# Create a correlation matrix (with plots) to see if there is any multicollinearity
import seaborn as sns
import matplotlib.pyplot as plt
# matrix = X.corr().round(1)
# sns.heatmap(data=matrix, annot=True)
# plt.show()
# %%
# Create a correlation matrix to see which features are most correlated with the target variable
Xp = data[['totalRevenue', 'totalAssets',
       'totalLiabilities', 'commonStockSharesOutstanding', 'GDP', 'Interest',
       'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
       'Unemployment_Rate', 'Recession_Probability', 'Corporate_Profits', 'open',
       '% delta']]
corr_matrix = Xp.corr().round(1)
sns.heatmap(data=corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=np.triu(np.ones_like(corr_matrix, dtype=bool)))
plt.show()

# %%
plt.savefig('collinearities.png')
# %%
X = data[['totalRevenue', 'totalAssets',
       'totalLiabilities', 'commonStockSharesOutstanding', 'GDP', 'Interest',
       'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
        'open']]
y = data[['% delta']]
# %%
X, y =  shuffle(X, y, random_state=42)
#%%
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]
# %%
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
                     
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)

# %%
# %%
