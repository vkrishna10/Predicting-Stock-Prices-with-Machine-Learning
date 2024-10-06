#%%
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# %%
data = pd.read_csv('final_df.csv')

# %%
X = X = data[['totalRevenue', 'totalAssets',
       'totalLiabilities', 'commonStockSharesOutstanding', 'GDP', 'Interest',
       'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand']]
# X = data[['totalRevenue', 'stockholderEquity', 'totalLiabilities', 'commonStockSharesOutstanding',
#        'Interest', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
#        'Unemployment_Rate']]

# X = data[['totalRevenue',  
#        'Interest', 'SP', 'Cons_Sent', 'Fin_Stress',
#        'Unemployment_Rate']]
#
# %%
y = data[['class']]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)
#%%
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)
#%%
from itertools import combinations
classifs = [[]]
for i in range(9, 12):
    for combo in combinations(X.columns, (i)):

       X_temp = data[list(combo)]
       y = data[['class']]

       X_train, X_test, y_train, y_test = train_test_split(X_temp, 
                                                           y,test_size=.2,random_state =123)

       
       reg = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
       models, predictions = reg.fit(X_train, X_test, y_train, y_test)
       print(models.iloc[0])
       classifs.append([combo, models.iloc[0]])
#%%
a = classifs[1:]

for index,val in enumerate(a):
       comp_min = 0
       min_index = -1
       for i in range(index, len(a),1):
              if a[i][1]['Accuracy'] > comp_min:
                     comp_min = a[i][1]['Accuracy']
                     min = a[i]
                     min_index = i
       temp = a[index]
       a[index] = min
       a[min_index] = temp
  
a
#%%
#calculate the accuracy of a Quadratic discriminant analysis model
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X = data[['totalRevenue', 'totalAssets',
         'totalLiabilities', 'commonStockSharesOutstanding', 'GDP', 'Interest',
            'Inflation', 'SP', 'Cons_Sent', 'Fin_Stress', 'Cash_On_Hand',
            'Unemployment_Rate', 'Recession_Probability', 'Corporate_Profits']]
y = data[['class']]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.2,random_state =123)

clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#plot a confusion matrix for clf
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Create the Confusion Matrix

y_pred = clf.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
class_names = [0,1] # Our diagnosis categories

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create a heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')



#%%
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import NuSVC
import numpy as np

#Run a 10 fold crossvalidation on all of the models

NC = NearestCentroid()
QDA = QuadraticDiscriminantAnalysis()
ADC = AdaBoostClassifier()
svc = NuSVC()

scores_nc = cross_val_score(NC, X, y, cv=10)
print('Model 1 calculated')
scores_qda = cross_val_score(QDA, X, y, cv=10)
print('Model 2 calculated')
scores_adc = cross_val_score(ADC, X, y, cv=10)
print('Model 3 calculated')
scores_svc = cross_val_score(svc, X, y, cv=10)
print('Model 4 calculated')

print("""NearestCentroid Accuracy: %0.2f (+/- %0.2f)""" 
% (scores_nc.mean(), ((scores_nc.std() * 2.262)/np.sqrt(len(scores_nc))) ) )

print("""QuadraticDiscriminatAnalysis Accuracy: %0.2f (+/- %0.2f)"""
% (scores_qda.mean(), ((scores_qda.std() * 2.262)/np.sqrt(len(scores_qda))) ) )

print("""AdaBoost Accuracy: %0.2f (+/- %0.2f)"""
% (scores_adc.mean(), ((scores_adc.std() * 2.262)/np.sqrt(len(scores_adc))) ) )

print("""NuSVC Accuracy: %0.2f (+/- %0.2f)"""
% (scores_svc.mean(), ((scores_svc.std() * 2.262)/np.sqrt(len(scores_svc))) ) )

#%%
#create a confusion matrix for the passive aggressive classifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Create the Confusion Matrix
# y_test = dataframe['diagnosis']
y_pred = ADC.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
class_names = [0,1] # Our diagnosis categories

fig, ax = plt.subplots()
# Setting up and visualizing the plot (do not worry about the code below!)
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') 
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual increase')
plt.xlabel('Predicted increase')
plt.show()
#%%
# RUn a 10 fold cross validation for the CalibratedClassifierCV, LogisticRegression, and RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
import numpy as np

#Run a 10 fold crossvalidation on all of the models

CCV = CalibratedClassifierCV()
LR = LogisticRegression()
RC = RidgeClassifier()

scores_ccv = cross_val_score(CCV, X, y, cv=10)
print('Model 1 calculated')
scores_lr = cross_val_score(LR, X, y, cv=10)
print('Model 2 calculated')
scores_rc = cross_val_score(RC, X, y, cv=10)
print('Model 3 calculated')

print("""CalibratedClassifierCV Accuracy: %0.2f (+/- %0.2f)"""
% (scores_ccv.mean(), ((scores_ccv.std() * 2.262)/np.sqrt(len(scores_ccv))) ) )

print("""LogisticRegression Accuracy: %0.2f (+/- %0.2f)"""
% (scores_lr.mean(), ((scores_lr.std() * 2.262)/np.sqrt(len(scores_lr))) ) )

print("""RidgeClassifier Accuracy: %0.2f (+/- %0.2f)"""
% (scores_rc.mean(), ((scores_rc.std() * 2.262)/np.sqrt(len(scores_rc))) ) )

#%%
# RUn a 10 fold cross validation for the CalibratedClassifierCV, LogisticRegression, and RidgeClassifier
#Use F1 score as the scoring metric
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
import numpy as np

#Run a 10 fold crossvalidation on all of the models

CCV = CalibratedClassifierCV()
LR = LogisticRegression()
RC = RidgeClassifier()

scores_ccv = cross_val_score(CCV, X, y, cv=10, scoring='f1')
print('Model 1 calculated')
scores_lr = cross_val_score(LR, X, y, cv=10, scoring='f1')
print('Model 2 calculated')
scores_rc = cross_val_score(RC, X, y, cv=10, scoring='f1')
print('Model 3 calculated')

print("""CalibratedClassifierCV Accuracy: %0.2f (+/- %0.2f)"""
% (scores_ccv.mean(), ((scores_ccv.std() * 2.262)/np.sqrt(len(scores_ccv))) ) )

print("""LogisticRegression Accuracy: %0.2f (+/- %0.2f)"""
% (scores_lr.mean(), ((scores_lr.std() * 2.262)/np.sqrt(len(scores_lr))) ) )

print("""RidgeClassifier Accuracy: %0.2f (+/- %0.2f)"""
% (scores_rc.mean(), ((scores_rc.std() * 2.262)/np.sqrt(len(scores_rc))) ) )
#%%
#create a confusion matrix for CCV
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# Create the Confusion Matrix
# y_test = dataframe['diagnosis']
CCV.fit(X_train, y_train)
y_pred = CCV.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
class_names = [0,1] # Our diagnosis categories

fig, ax = plt.subplots()
# Setting up and visualizing the plot (do not worry about the code below!)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual increase')
plt.xlabel('Predicted increase')
plt.show()

#%%
#                               Accuracy  Balanced Accuracy  ROC AUC  F1 Score  \
# Model                                                                           
# NearestCentroid                    0.67               0.66     0.66      0.67   
# NuSVC                              0.68               0.66     0.66      0.67   
# QuadraticDiscriminantAnalysis      0.66               0.65     0.65      0.65   
# AdaBoostClassifier                 0.66               0.64     0.64      0.65   
# SVC                                0.67               0.64     0.64      0.65   