
# coding: utf-8

# In[ ]:


#!pip install sklearn


# In[ ]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# # Read data

# In[ ]:


credit_df = pd.read_csv("C:\\Users\\yogesh\\Desktop\\credicardfraud.csv")


# In[ ]:


credit_df.shape


# In[ ]:


#credit_df.columns


# In[ ]:


credit_df.isnull().sum()


# In[ ]:


#credit_df.dtypes


# # Correlation with target variable

# In[ ]:


correlation = credit_df.corr()
print(correlation['Class'].sort_values(ascending = False))


# In[ ]:


# Target ie.class is highly positive correlated with V11, V4 and V2 and highly negative correlation with V1 ,V18,V7, V3, V16,V10,V12,V14 & V17 
# considering correlation >10%


# # Box plot for highly positive correlation of features with class

# In[ ]:


f, axes = plt.subplots(ncols = 3, figsize=(20,5))
# Bx plot for highly positive correlation of features with Class

sns.boxplot(x="Class", y="V11", data=credit_df, ax=axes[0])
axes[0].set_title('Positive Correlation between V11 & Class')

sns.boxplot(x="Class", y="V4", data=credit_df, ax=axes[1])
axes[1].set_title('Positive Correlation between V4 & Class')


sns.boxplot(x="Class", y="V2", data=credit_df,  ax=axes[2])
axes[2].set_title('Positive Correlation between V2 & Class')
plt.show()


# In[ ]:


sns.kdeplot(credit_df.V2[credit_df.Class == 0],label='0');
sns.kdeplot(credit_df.V2[credit_df.Class == 1],label='1');
#credit_df.groupby("Class").V2.plot(kind = 'kde')
plt.show()


# In[ ]:


# Above box plot shows no outliers for class 1 i.e being "Fraud" for variables V11 & V4. for V2 kde shows almost gussian distribution.


# # Box plot for highly negative correlation of features with class

# In[ ]:


f, axes = plt.subplots(ncols = 5, figsize=(20,5))
# Bx plot for highly negative correlation of features with Class

sns.boxplot(x="Class", y="V17", data=credit_df, ax=axes[0])
axes[0].set_title('Negative Correlation between V17 & Class')

sns.boxplot(x="Class", y="V14", data=credit_df, ax=axes[1])
axes[1].set_title('Negative Correlation between V14 & Class')


sns.boxplot(x="Class", y="V12", data=credit_df,  ax=axes[2])
axes[2].set_title('Negative Correlation between V12 & Class')


sns.boxplot(x="Class", y="V10", data=credit_df,  ax=axes[3])
axes[3].set_title('Negative Correlation between V10 & Class')

sns.boxplot(x="Class", y="V16", data=credit_df,  ax=axes[4])
axes[4].set_title('Negative Correlation between V16 & Class')

plt.show()


# In[ ]:


# Above box plot shows extreme outliers for variables V14,V12 and V10 of class 1 i.e being "Fraud".


# # Remove extreme outliers for V14, V12 and V10 variables of class 1.

# In[ ]:


## Removing outliers for features V14 as highly negative correlated with class 
v14_fraud = credit_df['V14'].loc[credit_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 3
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = credit_df.drop(credit_df[(credit_df['V14'] > v14_upper) | (credit_df['V14'] < v14_lower)].index)
print(new_df.shape)
print('****' * 30)



## Removing outliers for features V12 as highly negative correlated with class 
v12_fraud = credit_df['V12'].loc[credit_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v12_iqr = q75 - q25
print('iqr: {}'.format(v12_iqr))

v12_cut_off = v12_iqr * 3
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('Cut Off: {}'.format(v12_cut_off))
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('Feature V1 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = credit_df.drop(credit_df[(credit_df['V12'] > v12_upper) | (credit_df['V12'] < v12_lower)].index)
print(new_df.shape)
print('****' * 30)

## Removing outliers for features V10 as highly negative correlated with class 
v10_fraud = credit_df['V10'].loc[credit_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v10_iqr = q75 - q25
print('iqr: {}'.format(v12_iqr))

v10_cut_off = v10_iqr * 3
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('Cut Off: {}'.format(v10_cut_off))
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))

outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

new_df = credit_df.drop(credit_df[(credit_df['V1'] > v10_upper) | (credit_df['V10'] < v10_lower)].index)
print(new_df.shape)
print('****' * 30)


# # Box plot after reduction of outliers

# In[ ]:


f, axes = plt.subplots(ncols = 3, figsize=(20,5))
# Bx plot for highly negative correlation of features with Class

sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[0])
axes[0].set_title('Reduction of extreme outliers for V14 feature')


sns.boxplot(x="Class", y="V12", data=new_df,  ax=axes[1])
axes[1].set_title('Reduction of extreme outliers for V12 feature')


sns.boxplot(x="Class", y="V10", data=new_df,  ax=axes[2])
axes[2].set_title('Reduction of extreme outliers for V10 feature')


plt.show()


# # Again check the correlation of features with Class

# In[ ]:


correlation = new_df.corr()
print(correlation['Class'].sort_values(ascending = False))


# In[ ]:


# there is little change in correlation with target because of outliers reduction.


# In[ ]:


new_df.shape


# # Distribution of target

# In[ ]:


f, ax = plt.subplots(figsize = (10, 6))
ax = sns.countplot(x = "Class", data= new_df)
ax.set_title("Distribution of Class", size = 15)
plt.show()


# In[ ]:


# It is highly skewed since perentage of being fraud is 0.172%. Using SMOTE technique balance the data


# # SMOTE

# In[ ]:


X = new_df.loc[:, new_df.columns != 'Class']
y = new_df.loc[:, new_df.columns == 'Class']
from imblearn.over_sampling import SMOTE
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['Class'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of non Fraud in oversampled data",len(os_data_y[os_data_y['Class']==0]))
print("Number of Fraud",len(os_data_y[os_data_y['Class']==1]))
print("Proportion of non Fraud data in oversampled data is ",len(os_data_y[os_data_y['Class']==0])/len(os_data_X))
print("Proportion of Fraud data in oversampled data is ",len(os_data_y[os_data_y['Class']==1])/len(os_data_X))


# In[ ]:


print(os_data_X.shape)
print(os_data_y.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# # Considering features have correlation with target to build model

# In[ ]:


cols=["V11", "V4", "V2","V21","V5","V1","V9","V18","V7","V3","V16","V10","V12","V14","V17"] 
X=os_data_X[cols]
y=os_data_y['Class']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[ ]:


# the result is telling that 59545+57136 are correctly predicted while 2201+529 are wrongly predicted.


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


# Model has a precision of 0.98 i.e when it predicts person is non fraud, it is correct 98% of the time.
# Model has a recall of 0.98 , it correctly identifies 98% of all non fraud persons.


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

