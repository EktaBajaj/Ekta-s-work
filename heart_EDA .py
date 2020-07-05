
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


os.getcwd()


# # Read data

# In[ ]:


heart_df = pd.read_csv("C:\\Users\\yogesh\\Desktop\\Assignment_1_Intermediate\\Assignment_1_Intermediate\\Heart.csv")


# In[ ]:


heart_df.shape


# In[ ]:


heart_df.head(10)


# In[ ]:


heart_df.isnull().any()


# In[ ]:


heart_df.dtypes


# In[ ]:


heart_df.describe()


# In[ ]:


heart_df.columns


# # How many are suffering from heart disease? Also plot the stats.

# In[ ]:


# target variable has value 0 and 1. 0 indicates absence of heart disease and 1 indicates presence of heart disease


# In[ ]:


heart_df['target'].value_counts()


# In[ ]:


# Answer: 138 persons who do not suffering from heart disease and 165 persons are suffering from heart disease.


# In[ ]:


f, ax = plt.subplots(figsize = (10, 8))
ax = sns.countplot(x = "target", data= heart_df)
ax.set_title("heart disease", size = 15)
plt.show()


# # How many males and females have heart disease out of total?

# In[ ]:


heart_df.groupby('sex')['target'].value_counts()


# In[ ]:


f, ax = plt.subplots(figsize = (10, 8))
ax = sns.countplot(x = "target", data= heart_df, hue = 'sex')
ax.set_title("People suffering from heart disease")
plt.show()


# In[ ]:


# Answer :0 indicates female and 1 indicates male. out of 96 females 72 have heart disease (75%) and out of 207 males 93 have heart disease(44%).


# # Visualize frequency distribution of the thalach variable and find what's the heart rate and heart disease relation?

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.stripplot(x="target", y="thalach", data=heart_df)
ax.set_title("Frequency distribution of thalach with respect to target", size = 15)
plt.show()


# In[ ]:


# Answer: Persons suffering from heart disease have higher heart rate as compared to persons not suffering rom heart disease.


# # Find correlation matrix for all the variables with target.

# In[ ]:


heart_df.corr()


# In[ ]:


# Answer:
# features cp, thalach, slope and restecg are positively correlated related with target. 
# features age, sex, trestbps , exang, oldpeak, ca and thal are negatively correlated with target.
# features chol and fps have no correlaton with target.


# # Find Mean,Min & Max of age and plot its distribution.

# In[ ]:


heart_df['age'].describe()


# In[ ]:


# Answer: mean value of age is 54.36, min value is 29 and max value of age is 77.


# In[ ]:


f, ax = plt.subplots(figsize=(10,8))
x = heart_df['age']
ax = sns.distplot(x, bins=10)
ax.set_title("Distribution of age", size = 15)
plt.show()


# ## Age and its relation to heart disease. Are young people more prone to heart disease?

# In[ ]:


f, ax = plt.subplots(figsize=(10, 8))
sns.stripplot(x="target", y="age", data=heart_df)
ax.set_title("Frequency distribution of age with respect to target", size = 15)
plt.show()


# In[ ]:


# Answer : No conlusion that young people are suffering from heart disease as 3 data points showed heart disease below 35 age.


# ## Plot chest pain type pie chart.

# In[ ]:


p = heart_df['cp'].value_counts()
labels = ['0', '1', '2', '3']
plt.pie(p, labels= labels)
plt.show()


# In[ ]:


# Answer : chest type 0 has max counts follwoed by chest type 1 ,2 and 3.


# # What is the max heart rate achieved in non heart disease patients?

# In[ ]:


non_heart_dis = heart_df[heart_df['target'] == 0]


# In[ ]:


non_heart_dis.shape


# In[ ]:


max(non_heart_dis['thalach'])


# In[ ]:


#Answer: max heart rate achiever in non heart disease patients is 195.

