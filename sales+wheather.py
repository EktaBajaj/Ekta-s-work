#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


## Read data
full_data = pd.read_csv('C:\\Users\\Ekta\\weather+sales\\full_data.csv')


# In[4]:


## Dimension of data
full_data.shape


# In[5]:


## Check for null values
full_data.isna().sum()


# In[15]:


#full_data.columns


# In[6]:


## select desired columns
data_sub = full_data[['id','city','Region', 'Territory','District','Product','Pack', 'mpoints', 'X.OfCartons','Value',
       'SaleVolume.kl.Tons.','Month', 'Year','season','max_precipitation_rate', 'max_temperature', 'min_temperature',
       'max_dew_point_temperature', 'min_dew_point_temperature','avg_dew_point_temperature', 'total_precipitation', 'max_wind_speed',
       'min_wind_speed', 'avg_wind_speed', 'avg_wind_direction','max_wind_gust', 'max_relative_humidity', 'min_relative_humidity',
       'avg_relative_humidity', 'total_downward_solar_radiation','max_downward_solar_radiation', 'total_net_solar_radiation',
       'max_net_solar_radiation', 'max_atmospheric_pressure','min_atmospheric_pressure', 'avg_atmospheric_pressure',
       'avg_total_cloud_cover', 'avg_snow_depth', 'avg_snow_density','max_soil_temperature_level_1', 'max_soil_temperature_level_2',
       'max_soil_temperature_level_3', 'max_soil_temperature_level_4','min_soil_temperature_level_1', 'min_soil_temperature_level_2',
       'min_soil_temperature_level_3', 'min_soil_temperature_level_4','avg_soil_temperature_level_1', 'avg_soil_temperature_level_2',
       'avg_soil_temperature_level_3', 'avg_soil_temperature_level_4','avg_soil_moisture_level_1', 'avg_soil_moisture_level_2',
       'avg_soil_moisture_level_3', 'avg_soil_moisture_level_4']]


# In[6]:


## to decide the bins, find range
print(data_sub['SaleVolume.kl.Tons.'].min())
print(data_sub['SaleVolume.kl.Tons.'].max())


# In[10]:


## distribution of sales
sns.distplot(data_sub['SaleVolume.kl.Tons.'], kde=True,bins=int(24/5), color='blue', hist_kws={'edgecolor':'black'})


# In[70]:


## distribution of sales for different region
import matplotlib.pyplot as plt
#year = [2016,2017,2018,2019]
region = ['SUMATERA UTARA','SULAWESI','KALIMANTAN','JAWA BARAT','SUMATERA TENGAH','SUMATERA SELATAN','JAWA TENGAH',
     'JAWA TIMUR','NUSA TENGGARA DAN PAPUA']
for i in region:
    data_sub_year = data_sub[data_sub['Region'] == i]
    fig, ax =plt.subplots()
    #layout(5,1)
    plt.title(i)
    #print(data_sub_year.head())
    sns.distplot(data_sub_year['SaleVolume.kl.Tons.'], bins=int(24/5), color='blue')
fig.show()
    


# In[68]:


## to see the distribution of all numerical columns

#taking only the numeric columns from the dataframe.
numeric_features=[x for x in data_sub.columns if data_sub[x].dtype!="object"]

for i in data_sub[numeric_features].columns:
    plt.figure(figsize=(12,5))
    plt.title(i)
    sns.distplot(data_sub[i], bins = 5)


# In[16]:


# Boxplot
#data_sub_long = pd.melt(data_sub, "Region", var_name="Year", value_name="SaleVolume.kl.Tons.")
plt.figure(figsize=(12,8))
ax = sns.boxplot(x="Region", hue="Year", y="SaleVolume.kl.Tons.", data=data_sub, palette="Set3")
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)


# In[20]:


y = data_sub.groupby(['Year']).sum()
y = y['SaleVolume.kl.Tons.']
x = y.index.astype(int)

plt.figure(figsize=(12,8))
ax = sns.barplot(y = y, x = x)
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_xticklabels(labels = x, fontsize=12, rotation=50)
ax.set_ylabel(ylabel='Sale Volume', fontsize=16)
ax.set_title(label='Sale Volume in (Kilo Tons) in different Year', fontsize=20)
plt.show();


# In[21]:


table = data_sub.pivot_table('SaleVolume.kl.Tons.', index='Product', columns='Year', aggfunc='sum')
product = table.idxmax()
sales = table.max()
years = table.columns.astype(int)
data = pd.concat([product, sales], axis=1)
data.columns = ['Product', 'SaleVolume.kl.Tons.']

plt.figure(figsize=(12,8))
ax = sns.pointplot(y = 'SaleVolume.kl.Tons.', x = years, hue='Product', data=data, size=15)
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_ylabel(ylabel='Sale Volume', fontsize=16)
ax.set_title(label='Highest Product sale in (Kilo Tons) Per Year', fontsize=20)
ax.set_xticklabels(labels = years, fontsize=12, rotation=50)
plt.show();


# In[22]:


table = data_sub.pivot_table('SaleVolume.kl.Tons.', index='Region', columns='Year', aggfunc='sum')
Region = table.idxmax()
sales = table.max()
years = table.columns.astype(int)
data = pd.concat([Region, sales], axis=1)
data.columns = ['Region', 'SaleVolume.kl.Tons.']

plt.figure(figsize=(12,8))
ax = sns.pointplot(y = 'SaleVolume.kl.Tons.', x = years, hue='Region', data=data, size=15)
ax.set_xlabel(xlabel='Year', fontsize=16)
ax.set_ylabel(ylabel='Sale Volume', fontsize=16)
ax.set_title(label='Highest Sale (Kilo tons) in a Region for different years', fontsize=20)
ax.set_xticklabels(labels = years, fontsize=12, rotation=50)
plt.show();


# In[26]:


data = data_sub.groupby(['Region','season']).sum()['SaleVolume.kl.Tons.']
data = pd.DataFrame(data.sort_values(ascending=False))[0:10]
Region = data.index
Season = data.index
#Year = data.index
#Product = data.index
data.columns = ['SaleVolume.kl.Tons.']

colors = sns.color_palette("cool", len(data))
plt.figure(figsize=(12,8))
ax = sns.barplot(y = Region , x = 'SaleVolume.kl.Tons.', data=data, orient='h', palette=colors, hue_order=(Season))
ax.set_xlabel(xlabel='Sale Volume in Kilo Tons', fontsize=16)
ax.set_ylabel(ylabel='Region', fontsize=16)
ax.set_title(label='Top 10 Regions having high sales', fontsize=20)
plt.show();


# In[27]:


data = data_sub.groupby(['Region','season','Year','Product']).sum()['SaleVolume.kl.Tons.']
data = pd.DataFrame(data.sort_values(ascending=False))[0:10]
Region = data.index
Season = data.index
Year = data.index
Product = data.index
data.columns = ['SaleVolume.kl.Tons.']

colors = sns.color_palette("cool", len(data))
plt.figure(figsize=(12,8))
ax = sns.barplot(y = Region , x = 'SaleVolume.kl.Tons.', data=data, orient='h', palette=colors, hue_order=(Season, Year, Product))
ax.set_xlabel(xlabel='Sale Volume in Kilo Tons', fontsize=16)
ax.set_ylabel(ylabel='Region', fontsize=16)
ax.set_title(label='Top 10 Regions having high sales', fontsize=20)
plt.show();


# In[28]:


#data_sub.Product.unique()


# In[93]:


## subset the data for top performing regions for dry season and see the releationship among sales vs weather variables
data_sub_regions_top = data_sub[(data_sub['Region'] =='JAWA TIMUR') & (data_sub['Product']=='ANTRACOL 70WP')&
                                (data_sub['season']=='dry') & (data_sub['Year'] == 2017)]


# In[7]:


data_sub1 = data_sub[['SaleVolume.kl.Tons.','Region','X.OfCartons','max_precipitation_rate', 'max_temperature', 'min_temperature',
       'max_dew_point_temperature', 'min_dew_point_temperature','avg_dew_point_temperature', 'total_precipitation', 'max_wind_speed',
       'min_wind_speed', 'avg_wind_speed', 'avg_wind_direction','max_wind_gust', 'max_relative_humidity', 'min_relative_humidity',
       'avg_relative_humidity', 'total_downward_solar_radiation','max_downward_solar_radiation', 'total_net_solar_radiation',
       'max_net_solar_radiation', 'max_atmospheric_pressure','min_atmospheric_pressure', 'avg_atmospheric_pressure',
       'avg_total_cloud_cover','avg_snow_density','max_soil_temperature_level_1', 'max_soil_temperature_level_2',
       'max_soil_temperature_level_3', 'max_soil_temperature_level_4','min_soil_temperature_level_1', 'min_soil_temperature_level_2',
       'min_soil_temperature_level_3', 'min_soil_temperature_level_4','avg_soil_temperature_level_1', 'avg_soil_temperature_level_2',
       'avg_soil_temperature_level_3', 'avg_soil_temperature_level_4','avg_soil_moisture_level_1', 'avg_soil_moisture_level_2',
       'avg_soil_moisture_level_3', 'avg_soil_moisture_level_4']]


# In[8]:


#df[df.columns[1:]].corr()['LoanAmount'][:]
#data_sub1[data_sub1.columns[:]].corr()['Value'][:]


# In[95]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,15))
matrix = np.triu(data_sub1.corr())
sns.heatmap(data_sub1.corr(),annot=True ,cbar_kws={'orientation':'horizontal'}, mask=matrix)


# In[42]:


import matplotlib.pyplot as plt
#year = [2016,2017,2018,2019]
region = ['SUMATERA UTARA','SULAWESI','KALIMANTAN','JAWA BARAT','SUMATERA TENGAH','SUMATERA SELATAN','JAWA TENGAH',
     'JAWA TIMUR','NUSA TENGGARA DAN PAPUA']

for i in region:
    data_sub_year = data_sub1[data_sub1['Region'] == i]
    fig, ax =plt.subplots()
    plt.figure(figsize=(20,15))
    #df[df.columns[1:]].corr()['LoanAmount'][:]
    #layout(5,1)
    plt.title(i)
    print(i)
    correlation = data_sub_year.corr()
    corr = correlation[(correlation['SaleVolume.kl.Tons.']> 0.10) | (correlation['SaleVolume.kl.Tons.'] <-0.10)]
    #print(correlation['Class'].sort_values(ascending = False))
    print(corr['SaleVolume.kl.Tons.'].sort_values(ascending=False))
    #.sort_values(ascending = False))

    #corre = data_sub_year[data_sub_year[1:]].corr()['SaleVolume.kl.Tons.'][:]
    #cor = corr[corr>0.20]
    #print(cor)
    #matrix = np.triu(data_sub_year.corr())
    #sns.heatmap(data_sub_year.corr(),annot=True ,cbar_kws={'orientation':'horizontal'}, mask=matrix, fmt='.1g', 
     #           linewidths=3, linecolor= 'black')
    #print(data_sub_year.head())
    #sns.heatmap(data_sub_year['SaleVolume.kl.Tons.'], bins=int(24/5), color='blue')
#fig.show()
#print(matrix)


# In[14]:


import matplotlib.pyplot as plt
#year = [2016,2017,2018,2019]
region = ['SUMATERA UTARA','SULAWESI','KALIMANTAN','JAWA BARAT','SUMATERA TENGAH','SUMATERA SELATAN','JAWA TENGAH',
     'JAWA TIMUR','NUSA TENGGARA DAN PAPUA']

for i in region:
    data_sub_year = data_sub1[data_sub1['Region'] == i]
    fig, ax =plt.subplots()
    plt.figure(figsize=(20,15))
    #layout(5,1)
    plt.title(i)
    matrix = np.triu(data_sub_year.corr())
    sns.heatmap(data_sub_year.corr(),annot=True ,cbar_kws={'orientation':'horizontal'}, mask=matrix, fmt='.1g', 
                linewidths=3, linecolor= 'black')
    #print(data_sub_year.head())
    #sns.heatmap(data_sub_year['SaleVolume.kl.Tons.'], bins=int(24/5), color='blue')
fig.show()


# In[27]:


## subset the data for 2016
data_2016 = data_sub[data_sub.Year==2016]
data_2016.shape


# In[28]:


data_2016_sub = data_2016[['SaleVolume.kl.Tons.','max_precipitation_rate', 'max_temperature', 'min_temperature',
       'max_dew_point_temperature', 'min_dew_point_temperature','avg_dew_point_temperature', 'total_precipitation', 'max_wind_speed',
       'min_wind_speed', 'avg_wind_speed', 'avg_wind_direction','max_wind_gust', 'max_relative_humidity', 'min_relative_humidity',
       'avg_relative_humidity', 'total_downward_solar_radiation','max_downward_solar_radiation', 'total_net_solar_radiation',
       'max_net_solar_radiation', 'max_atmospheric_pressure','min_atmospheric_pressure', 'avg_atmospheric_pressure',
       'avg_total_cloud_cover','avg_snow_density','max_soil_temperature_level_1', 'max_soil_temperature_level_2',
       'max_soil_temperature_level_3', 'max_soil_temperature_level_4','min_soil_temperature_level_1', 'min_soil_temperature_level_2',
       'min_soil_temperature_level_3', 'min_soil_temperature_level_4','avg_soil_temperature_level_1', 'avg_soil_temperature_level_2',
       'avg_soil_temperature_level_3', 'avg_soil_temperature_level_4','avg_soil_moisture_level_1', 'avg_soil_moisture_level_2',
       'avg_soil_moisture_level_3', 'avg_soil_moisture_level_4']]


# In[29]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,15))
matrix = np.triu(data_2016_sub.corr())
sns.heatmap(data_2016_sub.corr(),annot=True ,cbar_kws={'orientation':'horizontal'}, mask=matrix)

