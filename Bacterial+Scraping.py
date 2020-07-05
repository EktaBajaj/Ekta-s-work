
# coding: utf-8

# ### Import Library
# 

# In[2]:


import pandas as pd
from selenium import webdriver


# ### Define Chrome path

# In[10]:


chrome_path = 'C:/Users/yogesh/Downloads/chromedriver.exe'


# ### Read file and convert pathogen name into list

# In[11]:


pathogen_list = pd.read_excel('C:/Users/yogesh/Desktop/PathogenName.xlsx')
user_input = pathogen_list.pathogen_name.tolist()


# ### Create links

# In[12]:


url = []
for spe in range(0,len(user_input)):
    Link = 'https://www.ezbiocloud.net/search?tn=' + user_input[spe]
    url.append(Link)


# ### Function that hit target url and extract info

# In[13]:


result = []
def function(targeturl):
    driver = webdriver.Chrome(chrome_path)
    driver.get(targeturl)
    element = driver.find_element_by_xpath("//*[@id='ezbioItems']")
    result.append(element.text)
    driver.quit()
    
for i in range(0,5):
    webPage = url[i]
    function(webPage)


# ### Data manipulation on result

# In[14]:


for i in range(0,len(result)):
    if result[i] != "No results were found":
        result[i] = result[i].split('\n')[1]
    if 'Basonym' not in result[i]:
        result[i] = "No result"


# In[16]:


result


# ### Create an empty dataframe with columns 'Species' and  'Basonym'

# In[31]:


df = pd.DataFrame(columns=['Species','Basonym'])
df = pd.DataFrame({'Species':user_input[0:5],'Basonym':result})
df['Basonym'] = df.Basonym.str.replace('Basonym:','', case = False)
print(df)

