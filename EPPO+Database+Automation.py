
# coding: utf-8

# ### Import Library

# In[ ]:


import pandas as pd
from selenium import webdriver


# ### Define Chrome path

# In[ ]:


chrome_path = 'C:/Users/yogesh/Downloads/chromedriver.exe'


# ### Call chrome path and pass the url to scrap

# In[ ]:


driver = webdriver.Chrome(chrome_path)
driver.get("https://gd.eppo.int/")


# ### Ask user to give input

# In[ ]:


input_user = input("Search by name or EPPO code ")


# ### Search the user input element through xpath 

# In[ ]:


search = driver.find_element_by_xpath("""//*[@id="quickkw"]""")
search.send_keys(input_user)
search.submit()


# ### Collect mutiple links

# In[ ]:


collective_links = driver.find_element_by_xpath("""//*[@id="tresults"]/tbody""")

## function to get all links
def get_all_links(driver):
    links = []
    elements = driver.find_elements_by_tag_name('a')
    for elem in elements:
        href = elem.get_attribute("href")
        links.append(href)
    return links

## Call multile links using function
all_link = get_all_links(collective_links)
print(all_link)


# ### Create an empty dataftame with columns 'EPPOCode', 'ScientificName','CommonName'

# In[ ]:


df = pd.DataFrame(columns = ['EPPOCode','ScientificName','CommonName'])
for url in all_link:
    driver.get(url)
    EPPO_Code = driver.find_element_by_xpath("""/html/body/div[3]/div/div/div[2]/div/div/div[2]/div/div[1]/div[1]/ul/li[1]""").text
    Scientific_Name = driver.find_element_by_xpath("""/html/body/div[3]/div/div/div[2]/div/div/div[2]/div/div[1]/div[1]/ul/li[2]""").text
    Common_Name = driver.find_element_by_xpath("""//*[@id="tbcommon"]/tbody""").text
    Common_Name = Common_Name.split('\n')
    for rawdata in Common_Name:
        if 'English' in rawdata:
            #rint(">>", EPPO_Code, Scientific_Name,rawdata)
            #print(rawdata)
            df = df.append({'EPPOCode':EPPO_Code.replace("EPPO Code:",''), 'ScientificName':Scientific_Name.replace("Preferred name:",''),'CommonName':rawdata}, ignore_index = True)
    #driver.quit()
print(df)

