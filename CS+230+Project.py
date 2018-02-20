
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
##load the data to pandas dataframe
data = pd.read_table('HiSeqV2')
print (data.info())


# In[32]:


##visualize the first few rows of the data
#print (data.iloc[1:8])
data_clean = data.select_dtypes(include=['float64'])
print(data_clean)


# In[48]:


## column names
column_names = list(data_clean.columns.values)
people = [s[:-3] for s in column_names]
label = [s[-2:] for s in column_names]


train_x = data_clean[0:800]
train_y = label[0:800]
test_x = data_clean[801:1218]
test_y = label[801:1218]


##print(len(train_x))
##print(len(train_y))
print(len(test_x))
print(len(test_y))

#print(train_x)
#print(train_y)
#print(len(test_x))
#print(len(test_y))







# In[25]:


logreg = sklearn.linear_model.LogisticRegression(C=1e5)


# In[49]:


logreg.fit(train_x, train_y)


# In[35]:


Z = logreg.predict(test_x)


# In[36]:


print(Z)


# In[50]:


logreg.score(test_x, test_y)

