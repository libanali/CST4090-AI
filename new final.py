#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score
import seaborn as sb
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_score
from sklearn import metrics

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('Desktop/TheDataset.csv')


# In[3]:


df.head()


# In[10]:


X = df.iloc[:, 0]
Y = df.iloc[:, 1]
sc = StandardScaler()


# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[12]:


X_train= X_train.values.reshape(-1, 1)
X_train= Y_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)


# In[13]:


KNN = KNeighborsClassifier(n_neighbors = 3) 
KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)


# In[14]:


acc = accuracy_score(Y_test, Y_pred) 
acc


# In[8]:


prediction_output = pd.DataFrame(data = [Y_test.values, Y_pred], index = ['Y_test', 'Y_Prediction'])


# In[9]:


prediction_output.transpose()


# In[ ]:




