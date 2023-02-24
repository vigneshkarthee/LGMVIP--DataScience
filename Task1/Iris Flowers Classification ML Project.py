#!/usr/bin/env python
# coding: utf-8

# # Iris Flowers Classification ML Project

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Iris.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df['Species'].value_counts()


# In[9]:


df['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[12]:


df['Species'].unique()


# In[13]:


from sklearn.model_selection import train_test_split

features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = df.loc[:, features].values   #defining the feature matrix
Y = df.Species

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 40,random_state=0)
     


# In[14]:


X_Train.shape


# In[15]:


Y_Train.shape


# In[16]:


Y_Test.shape


# In[17]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)


# In[18]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix
     


# In[19]:


#Model Creation
from sklearn.linear_model import LogisticRegression
log_model= LogisticRegression(random_state = 0)
log_model.fit(X_Train, Y_Train)

# model training
log_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_log_res=log_model.predict(X_Test)


# In[20]:


Y_Pred_Test_log_res


# In[21]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)*100)


# In[22]:


print(classification_report(Y_Test, Y_Pred_Test_log_res))


# In[23]:


confusion_matrix(Y_Test,Y_Pred_Test_log_res )

