#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[2]:


data=pd.read_csv("Iris.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.columns


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data['Species'].value_counts()


# In[9]:


data['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.08,0.08,0.08])


# In[11]:


plt.figure(figsize=(9,7))
sns.heatmap(data.corr(),cmap='CMRmap',annot=True,linewidths=2)
plt.title("Correlation Graph",size=20)
plt.show()


# In[12]:


features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[14]:


from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# In[15]:


y_pred = dtree.predict(X_test)
y_pred


# In[16]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[17]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[18]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[19]:


dtree.predict([[5, 3.6, 1.4 , 0.2]])


# In[20]:


dtree.predict([[9, 3.1, 5, 1.5]])


# In[21]:


dtree.predict([[4.1, 3.0, 5.1, 1.8]])


# In[ ]:




