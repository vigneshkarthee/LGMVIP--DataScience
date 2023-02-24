#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Reading the dataset

# In[2]:


url= "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"
df=pd.read_csv(url)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


print(df.shape)
print(len(df))


# In[6]:


df.info()
df.describe()


# In[7]:


df['Date'] = pd.to_datetime(df['Date'])


# In[8]:


df.sort_values(by='Date',ignore_index=True,inplace=True) #Sorting Values w.r.t the dates
df


# In[9]:


df_close = df.reset_index()['Close']
df_close


# Plotting the Close value

# In[10]:


df_close.plot()


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range=(0,1))
df1 = scale.fit_transform(np.array(df_close).reshape(-1,1))
df1


# Spiltting the data into train and test

# In[12]:


train_size = int(len(df1)*0.80)
test_size = len(df1)-train_size
train_data = df1[0:train_size,:]
test_data = df1[train_size:len(df1),:]


# In[13]:


train_data


# In[14]:


test_data


# In[15]:


time_step = 100
x_train,y_train = [],[]
for i in range(len(train_data)-time_step-1):
    a = train_data[i:(i+time_step),0]
    x_train.append(a)
    y_train.append(train_data[i+ time_step,0])
x_train,y_train = np.array(x_train),np.array(y_train)


# In[16]:


x_test,y_test = [],[]
for i in range(len(test_data)-time_step-1):
    b = test_data[i:(i+time_step),0]
    x_test.append(b)
    y_test.append(test_data[i+time_step,0])
x_test,y_test = np.array(x_test),np.array(y_test)


# In[17]:


print("X_train shape :", x_train.shape)
print("Y_train shape :", y_train.shape)
print("X_test shape  :", x_test.shape)
print("Y_test shape  :", y_test.shape)


# In[18]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test  = x_test.reshape(x_test.shape[0],x_test.shape[1],1)


# In[19]:


print("X_train shape :", x_train.shape)
print("Y_train shape :", y_train.shape)
print("X_test shape  :", x_test.shape)
print("Y_test shape  :", y_test.shape)


# Importing libraries for the Neural Network

# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout


# Building the Model

# In[21]:


model = Sequential()
#Adding the 1st layer of stacked lstm model
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))


# In[22]:


model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))


# In[23]:


model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))


# In[24]:


model.add(LSTM(units=50))
model.add(Dropout(0.2))


# In[25]:


model.add(Dense(units=1))


# In[26]:


model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# In[27]:


model.fit(x=x_train,  y=y_train, batch_size=32, epochs=100,   validation_data=(x_test,y_test), verbose=1)


# Prediction

# In[28]:


train_pred = model.predict(x_train)
test_pred = model.predict(x_test)


# In[29]:


train_pred = scale.inverse_transform(train_pred)
test_pred = scale.inverse_transform(test_pred)


# Evaluation

# In[30]:


from sklearn.metrics import mean_squared_error
from math import sqrt
sqrt(mean_squared_error(y_train,train_pred))


# In[31]:


sqrt(mean_squared_error(y_test, test_pred))


# In[32]:


lookback = 100
train_pred_plot = np.empty_like(df1)
train_pred_plot[:,:] = np.nan
train_pred_plot[lookback: len(train_pred)+lookback, :] = train_pred

test_pred_plot = np.empty_like(df1)
test_pred_plot[:,:] = np.nan
test_pred_plot[len(train_pred)+(lookback*2)+1: len(df1)-1, : ] = test_pred


# Plotting the Baselines and Predictions

# In[33]:


plt.plot(scale.inverse_transform(df1),label='baseline')
plt.plot(train_pred_plot,label='train_predictions')
plt.plot(test_pred_plot,label= 'test_predictions')
plt.legend()
plt.show()


# In[34]:


len(test_data)


# In[35]:


x_input = test_data[307:].reshape(1,-1)
x_input.shape


# In[36]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# Forecasting the Prediction upto 20 Days

# In[37]:


output=[]
n_steps=100
i=0
while(i<21):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        output.extend(yhat.tolist())
        i=i+1
    

print(output)


# In[38]:


df = df1.tolist()
df.extend(output)
df=scale.inverse_transform(df).tolist()
plt.plot(df)


# In[ ]:




