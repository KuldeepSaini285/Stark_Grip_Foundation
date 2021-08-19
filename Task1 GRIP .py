#!/usr/bin/env python
# coding: utf-8

# 
# Importing all libraries required

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Reading data from given link

# In[3]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(15)


# Plotting the distribution of scores

# In[4]:


s_data.plot(x='Hours', y='Scores', style='*')  
plt.title('Hours Studied vs Percentage Score')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Features Selections

# In[5]:


X = s_data.iloc[:, :-1].values  
Y = s_data.iloc[:, 1].values
print("Feature Selection Successfull")


# Training Testing and Spliting the model

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


# In[7]:


from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, Y_train) 

print("Training complete.")


# Plotting the regression line

# In[8]:


line = model.coef_*X+model.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# In[9]:


Y_pred = model.predict(X_test) # Predicting the scores


# Comparing Actual vs Predicted

# In[11]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
print(df)


# In[14]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(Y_test, Y_pred))


# In[17]:


x_input = eval(input("Enter no of hr "))
predicated_value = model.predict([[x_input]])
print('predicated_value=' ,predicated_value)


# In[ ]:




