#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'><img src='../Pierian_Data_Logo.png'/></a>
# ___
# <center><em>Copyright by Pierian Data Inc.</em></center>
# <center><em>For more information, visit us at <a href='http://www.pieriandata.com'>www.pieriandata.com</a></em></center>

# # Logistic Regression Project
# 
# **GOAL: Create a Classification Model that can predict whether or not a person has presence of heart disease based on physical features of that person (age,sex, cholesterol, etc...)**

# ## Imports
# 
# **TASK: Run the cell below to import the necessary libraries.**

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ## Data
# 
# This database contains 14 physical attributes based on physical testing of a patient. Blood samples are taken and the patient also conducts a brief exercise test. The "goal" field refers to the presence of heart disease in the patient. It is integer (0 for no presence, 1 for presence). In general, to confirm 100% if a patient has heart disease can be quite an invasive process, so if we can create a model that accurately predicts the likelihood of heart disease, we can help avoid expensive and invasive procedures.
# 
# Content
# 
# Attribute Information:
# 
# * age
# * sex
# * chest pain type (4 values)
# * resting blood pressure
# * serum cholestoral in mg/dl
# * fasting blood sugar > 120 mg/dl
# * resting electrocardiographic results (values 0,1,2)
# * maximum heart rate achieved
# * exercise induced angina
# * oldpeak = ST depression induced by exercise relative to rest
# * the slope of the peak exercise ST segment
# * number of major vessels (0-3) colored by flourosopy
# * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# * target:0 for no presence of heart disease, 1 for presence of heart disease
# 
# Original Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
# 
# Creators:
# 
# Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
# University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
# University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
# V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

# ----
# 
# **TASK: Run the cell below to read in the data.**

# In[2]:


df = pd.read_csv('../DATA/heart.csv')


# In[3]:


df.head()


# In[4]:


df['target'].unique()


# ### Exploratory Data Analysis and Visualization
# 
# Feel free to explore the data further on your own.
# 
# **TASK: Explore if the dataset has any missing data points and create a statistical summary of the numerical features as shown below.**

# In[5]:


# CODE HERE


# In[6]:


df.info()


# In[7]:


# CODE HERE


# In[8]:


df.describe().transpose()


# ### Visualization Tasks
# 
# **TASK: Create a bar plot that shows the total counts per target value.**

# In[9]:


# CODE HERE!


# In[10]:


sns.countplot(x='target',data=df)


# **TASK: Create a pairplot that displays the relationships between the following columns:**
# 
#     ['age','trestbps', 'chol','thalach','target']
#    
# *Note: Running a pairplot on everything can take a very long time due to the number of features*

# In[11]:


# CODE HERE


# In[12]:


df.columns


# In[13]:


# Running pairplot on everything will take a very long time to render!
sns.pairplot(df[['age','trestbps', 'chol','thalach','target']],hue='target')


# **TASK: Create a heatmap that displays the correlation between all the columns.**

# In[14]:


# CODE HERE


# In[15]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap='viridis',annot=True)


# ----
# ----
# 
# # Machine Learning
# 
# ## Train | Test Split and Scaling
# 
# **TASK: Separate the features from the labels into 2 objects, X and y.**

# In[16]:


# CODE HERE


# In[17]:


X = df.drop('target',axis=1)
y = df['target']


# **TASK: Perform a train test split on the data, with the test size of 10% and a random_state of 101.**

# In[18]:


# CODE HERE


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


# **TASK: Create a StandardScaler object and normalize the X train and test set feature data. Make sure you only fit to the training data to avoid data leakage (data knowledge leaking from the test set).**

# In[21]:


# CODE HERE


# In[22]:


scaler = StandardScaler()


# In[23]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# ## Logistic Regression Model
# 
# **TASK: Create a Logistic Regression model and use Cross-Validation to find a well-performing C value for the hyper-parameter search. You have two options here, use *LogisticRegressionCV* OR use a combination of *LogisticRegression* and *GridSearchCV*. The choice is up to you, the solutions use the simpler *LogisticRegressionCV* approach.**

# In[24]:


# CODE HERE


# In[25]:


from sklearn.linear_model import LogisticRegressionCV 


# In[26]:


# help(LogisticRegressionCV)


# In[27]:


log_model = LogisticRegressionCV()


# In[28]:


log_model.fit(scaled_X_train,y_train)


# **TASK: Report back your search's optimal parameters, specifically the C value.** 
# 
# *Note: You may get a different value than what is shown here depending on how you conducted your search.*

# In[29]:


# CODE HERE


# In[30]:


log_model.C_


# In[31]:


log_model.get_params()


# ### Coeffecients
# 
# **TASK: Report back the model's coefficients.**

# In[32]:


log_model.coef_


# **BONUS TASK: We didn't show this in the lecture notebooks, but you have the skills to do this! Create a visualization of the coefficients by using a barplot of their values. Even more bonus points if you can figure out how to sort the plot! If you get stuck on this, feel free to quickly view the solutions notebook for hints, there are many ways to do this, the solutions use a combination of pandas and seaborn.**

# In[33]:


#CODE HERE


# In[34]:


coefs = pd.Series(index=X.columns,data=log_model.coef_[0])


# In[35]:


coefs = coefs.sort_values()


# In[36]:


plt.figure(figsize=(10,6))
sns.barplot(x=coefs.index,y=coefs.values);


# ---------
# 
# ## Model Performance Evaluation

# **TASK: Let's now evaluate your model on the remaining 10% of the data, the test set.**
# 
# **TASK: Create the following evaluations:**
# * Confusion Matrix Array
# * Confusion Matrix Plot
# * Classification Report

# In[53]:


# CODE HERE


# In[54]:


from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix


# In[55]:


y_pred = log_model.predict(scaled_X_test)


# In[56]:


confusion_matrix(y_test,y_pred)


# In[57]:


# CODE HERE


# In[58]:


plot_confusion_matrix(log_model,scaled_X_test,y_test)


# In[59]:


# CODE HERE


# In[60]:


print(classification_report(y_test,y_pred))


# ### Performance Curves
# 
# **TASK: Create both the precision recall curve and the ROC Curve.**

# In[63]:


from sklearn.metrics import plot_precision_recall_curve,plot_roc_curve


# In[64]:


# CODE HERE


# In[65]:


plot_precision_recall_curve(log_model,scaled_X_test,y_test)


# In[66]:


# CODE HERE


# In[67]:


plot_roc_curve(log_model,scaled_X_test,y_test)


# **Final Task: A patient with the following features has come into the medical office:**
# 
#     age          48.0
#     sex           0.0
#     cp            2.0
#     trestbps    130.0
#     chol        275.0
#     fbs           0.0
#     restecg       1.0
#     thalach     139.0
#     exang         0.0
#     oldpeak       0.2
#     slope         2.0
#     ca            0.0
#     thal          2.0

# **TASK: What does your model predict for this patient? Do they have heart disease? How "sure" is your model of this prediction?**
# 
# *For convience, we created an array of the features for the patient above*

# In[68]:


patient = [[ 54. ,   1. ,   0. , 122. , 286. ,   0. ,   0. , 116. ,   1. ,
          3.2,   1. ,   2. ,   2. ]]


# In[69]:


X_test.iloc[-1]


# In[70]:


y_test.iloc[-1]


# In[71]:


log_model.predict(patient)


# In[72]:


log_model.predict_proba(patient)


# 
