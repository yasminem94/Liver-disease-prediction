#!/usr/bin/env python
# coding: utf-8

# In[196]:


#Import all required libraries for reading data, analysing and visualizing data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder


# # Data Analysis

# In[197]:


#Read the training & test data
liver_df = pd.read_csv(r'C:\Users\bahareh chimehi\Desktop\Indian Liver Patient Dataset (ILPD) (3).csv')


# This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Dataset" column is a class label used to divide groups into liver patient (liver disease) or not (no disease).

# In[198]:


liver_df.head()


# In[199]:


liver_df.info()


# Observation from the dataset:
# 1) Only gender is non-numeric veriable. All others are numeric.
# 2) There are 10 features and 1 output - dataset.
# indicates that the patient has liver disease and 0 indicates the patient does not have liver disease.

# In[200]:


# statistical information about data
liver_df.describe(include='all')

#We can see that there are missing values for Albumin_and_Globulin_Ratio as only 579 entries have valid values indicating 4 missing values.
#Gender has only 2 values - Male/Female


# In[201]:


#Which features are available in the dataset
liver_df.columns


# In[202]:


#Check for any null values
liver_df.isnull().sum()


# The only data that is null is the Albumin_and_Globulin_Ratio - Only 4 rows are null. Lets see whether this is an important feature

# # Data Visualization

# In[203]:


sns.countplot(data=liver_df, x = 'Dataset', label='Count')

LD, NLD = liver_df['Dataset'].value_counts()
print('Number of patients diagnosed with liver disease: ',LD)
print('Number of patients not diagnosed with liver disease: ',NLD)


# In[204]:


g = sns.FacetGrid(liver_df, col="Dataset", row="Gender", margin_titles=True)
g.map(plt.hist, "Age", color="red")
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Disease by Gender and Age');


# In[205]:


liver_df.head(3)


# In[206]:


#Convert categorical variable "Gender" to indicator variables
pd.get_dummies(liver_df['Gender'], prefix = 'Gender').head()


# In[207]:


liver_df = pd.concat([liver_df,pd.get_dummies(liver_df['Gender'], prefix = 'Gender')], axis=1)


# In[208]:


liver_df.head()


# In[209]:


liver_df.describe()


# In[210]:


liver_df[liver_df['Albumin_and_Globulin_Ratio'].isnull()]


# In[211]:


liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())


# In[212]:


# The input variables/features are all the inputs except Dataset. The prediction or label is 'Dataset' that determines whether the patient has liver disease or not. 
X = liver_df.drop(['Gender','Dataset'], axis=1)
X.head(3)


# In[213]:


y = liver_df['Dataset'] # 1 for liver disease; 2 for no liver disease


# In[214]:


# Correlation
liver_corr = X.corr()


# In[215]:


liver_corr


# In[216]:


plt.figure(figsize=(30, 30))
sns.heatmap(liver_corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           cmap= 'coolwarm')
plt.title('Correlation between features');


# # Machine Learning

# In[217]:


# Importing modules
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier


# In[218]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[219]:


print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[220]:


#2) Logistic Regression
# Create logistic regression object
logreg = LogisticRegression()
# Train the model using the training sets and check score
logreg.fit(X_train, y_train)
#Predict Output
log_predicted= logreg.predict(X_test)


# In[221]:


#Equation coefficient and Intercept

print('Coefficient: \n', logreg.coef_)
print('Intercept: \n', logreg.intercept_)
print('Accuracy: \n', accuracy_score(y_test,log_predicted))


# In[222]:


sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")


# In[223]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
#Predict Output
rf_predicted = random_forest.predict(X_test)


# In[224]:



print('Accuracy: \n', accuracy_score(y_test,rf_predicted))


# In[225]:


sns.heatmap(confusion_matrix(y_test,log_predicted),annot=True,fmt="d")

