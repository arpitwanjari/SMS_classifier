#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[14]:


data = pd.read_csv('spam.csv', encoding='latin-1')


# In[3]:


print("First few rows of the dataset:")
print(data.head())


# In[4]:


print("\nSummary statistics of the dataset:")
print(data.describe())


# In[ ]:





# In[5]:


print("\nMissing values in the dataset:")
print(data.isnull().sum())


# In[6]:


X = data['v1']
y = data['v2']


# In[7]:


vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)


# In[9]:


predictions = model.predict(X_test)


# In[10]:


accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)
print(f'\nAccuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:





# In[ ]:




