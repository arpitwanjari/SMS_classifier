#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


data = pd.read_csv('spam.csv', encoding='latin-1')


# In[5]:


data.head()


# In[6]:


data.describe()


# In[12]:


plt.figure(figsize=(6, 4))
sns.heatmap(pd.crosstab(index=data['v1'], columns="count"), annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Distribution of Spam and Non-Spam Messages')
plt.show()


# In[13]:


X = data['v2']
y = data['v1']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[16]:


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# In[17]:


predictions = model.predict(X_test_vectorized)


# In[18]:


accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Display evaluation metrics
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:




