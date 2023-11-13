#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


import pandas as pd

# Specify the correct encoding (e.g., 'latin-1' or 'ISO-8859-1') when reading the CSV file
data = pd.read_csv('spam.csv', encoding='latin-1')

# Display the first few rows of the dataset
print(data.head())


# In[5]:


X = data['v2']
y = data['v1'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[7]:


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)


# In[8]:


# Make predictions on the test set
predictions = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')


# In[ ]:




