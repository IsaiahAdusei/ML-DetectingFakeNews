
'''
News Dataset to detect fake news
'''

# Importing the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import string
from nltk.corpus import stopwords
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Loading the data
news=pd.read_csv('news.csv')
news.head()


## Understanding the data
news.info()
# Checking the number of rows and columns
news.shape

# Checking the number of fake and real news in the data
news['label'].value_counts()
label_count=news.groupby('label').count()
label_count

# Visualizing the number of fake and real news in the data
plt.bar(label_count.index.values, label_count['text'])
plt.xlabel('News Label')
plt.ylabel('Number of each label')
plt.show()

# Extracting the input and label from the data
text=news.text
label=news.label
text.head()


## Cleaning the data
#Defining a function to clean the text data
def text_preproc(x):
  x = x.lower() #Transforming all the text to lower cases
  x = x.encode('ascii', 'ignore').decode() #Removing unicode characters in the text
  x = re.sub(r'https*\S+', ' ', x) #Removing URLs
  x = re.sub(r'@\S+', ' ', x) #Removing mentions
  x = re.sub(r'#\S+', ' ', x) #Removing hastags
  x = re.sub(r'\'\w+', '', x) #Removing ticks and the next character
  x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) #Removing puntuations
  x = re.sub(r'\w*\d+\w*', '', x) #Removing numbers
  x = re.sub(r'\s{2,}', ' ', x) #Replacing over spaces with single spacing
  return x
text = news.text.apply(text_preproc)


## Training and Clssifying data
#Splitting the data into training and testing set
x_train,x_test,y_train,y_test=train_test_split(text,label,test_size=0.2,random_state=10)

#Initializing a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set,and then transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initializing a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=100)
pac.fit(tfidf_train,y_train)


## Checking the accuracy of the model
#calculating the accuracy of the model
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Building the confusion matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
