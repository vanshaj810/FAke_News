import pandas as pd
train=pd.read_csv("DATA SET/train.csv")
train.columns
#Checking for duplicates and removing them
train.drop_duplicates(inplace = True)
#Show the new shape (number of rows & columns)
train.shape
#Show the number of missing (NAN, NaN, na) data for each column
train.isnull().sum()
# data cleaning 
train["text"]=train["text"].str.lower()
train["keyword"].fillna("",inplace=True)
train["text"]=train["text"].str.lower()+" "+train["keyword"]
train = train[train['location'].notnull()]
#Show the number of missing (NAN, NaN, na) data for each column
train.isnull().sum()
train.head(5)

# data cleaning

import numpy as np
import nltk
from nltk.corpus import stopwords
import string

#Need to download stopwords
nltk.download('stopwords')

#Tokenization (a list of tokens), will be used as the analyzer
#1.Punctuations are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#2.Stop words in natural language processing, are useless words (data).
def process_text(text):
    
    #1 Remove Punctuationa
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3 Return a list of clean words
    return clean_words


#Show the Tokenization (a list of tokens )
train['text'].apply(process_text)

from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(train['text'])

#Split data into 75% training & 20% testing data sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, train['target'], test_size = 0.30, random_state = 0)

#Get the shape of messages_bow
messages_bow.shape

# Create and train the Multinomial Naive Bayes classifier which is suitable for
# classification with discrete features (e.g., word counts for text classification)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print('Accuracy: ', accuracy_score(y_test,pred))


import pickle
pickle.dump(classifier, open('model.pkl','wb'))