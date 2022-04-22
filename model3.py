import pandas as pd
train=pd.read_csv("DATA SET/train.csv")
train.columns
train.head()

# data cleaning 

train["text"]=train["text"].str.lower()
train["keyword"].fillna("",inplace=True)
train["text"]=train["text"].str.lower()+" "+train["keyword"]

train.head()

train.target.value_counts()

#Split data into 75% training & 25% testing data sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train['text'], train['target'], test_size=0.25, random_state=7, shuffle=True)

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
x_train.head().apply(process_text)
#Show the Tokenization (a list of tokens )
x_test.head().apply(process_text)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.75)
vec_train = vectorizer.fit_transform(x_train.values.astype('U'))
vec_test = vectorizer.transform(x_test.values.astype('U'))

vec_train.shape

# Create and train the Multinomial Naive Bayes classifier which is suitable for
# classification with discrete features (e.g., word counts for text classification)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec_train, y_train)

y_pred=classifier.predict(vec_test)

#Evaluate the model on the test data set
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(vec_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

def findlabel(newtext):
    vec_newtest=vectorizer.transform([newtext])
    y_pred1=classifier.predict(vec_newtest)
    return y_pred1[0]

print(findlabel('The Latest: More Homes Razed by Northern California Wildfire - ABC News http://t.co/YmY4rSkQ3d'))

import pickle
pickle.dump(classifier, open('model3.pkl','wb'))
pickle.dump(vectorizer,open("vectorizer3.pickle", "wb"))