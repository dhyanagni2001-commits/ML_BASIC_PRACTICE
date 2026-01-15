import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("/Users/dhyanagni/Desktop/ML_practice/ML_BASIC_PRACTICE/Spam_email_Project/spam.csv",encoding="latin-1")
df = df[['v1','v2']] ##selects two columns
df.columns=['labels','text'] #renames columns as labels and text
print(df.head(10))


import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stop_words = set(ENGLISH_STOP_WORDS) #words which are useless like is a am list will be downloaded
 

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text) #removes !#? number and add '' to that re.sub(finds all regex)
    text = text.lower()
    words = text.split() #sentence to words
    words = [word for word in words if word not in stop_words]
    text = " ".join(words) # make sentence again
    return text

df['text'] = df['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(df['text'])
Y = df['labels'].map({'spam':1, 'ham':0})
 

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
model=MultinomialNB()
model.fit(X_train,Y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)


accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

email = ["Dear XYZ, Lets schedule meet"]
email_vec = vectorizer.transform(email)
print(model.predict(email_vec))




