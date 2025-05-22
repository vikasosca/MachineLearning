'''
Quick Summary of What You're Doing:
Step            	Description
✅ Loading Data	From news_Fake.csv and news_True.csv
✅ Preprocessing	Lowercasing, removing punctuation, removing stopwords
✅ Feature Extraction	CountVectorizer turns text into a numeric matrix
✅ Model	MultinomialNB from scikit-learn
✅ Evaluation	Accuracy, confusion matrix, classification report
'''

import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Read files
fake_news = pd.read_csv("C:/Users/Admin/Desktop/AI/python/Datasets/news_Fake.csv")
true_news = pd.read_csv("C:/Users/Admin/Desktop/AI/python/Datasets/news_True.csv")

fake_news['label'] = 0
true_news['label'] = 1
fake_news.isnull().sum()
true_news.isnull().sum()
#combine datasets
df = pd.concat([fake_news[['title','text','label']], true_news[['title','text','label']]],ignore_index = True)

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#clean text
def clean_text(text):
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in  text.split()  if word not in stop_words])
    return text
df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.strip().astype(bool)]

vectorizer = CountVectorizer() #Tokenization of each owrod in the cleaned_text,  produces a 
                            #matrix where each row represents a document and each column represents a word
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']
#print(df['cleaned_text'].head())
#print(df['cleaned_text'].apply(lambda x: len(x.strip())).describe())
X_Train,X_Test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# Initialize Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_Train,y_train)
#Predictions
y_pred = nb.predict(X_Test)
# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
#Save the model
import joblib
joblib.dump(nb, 'fake_news_model.pkl')               # Save the trained model
joblib.dump(vectorizer, 'vectorizer.pkl')            # Save the CountVectorizer or TfidfVectorizer

import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Clean and classify new input /Lets test
new_article = """
India claims to have won the war against Pakistan in 1971 with glorious victory
"""
def classify_news(new_article):
    cleaned = clean_text(new_article)
    X_new = vectorizer.transform([cleaned])
    result = model.predict(X_new)[0]
    print(f"result: {result}")
    print("Prediction:", "Real News" if result == 1 else "Fake News")
    return "Real News" if result == 1 else "Fake News"

print(classify_news(new_article))