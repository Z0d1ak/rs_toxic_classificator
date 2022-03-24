from flask import Flask, render_template, request, jsonify
import json
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')
filename = 'finalized_model.sav'
snowball = SnowballStemmer(language="english")
russian_stop_words = stopwords.words("english")
model_filename = 'model.sav'
vectorizer_filename = 'vectorizer.sav'

def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    tokens = word_tokenize(sentence, language="english")
    tokens = [i for i in tokens if i not in string.punctuation]
    if remove_stop_words:
        tokens = [i for i in tokens if i not in russian_stop_words]
    tokens = [snowball.stem(i) for i in tokens]
    return tokens
    
loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
loaded_vectorizer.tokenizer = lambda x: tokenize_sentence(x, remove_stop_words=True)

pipeline = Pipeline([
    ("vectorizer", loaded_vectorizer),
    ("model", loaded_model)
])

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/classify')
def classify():
    body = request.json
    comment = body['text']
    result = pipeline.predict([comment])[0]
    probability = pipeline.predict_proba([comment])[0][1]
    return jsonify({"verdict": str(result), "probability": str(probability)})

if __name__ == '__main__': app.run(host='127.0.0.1',port=8000,debug=True)