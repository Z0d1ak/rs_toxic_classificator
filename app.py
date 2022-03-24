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
import psycopg2
from datetime import date

app = Flask(__name__)

create_table_sql = """
CREATE TABLE IF NOT EXISTS comments (
  id serial PRIMARY KEY,
  toxic boolean NOT NULL,
  category VARCHAR ( 50 ) NOT NULL,
  comment_date timestamp NOT NULL
);
"""

url = urlparse.urlparse(os.environ['DATABASE_URL'])
dbname = url.path[1:]
user = url.username
password = url.password
host = url.hostname
port = url.port

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
    
@app.route('/classify')
def classify():
    body = request.json
    
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cursor = conn.cursor()
    category = body['category']
    cur_date = date.today()
    

    cursor.execute(create_table_sql)
    comment = body['text']
    result = pipeline.predict([comment])[0]
    probability = pipeline.predict_proba([comment])[0][1]
    
    cursor.execute(f"INSERT INTO comments (toxic, category, comment_date) VALUES ({result},'{category}','{cur_date}')")
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({"verdict": str(result), "probability": str(probability)})

if __name__ == '__main__': app.run(debug=False)