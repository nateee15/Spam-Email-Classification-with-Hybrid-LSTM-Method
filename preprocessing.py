import re
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

# TRY FIX → auto download NLTK stopwords if missing
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords

# load stopwords
stop_words_indo = set(stopwords.words('indonesian'))
stop_words_eng = set(stopwords.words('english'))
stop_words = stop_words_indo.union(stop_words_eng)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)
