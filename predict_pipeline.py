import pickle
import numpy as np
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from preprocessing import preprocess_text
from numeric_processing import extract_numeric_features

# Load semua komponen
model = load_model("spamnet_hybrid_attention.keras", compile=False)

tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
scaler     = pickle.load(open("scaler.pkl", "rb"))
le_labels  = pickle.load(open("label_encoder.pkl", "rb"))
numeric_info = pickle.load(open("numeric_info.pkl", "rb"))

maxlen           = numeric_info["maxlen"]
numeric_features = numeric_info["numeric_features"]

def predict_email(text, from_domain="unknown.com"):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post')

    numeric_scaled = extract_numeric_features(
        text, from_domain, scaler, le_labels
    )

    prob = model.predict([padded, numeric_scaled])[0][0]
    label = "spam" if prob >= 0.5 else "ham"
    return label, float(prob)
