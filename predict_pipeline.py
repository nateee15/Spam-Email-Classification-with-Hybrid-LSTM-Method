import pickle
import numpy as np
import onnxruntime as ort
from tensorflow.keras.utils import pad_sequences
from preprocessing import preprocess_text
from numeric_processing import extract_numeric_features

# Load tokenizer, scaler, label encoder
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
scaler     = pickle.load(open("scaler.pkl", "rb"))
le_labels  = pickle.load(open("label_encoder.pkl", "rb"))
numeric_info = pickle.load(open("numeric_info.pkl", "rb"))

maxlen           = numeric_info["maxlen"]
numeric_features = numeric_info["numeric_features"]

# Load ONNX model
session = ort.InferenceSession("spamnet_hybrid_attention.onnx")

def predict_email(text, from_domain="example.com"):
    # Preprocess text
    processed = preprocess_text(text)

    # Tokenize → pad
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=maxlen, padding='post').astype(np.int32)

    # Numerical features
    numeric_scaled = extract_numeric_features(
        text, from_domain, scaler, le_labels
    ).astype(np.float32)

    # Run ONNX session
    inputs = {
        "text_input": padded,
        "num_input": numeric_scaled
    }
    result = session.run(None, inputs)[0]
    prob = float(result[0][0])

    label = "spam" if prob >= 0.5 else "ham"

    return label, prob
