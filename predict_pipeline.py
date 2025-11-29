import json
import numpy as np
import onnxruntime as ort
from keras_preprocessing.sequence import pad_sequences

from preprocessing import preprocess_text
from numeric_processing import extract_numeric_features


# ==========================
# LOAD TOKENIZER (JSON)
# ==========================
with open("tokenizer.json") as f:
    tokenizer_data = json.load(f)

from keras_preprocessing.text import tokenizer_from_json
tokenizer = tokenizer_from_json(tokenizer_data)


# ==========================
# LOAD SCALER & LABEL ENCODER & NUMERIC INFO
# ==========================
import pickle

scaler = pickle.load(open("scaler.pkl", "rb"))
le_labels = pickle.load(open("label_encoder.pkl", "rb"))
numeric_info = pickle.load(open("numeric_info.pkl", "rb"))

maxlen           = numeric_info["maxlen"]
numeric_features = numeric_info["numeric_features"]


# ==========================
# LOAD ONNX MODEL
# ==========================
session = ort.InferenceSession("spamnet_hybrid_attention.onnx")


# ==========================
# PREDICT FUNCTION
# ==========================
def predict_email(text, from_domain="example.com"):
    # 1️⃣ PREPROCESS TEXT
    processed = preprocess_text(text)

    # 2️⃣ TOKENIZE
    seq = tokenizer.texts_to_sequences([processed])

    # 3️⃣ PAD SEQUENCES (WAJIB int32 untuk ONNX)
    padded = pad_sequences(
        seq,
        maxlen=maxlen,
        padding='post'
    ).astype(np.int32)          # *** FIX PENTING ***

    # 4️⃣ NUMERIC FEATURES (WAJIB float32 untuk ONNX)
    numeric_scaled = extract_numeric_features(
        text,
        from_domain,
        scaler,
        le_labels
    ).astype(np.float32)        # *** FIX PENTING ***

    # 5️⃣ RUN ONNX MODEL
    inputs = {
        "text_input": padded,
        "num_input": numeric_scaled
    }

    outputs = session.run(None, inputs)
    prob = float(outputs[0][0][0])

    # 6️⃣ LABEL BASED ON PROBABILITY
    label = "spam" if prob >= 0.5 else "ham"

    return label, prob
