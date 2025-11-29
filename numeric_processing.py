import re
import numpy as np

def extract_numeric_features(email_text, from_domain, scaler, le_domain):
    # fitur numerik mentah
    num_urls        = len(re.findall(r"http[s]?://", email_text))
    num_exclaim     = email_text.count("!")
    has_attachment  = 0
    body_len        = len(email_text)
    num_special     = len(re.findall(r"[^a-zA-Z0-9\s]", email_text))

    # domain encoding
    from_domain_enc = le_domain.transform([from_domain])[0]

    # avg word len
    words = email_text.split()
    avg_word_len = np.mean([len(w) for w in words]) if len(words) else 0

    numeric = np.array([[num_urls, num_exclaim, has_attachment, body_len,
                         from_domain_enc, num_special, avg_word_len]])
    
    numeric_scaled = scaler.transform(numeric)
    return numeric_scaled
