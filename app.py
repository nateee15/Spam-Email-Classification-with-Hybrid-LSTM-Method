import streamlit as st
from predict_pipeline import predict_email

st.title("Spam Email Classifier — Hybrid LSTM + Attention")

email_text = st.text_area("Masukkan teks email:")
from_domain = st.text_input("From Domain:", value="example.com")

if st.button("Prediksi"):
    label, prob = predict_email(email_text, from_domain)
    st.write(f"### Hasil: **{label.upper()}**")
    st.write(f"Probabilitas Spam: **{prob:.4f}**")
