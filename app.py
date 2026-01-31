import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("📧Email Spam Detection")

text = st.text_area("Email Content",height=200)

if st.button("Check Spam"):
    if not text.strip():
        st.warning("please enter email")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("☠️ SPAM")
        else:
            st.success("👍 NOT SPAM")

