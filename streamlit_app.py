import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd


load_model = joblib.load('data_resources/svm.pkl')
vectorizer = joblib.load('data_resources/svm_vectorize.pkl')


selected = option_menu(
    menu_title="Sentiment Analisis Ulasan Kebun Binatang Surabaya",
    options=["Dataset", "Prediksi Ulasan Baru"],
    orientation="horizontal",
)

if selected == 'Dataset':
    df = pd.read_csv("data_resources/ulasan_kebun_binatang.csv")
    df

else:
    # inputan
    ulasan = st.text_input('Masukkan ulasan')
    button = st.button('Predict')

    if button:
        # pembobotan menggunakan vectorize
        x_new = vectorizer.transform([ulasan])

        # predict menggunakan model
        predictions = load_model.predict(x_new)
        sentimen_class = ['Negatif', 'Netral', 'Positif']
        st.write("Sentimen: ", sentimen_class[predictions[0]])
