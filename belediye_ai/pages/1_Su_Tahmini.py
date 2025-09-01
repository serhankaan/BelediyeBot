import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import leafmap.foliumap as leafmap
from streamlit_option_menu import option_menu
import tempfile
import streamlit.components.v1 as components
from html import escape


@st.cache_resource
def yukle_model():
    return load_model("modelim.keras", compile=False)

@st.cache_data
def yukle_veri():
    return pd.read_csv("birlesik_su_tuketimi_3.csv", encoding="utf-8-sig")

@st.cache_data
def yukle_scaler_ve_kolonlar():
    scaler = joblib.load("scaler.save")
    columns = joblib.load("columns.save")
    return scaler, columns


st.markdown("""
<style>
.title {font-size: 36px; color: #0077B6; font-weight: bold;}
.sub {font-size: 18px; color: #023E8A;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    secilen = option_menu(
        menu_title="📍 Menü",
        options=["Tahmin", "Harita"],
        icons=["droplet", "map"],
        menu_icon="compass",
        default_index=0,
        orientation="vertical",
    )


if secilen == "Tahmin":
    st.markdown("<div class='title'>💧 Mahalle Bazlı Su Tüketimi Tahmini</div>", unsafe_allow_html=True)

    model = yukle_model()
    veri = yukle_veri()
    scaler, X_columns = yukle_scaler_ve_kolonlar()

    mahalleler = sorted(veri["Mahalle"].unique())
    mahalle = st.selectbox("📌 Mahalle Seçin", mahalleler)
    mevsim = st.selectbox("🌦️ Mevsim Seçin", sorted([col.replace("Mevsim_", "") for col in X_columns if "Mevsim_" in col]))
    sicaklik = st.slider("🌡️ Sıcaklık (°C)", 0, 45, 25)
    nufus = st.number_input("👥 Nüfus", min_value=0, value=10000)

    if st.button("Tahmin Et"):
        sinirlar = {"Kış": (0, 20), "İlkbahar": (15, 30), "Yaz": (30, 45), "Sonbahar": (10, 25)}
        min_s, max_s = sinirlar.get(mevsim, (0, 45))

        if not (min_s <= sicaklik <= max_s):
            st.error(f"{mevsim} için sıcaklık {min_s}–{max_s} °C arasında olmalı. Girdiğiniz: {sicaklik} °C")
        else:
            user_data = pd.DataFrame([[0]*len(X_columns)], columns=X_columns)
            user_data[f"Mevsim_{mevsim}"] = 1
            user_data["Sicaklik"] = sicaklik
            user_data["Nufus"] = nufus

            user_scaled = scaler.transform(user_data)
            tahmin = model.predict(user_scaled)[0][0]

            st.success(f"📍 {mahalle} mahallesinde tahmini su tüketimi: **{tahmin:.0f} m³** (≈ {tahmin * 1000:.0f} litre)")


elif secilen == "Harita":
    st.markdown("<div class='title'>🗺️ Leafmap ile Hızlı Harita</div>", unsafe_allow_html=True)

    df = yukle_veri()
    mevsim_sec = st.selectbox("🌦️ Mevsim Seçin", ["Kış", "İlkbahar", "Yaz", "Sonbahar"])

    secili_df = df[df["Mevsim"] == mevsim_sec].dropna(subset=["Enlem", "Boylam"])

    pivot = secili_df.groupby("Mahalle").agg({
        "SuTuketimi": "mean",
        "Sicaklik": "mean",
        "Enlem": "first",
        "Boylam": "first"
    }).reset_index()

    m = leafmap.Map(center=(pivot["Enlem"].mean(), pivot["Boylam"].mean()), zoom=13)

    for _, row in pivot.iterrows():
        popup = f"""
        <b>{escape(row['Mahalle'])}</b><br>
        Su Tüketimi: {row['SuTuketimi']:.1f} m³<br>
        Sıcaklık: {row['Sicaklik']:.1f} °C
        """
        m.add_marker(location=(row["Enlem"], row["Boylam"]), popup=popup)

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmpfile:
        m.to_html(outfile=tmpfile.name)
        html_path = tmpfile.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, width=1100, height=700)
