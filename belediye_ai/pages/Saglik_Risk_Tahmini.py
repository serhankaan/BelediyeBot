import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
import leafmap.foliumap as leafmap
import tempfile
import streamlit.components.v1 as components
from html import escape

st.set_page_config(page_title="Sağlık Risk Tahmini", page_icon="🩺", layout="wide")


@st.cache_resource
def yukle_model():
    return load_model("risk_model.keras", compile=False)

@st.cache_data
def yukle_columns():
    return joblib.load("risk_columns.save")  

@st.cache_data
def yukle_scaler():
    return joblib.load("risk_scaler.save")

@st.cache_data
def yukle_harita_veri():
    
    return pd.read_excel("sentetik_saglik_risk_verisi_ilce_enlem_boylam_kategorili.xlsx")


st.markdown("""
<style>
.title {font-size: 36px; color: #0F766E; font-weight: bold;}
.sub {font-size: 18px; color: #115E59;}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    secilen = option_menu(
        menu_title="📍 Menü",
        options=["Tahmin", "Harita"],
        icons=["activity", "map"],
        menu_icon="stethoscope",
        default_index=0,
        orientation="vertical",
    )


def risk_kategori(skor: float) -> str:
    if skor <= 0.33: return "Düşük Risk"
    if skor <= 0.66: return "Orta Risk"
    return "Yüksek Risk"


if secilen == "Tahmin":
    st.markdown("<div class='title'>🩺 Mahalle Bazlı Sağlık Risk Tahmini</div>", unsafe_allow_html=True)

    model = yukle_model()
    X_columns = yukle_columns()
    scaler = yukle_scaler()

    
    df_all = yukle_harita_veri()
    defaults = {
        "Nufus": int(df_all["Nufus"].median()) if "Nufus" in df_all else 10000,
        "Yas_Ortalamasi": float(df_all["Yas_Ortalamasi"].median()) if "Yas_Ortalamasi" in df_all else 40.0,
        "Gelir_Seviyesi": float(df_all["Gelir_Seviyesi"].median()) if "Gelir_Seviyesi" in df_all else 15000,
        "Saglik_Basvuru": int(df_all["Saglik_Basvuru"].median()) if "Saglik_Basvuru" in df_all else 800,
        "Yesil_Alan_Orani": float(df_all["Yesil_Alan_Orani"].median()) if "Yesil_Alan_Orani" in df_all else 10.0,
        "Nufus_Yogunlugu": int(df_all["Nufus_Yogunlugu"].median()) if "Nufus_Yogunlugu" in df_all else 5000,
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        nufus = st.number_input("👥 Nüfus", min_value=0, value=defaults["Nufus"])
        yas = st.number_input("👶 Yaş Ortalaması", min_value=0.0, max_value=120.0, value=defaults["Yas_Ortalamasi"], step=0.1)
    with col2:
        gelir = st.number_input("💰 Gelir Seviyesi", min_value=0.0, value=defaults["Gelir_Seviyesi"], step=1.0)
        basvuru = st.number_input("🏥 Geçmiş Sağlık Başvurusu", min_value=0, value=defaults["Saglik_Basvuru"])
    with col3:
        yesil = st.number_input("🌳 Yeşil Alan Oranı (%)", min_value=0.0, max_value=100.0, value=defaults["Yesil_Alan_Orani"], step=0.1)
        yogunluk = st.number_input("🏙️ Nüfus Yoğunluğu", min_value=0, value=defaults["Nufus_Yogunlugu"])

    if st.button("Tahmin Et", type="primary"):
        
        girdi = {
            "Nufus": nufus,
            "Yas_Ortalamasi": yas,
            "Gelir_Seviyesi": gelir,
            "Saglik_Basvuru": basvuru,
            "Yesil_Alan_Orani": yesil,
            "Nufus_Yogunlugu": yogunluk,
        }
        user_df = pd.DataFrame([[girdi.get(col, 0) for col in X_columns]], columns=X_columns)

        
        user_scaled = scaler.transform(user_df)
        skor = float(model.predict(user_scaled)[0][0])
        kategori = risk_kategori(skor)

        st.success(f"🔮 Tahmini Risk Skoru: **{skor:.3f}**  →  **{kategori}**")
        st.caption("Not: Skor 0–1 aralığındadır; 0’a yakın düşük, 1’e yakın yüksek risktir.")


elif secilen == "Harita":
    import folium
    import tempfile
    import streamlit.components.v1 as components
    from html import escape
    import pandas as pd

    st.markdown("<div class='title'>🗺️ Mahalle Risk Haritası</div>", unsafe_allow_html=True)

    
    def yukle_harita_veri():
        
        return pd.read_excel("sentetik_saglik_risk_verisi_ilce_enlem_boylam_kategorili.xlsx")

    def risk_kategori(skor: float) -> str:
        if skor <= 0.33: return "Düşük Risk"
        if skor <= 0.66: return "Orta Risk"
        return "Yüksek Risk"

    
    df = yukle_harita_veri().dropna(subset=["Enlem", "Boylam"])

    
    renk = {"Düşük Risk": "green", "Orta Risk": "orange", "Yüksek Risk": "red"}

    
    center_lat = float(df["Enlem"].mean())
    center_lon = float(df["Boylam"].mean())

    
    m = folium.Map(location=(center_lat, center_lon), zoom_start=12)

    
    for _, row in df.iterrows():
        kat = row.get("Risk_Kategorisi", risk_kategori(float(row["Risk_Skoru"])))
        color = renk.get(str(kat), "gray")

        
        nufus = f"{int(row['Nufus']):,}".replace(",", ".")
        yas   = f"{float(row['Yas_Ortalamasi']):.1f}"
        gelir = f"{float(row['Gelir_Seviyesi']):,.0f}".replace(",", ".")
        basv  = f"{int(row['Saglik_Basvuru']):,}".replace(",", ".")
        yesil = f"{float(row['Yesil_Alan_Orani']):.1f}%"
        risk  = f"{float(row['Risk_Skoru']):.3f}"

        popup_html = f"""
        <div style="font-size:14px; line-height:1.35; min-width:360px;">
          <h4 style="margin:0 0 6px 0;">{escape(str(row['Mahalle']))}</h4>
          <div style="margin-bottom:6px; color:#555;">İlçe: <b>{escape(str(row['Ilce']))}</b></div>
          <table style="width:100%; border-collapse:collapse;">
            <tr><td style="padding:6px 8px;">Risk Skoru</td><td style="padding:6px 8px;"><b>{risk}</b></td></tr>
            <tr><td style="padding:6px 8px;">Kategori</td><td style="padding:6px 8px;"><b>{escape(str(kat))}</b></td></tr>
            <tr><td style="padding:6px 8px;">Nüfus</td><td style="padding:6px 8px;">{nufus}</td></tr>
            <tr><td style="padding:6px 8px;">Yaş Ortalaması</td><td style="padding:6px 8px;">{yas}</td></tr>
            <tr><td style="padding:6px 8px;">Gelir Seviyesi</td><td style="padding:6px 8px;">{gelir}</td></tr>
            <tr><td style="padding:6px 8px;">Sağlık Başvurusu</td><td style="padding:6px 8px;">{basv}</td></tr>
            <tr><td style="padding:6px 8px;">Yeşil Alan Oranı</td><td style="padding:6px 8px;">{yesil}</td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=(float(row["Enlem"]), float(row["Boylam"])),
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(folium.IFrame(html=popup_html, width=380, height=260), max_width=440)
        ).add_to(m)

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmpfile:
        m.save(tmpfile.name)
        html_path = tmpfile.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, width=1100, height=720)


