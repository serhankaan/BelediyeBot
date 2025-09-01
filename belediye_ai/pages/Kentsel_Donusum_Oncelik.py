import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import leafmap.foliumap as leafmap
from streamlit_option_menu import option_menu
import tempfile
import streamlit.components.v1 as components
from html import escape


@st.cache_resource
def yukle_model():
    return load_model("urban_model.keras", compile=False)

@st.cache_data
def yukle_veri():
    return pd.read_excel("kentsel_donusum_oncelik_verisi_v2.xlsx")

@st.cache_data
def yukle_scaler_ve_kolonlar():
    scaler = joblib.load("urban_scaler.save")
    columns = joblib.load("urban_columns.save")
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
        icons=["activity", "map"],
        menu_icon="compass",
        default_index=0,
        orientation="vertical",
    )


if secilen == "Tahmin":
    st.markdown("<div class='title'>🏙️ Kentsel Dönüşüm Öncelik Tahmini</div>", unsafe_allow_html=True)

    model = yukle_model()
    veri = yukle_veri()
    scaler, X_columns = yukle_scaler_ve_kolonlar()

    
    med = veri[X_columns].median()

    bina_yasi = st.slider("🏚️ Bina Yaşı (yıl)", 0, 50, int(med.get("Bina_Yasi", 25)))
    imar = st.slider("📏 İmar Uyumsuzluk (0–1)", 0.0, 1.0, float(med.get("Imar_Uyumsuzluk", 0.5)), step=0.01)
    sikayet = st.slider("📢 Şikayet Oranı (0–1)", 0.0, 1.0, float(med.get("Sikayet_Orani", 0.4)), step=0.01)
    afet = st.slider("🌪️ Afet Riski (0–1)", 0.0, 1.0, float(med.get("Afet_Riski", 0.5)), step=0.01)
    zemin = st.slider("🏞️ Zemin Kötü (0–1)", 0.0, 1.0, float(med.get("Zemin_Kotu", 0.5)), step=0.01)
    denetim = st.slider("🛠️ Denetim Eksiği (0–1)", 0.0, 1.0, float(med.get("Denetim_Eksigi", 0.3)), step=0.01)

    if st.button("Tahmin Et"):
       
        user_data = pd.DataFrame([[0]*len(X_columns)], columns=X_columns)
        user_data["Bina_Yasi"] = bina_yasi
        user_data["Imar_Uyumsuzluk"] = imar
        user_data["Sikayet_Orani"] = sikayet
        user_data["Afet_Riski"] = afet
        user_data["Zemin_Kotu"] = zemin
        user_data["Denetim_Eksigi"] = denetim

        user_scaled = scaler.transform(user_data)
        tahmin = model.predict(user_scaled)[0][0]

        
        if tahmin <= 0.33:
            kategori = "İyi"
        elif tahmin <= 0.66:
            kategori = "Orta"
        else:
            kategori = "Yüksek"

        st.success(f"🏘️ Öncelik Skoru: **{tahmin:.3f}** → Kategori: **{kategori}**")


elif secilen == "Harita":
    st.markdown("<div class='title'>🗺️ Mahalle Öncelik Haritası</div>", unsafe_allow_html=True)

    df = yukle_veri()

    ilceler = ["(Hepsi)"] + sorted(df["Ilce"].dropna().unique().tolist())
    ilce = st.selectbox("🏘️ İlçe Seçin", ilceler)

    if ilce != "(Hepsi)":
        secili_df = df[df["Ilce"] == ilce].dropna(subset=["Enlem", "Boylam"])
    else:
        secili_df = df.dropna(subset=["Enlem", "Boylam"])

    pivot = secili_df.groupby("Mahalle").agg({
        "Oncelik_Skoru": "mean",
        "Bina_Yasi": "mean",
        "Imar_Uyumsuzluk": "mean",
        "Sikayet_Orani": "mean",
        "Afet_Riski": "mean",
        "Zemin_Kotu": "mean",
        "Denetim_Eksigi": "mean",
        "Enlem": "first",
        "Boylam": "first"
    }).reset_index()

    m = leafmap.Map(center=(pivot["Enlem"].mean(), pivot["Boylam"].mean()), zoom=13)

    for _, row in pivot.iterrows():
        score = row['Oncelik_Skoru']
        if score <= 0.33:
            kategori = "İyi"
        elif score <= 0.66:
            kategori = "Orta"
        else:
            kategori = "Yüksek"

        popup_html = f"""
        <div style='font-size:14px; line-height:1.5;'>
        <b style='font-size:16px;'>{escape(row['Mahalle'])}</b><br>
        <b>Öncelik Skoru:</b> {score:.3f} ({kategori})<br>
        <b>Bina Yaşı:</b> {row['Bina_Yasi']:.1f}<br>
        <b>İmar Uyumsuzluk:</b> {row['Imar_Uyumsuzluk']:.2f}<br>
        <b>Şikayet Oranı:</b> {row['Sikayet_Orani']:.2f}<br>
        <b>Afet Riski:</b> {row['Afet_Riski']:.2f}<br>
        <b>Zemin Kötü:</b> {row['Zemin_Kotu']:.2f}<br>
        <b>Denetim Eksiği:</b> {row['Denetim_Eksigi']:.2f}
        </div>
        """
        m.add_marker(
            location=(row["Enlem"], row["Boylam"]),
            popup=popup_html,
            popup_max_width=400
        )

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as tmpfile:
        m.to_html(outfile=tmpfile.name)
        html_path = tmpfile.name

    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, width=1100, height=700)
