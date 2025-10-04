import streamlit as st
import google.generativeai as genai


genai.configure(api_key="AIzaSyDIuixfLJ70KxcXc8Qo7KVXawPjW7G55e0")


model = genai.GenerativeModel("gemini-2.5-flash")
chat = model.start_chat()


def kontrol_et_ve_yonlendir(mesaj: str):
    m = mesaj.lower()
    cevaplar, linkler = [], []

    
    if any(k in m for k in ["su", "tüketim", "tahmin", "nüfus", "mahalle", "litre", "metreküp", "su tüketimi"]):
        cevaplar.append("💧 Su tahmini hizmeti için aşağıdaki bağlantıya tıklayabilirsiniz:")
        linkler.append(("💧 Su Tahmini", "1_Su_Tahmini"))

    
    if any(k in m for k in ["otopark", "park", "boş yer", "doluluk", "tek kare", "kare analizi", "park analizi"]):
        cevaplar.append("🅿️ Otopark durumu için aşağıdaki bağlantıya tıklayabilirsiniz:")
        linkler.append(("🅿️ Otopark Analizi", "2_Otopark_Analizi"))

    if any(k in m for k in ["sağlık", "risk", "hastane", "enfeksiyon", "yeşil alan", "nüfus yoğunluğu", "risk skoru"]):
        cevaplar.append("🩺 Sağlık risk tahmini için aşağıdaki bağlantıya tıklayabilirsiniz:")
        linkler.append(("🩺 Sağlık Risk Tahmini", "Saglik_Risk_Tahmini"))

    
    if any(k in m for k in ["kentsel", "dönüşüm", "kentsel dönüşüm", "imar", "zemin", "afet", "bina yaşı", "denetim", "öncelik", "riskli yapı"]):
        cevaplar.append("🏙️ Kentsel Dönüşüm Öncelik için aşağıdaki bağlantıya tıklayabilirsiniz:")
        linkler.append(("🏙️ Kentsel Dönüşüm Öncelik", "Kentsel_Donusum_Oncelik"))

    if any(k in m for k in ["Çalışan", "güvenlik", "ekipman", "kask", "kontrol", "yelek"]):
        cevaplar.append("🦺 Çalışan ekipmanları Kontrol için aşağıdaki bağlantıya tıklayabilirsiniz:")
        linkler.append(("🦺 Çalışan ekipmanları Kontrol", "Calisan_ekipman_kontrol"))

    return "\n\n".join(cevaplar) if cevaplar else None, linkler


st.set_page_config(page_title="BelediyeBot", page_icon="🤖")

st.markdown("""
<style>
    .chat-box {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .user-message {
        background-color: #d0ebff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .bot-message {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ BelediyeBot 🤖")
st.write("Merhaba! Size nasıl yardımcı olabilirim?")


if "chat_gecmisi" not in st.session_state:
    st.session_state.chat_gecmisi = []


if st.button("🗑️ Sohbeti Temizle"):
    st.session_state.chat_gecmisi = []
    st.rerun()


with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for sender, mesaj in st.session_state.chat_gecmisi:
        sinif = "bot-message" if sender == "BelediyeBot" else "user-message"
        st.markdown(f"<div class='{sinif}'>{mesaj}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if st.session_state.chat_gecmisi:
    son_mesaj = st.session_state.chat_gecmisi[-1]
    if son_mesaj[0] == "BelediyeBot":
        linkli_cevap, linkler = kontrol_et_ve_yonlendir(son_mesaj[1])
        if linkler:
            for label, sayfa_dosya_adi in linkler:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.page_link(f"pages/{sayfa_dosya_adi}.py", label=label, use_container_width=True)


kullanici_girdisi = st.chat_input("Sorunuzu buraya yazın...")

if kullanici_girdisi and kullanici_girdisi.strip() != "":
   
    st.session_state.chat_gecmisi.append(("Kullanıcı", kullanici_girdisi))

    
    linkli_cevap, _ = kontrol_et_ve_yonlendir(kullanici_girdisi)

   
    if linkli_cevap:
        cevap = linkli_cevap
    else:
        yanit = chat.send_message(kullanici_girdisi)
        cevap = yanit.text

    
    st.session_state.chat_gecmisi.append(("BelediyeBot", cevap))
    st.rerun()
