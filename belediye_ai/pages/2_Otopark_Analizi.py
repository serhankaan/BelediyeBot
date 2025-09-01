import streamlit as st
import cv2
import numpy as np
import pickle, os, time

# --- Otopark oturum değişkenleri (izole) ---
if "op_cap" not in st.session_state: st.session_state.op_cap = None
if "op_running" not in st.session_state: st.session_state.op_running = False
if "op_frame_idx" not in st.session_state: st.session_state.op_frame_idx = 0
if "op_last_raw" not in st.session_state: st.session_state.op_last_raw = None
if "op_last_vis" not in st.session_state: st.session_state.op_last_vis = None
if "op_last_durumlar" not in st.session_state: st.session_state.op_last_durumlar = []
if "op_last_bos" not in st.session_state: st.session_state.op_last_bos = 0
if "op_last_dolu" not in st.session_state: st.session_state.op_last_dolu = 0
if "op_video_placeholder" not in st.session_state: st.session_state.op_video_placeholder = None
if "op_grid_placeholder" not in st.session_state: st.session_state.op_grid_placeholder = None

VIDEO_PATH = "video.mp4"
PKL_PATH   = "ParkAlanlari.pkl"
WIDTH, HEIGHT = 28, 16
THRESH = 150
GRID_COLS = 8

def load_positions(pkl_path):
    if not os.path.exists(pkl_path):
        return []
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def process_frame(frame, pos_list, w, h, thresh):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (3, 3), 1)
    thr   = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 16)
    med   = cv2.medianBlur(thr, 5)
    dil   = cv2.dilate(med, np.ones((3, 3), np.uint8), iterations=1)

    bos = 0
    durumlar = []
    vis = frame.copy()

    for (x, y), (row, col) in pos_list:
        crop  = dil[y:y+h, x:x+w]
        count = int(cv2.countNonZero(crop))
        empty = count < thresh
        renk  = (0,255,0) if empty else (0,0,255)
        durum = "BOŞ" if empty else "DOLU"
        if empty: bos += 1

        cv2.rectangle(vis, (x, y), (x+w, y+h), renk, 2)
        cv2.putText(vis, f"{row}-{col}", (x+2, y+12),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, renk, 1)
        cv2.putText(vis, durum, (x+2, y+h-2),
                    cv2.FONT_HERSHEY_PLAIN, 1, renk, 1)

        durumlar.append((f"{row}-{col}", durum))

    cv2.putText(vis, f"Boş: {bos}/{len(pos_list)}", (15, 28),
                cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
    dolu = len(pos_list) - bos
    return vis, durumlar, bos, dolu

def ensure_cap():
    if st.session_state.op_cap is None:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            st.error(f"❌ Video açılamadı: {VIDEO_PATH}")
            st.stop()
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.op_frame_idx)
        st.session_state.op_cap = cap
    return st.session_state.op_cap

def read_next_frame(cap):
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.op_frame_idx = 0
        ok, frame = cap.read()
    if ok:
        st.session_state.op_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        st.session_state.op_last_raw = frame.copy()
    return ok, frame

def render_ui_image_and_grid(img_bgr, durumlar, bos, dolu, posList):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.session_state.op_video_placeholder.image(rgb, use_container_width=True)
    with st.session_state.op_grid_placeholder.container():
        st.subheader("📋 Park Yeri Durumları")
        cols = st.columns(GRID_COLS)
        for idx, (etiket, durum) in enumerate(durumlar):
            renk = "#4CAF50" if durum == "BOŞ" else "#F44336"
            with cols[idx % GRID_COLS]:
                st.markdown(
                    f"<div style='background-color:{renk};padding:10px 8px;"
                    f"border-radius:10px;text-align:center;color:white;"
                    f"font-weight:bold;margin-bottom:8px'>{etiket}<br>{durum}</div>",
                    unsafe_allow_html=True
                )
        st.info(f"Boş: **{bos}** | Dolu: **{dolu}** | Doluluk: **{(dolu/max(len(posList),1))*100:.1f}%**")

st.set_page_config(page_title="Otopark Takip", page_icon="🅿️", layout="wide")
st.title("🅿️ Otopark Durum Takip Sistemi")

posList = load_positions(PKL_PATH)
if not posList:
    st.error("❌ Kayıtlı park alanı bulunamadı! Önce ou.py ile işaretleme yapın.")
    st.stop()

# --- Kontroller ---
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
if c1.button("▶️ Başlat / Sürdür", type="primary"):
    st.session_state.op_running = True
if c2.button("⏸️ Durdur"):
    st.session_state.op_running = False
if c3.button("🖼️ Tek Kare Analizi"):
    if st.session_state.op_last_raw is not None:
        vis, durumlar, bos, dolu = process_frame(
            st.session_state.op_last_raw, posList, WIDTH, HEIGHT, THRESH
        )
        st.session_state.op_last_vis = vis
        st.session_state.op_last_durumlar = durumlar
        st.session_state.op_last_bos = bos
        st.session_state.op_last_dolu = dolu
if c4.button("⏭️ +1 Kare İleri"):
    cap = ensure_cap()
    ok, frame = read_next_frame(cap)
    if ok:
        vis, durumlar, bos, dolu = process_frame(frame, posList, WIDTH, HEIGHT, THRESH)
        st.session_state.op_last_vis = vis
        st.session_state.op_last_durumlar = durumlar
        st.session_state.op_last_bos = bos
        st.session_state.op_last_dolu = dolu
if c5.button("⏮️ Başa Sar"):
    if st.session_state.op_cap is not None:
        st.session_state.op_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    st.session_state.op_frame_idx = 0

# --- Hız ---
speed = st.slider("⏱️ Oynatma Hızı", 0.1, 1.0, 0.3, 0.1)
FRAME_DELAY = 0.03 / max(speed, 1e-6)

# --- Placeholder'lar ---
video_col, grid_col = st.columns([2,1])
if st.session_state.op_video_placeholder is None:
    st.session_state.op_video_placeholder = video_col.empty()
    st.session_state.op_grid_placeholder  = grid_col.empty()

# --- Çalıştırma ---
if st.session_state.op_running:
    cap = ensure_cap()
    loop_start = time.time()
    for _ in range(1):
        ok, frame = read_next_frame(cap)
        if not ok:
            break
        vis, durumlar, bos, dolu = process_frame(frame, posList, WIDTH, HEIGHT, THRESH)
        st.session_state.op_last_vis = vis
        st.session_state.op_last_durumlar = durumlar
        st.session_state.op_last_bos = bos
        st.session_state.op_last_dolu = dolu
        render_ui_image_and_grid(vis, durumlar, bos, dolu, posList)

        elapsed = time.time() - loop_start
        wait_more = max(FRAME_DELAY - elapsed, 0)
        time.sleep(wait_more)
        st.rerun()

if not st.session_state.op_running:
    if st.session_state.op_last_vis is not None:
        render_ui_image_and_grid(
            st.session_state.op_last_vis,
            st.session_state.op_last_durumlar,
            st.session_state.op_last_bos,
            st.session_state.op_last_dolu,
            posList
        )
    else:
        cap = ensure_cap()
        ok, frame = read_next_frame(cap)
        if ok:
            vis, durumlar, bos, dolu = process_frame(frame, posList, WIDTH, HEIGHT, THRESH)
            st.session_state.op_last_vis = vis
            st.session_state.op_last_durumlar = durumlar
            st.session_state.op_last_bos = bos
            st.session_state.op_last_dolu = dolu
            render_ui_image_and_grid(vis, durumlar, bos, dolu, posList)
