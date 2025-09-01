import json, os, time
from pathlib import Path
import cv2, numpy as np, streamlit as st


if "ce_cap" not in st.session_state: st.session_state.ce_cap = None
if "ce_running" not in st.session_state: st.session_state.ce_running = False
if "ce_frame_idx" not in st.session_state: st.session_state.ce_frame_idx = 0
if "ce_last_raw" not in st.session_state: st.session_state.ce_last_raw = None
if "ce_last_vis" not in st.session_state: st.session_state.ce_last_vis = None
if "ce_last_durumlar" not in st.session_state: st.session_state.ce_last_durumlar = []
if "ce_last_tam" not in st.session_state: st.session_state.ce_last_tam = 0
if "ce_last_eksik" not in st.session_state: st.session_state.ce_last_eksik = 0
if "ce_video_placeholder" not in st.session_state: st.session_state.ce_video_placeholder = None
if "ce_grid_placeholder" not in st.session_state: st.session_state.ce_grid_placeholder = None

VIDEO_PATH = "Kanalizasyon altyapı işleri.mp4"
ROI_JSON   = "rois.json"

HSV_YO_LOW, HSV_YO_HIGH = (10, 50, 120), (30, 255, 255)  
ENABLE_WHITE_HELMET = False
HSV_WHITE_LOW, HSV_WHITE_HIGH = (0, 0, 200), (180, 50, 255)
HELMET_SCORE_TH    = 0.05
VEST_SCORE_TH      = 0.08
VEST_BRIGHT_WEIGHT = 0.40
HEAD_RATIO, TORSO_Y, TORSO_H, PAD = 0.25, 0.25, 0.50, 0
GRID_COLS = 6

def load_rois(path: str = ROI_JSON):
    p = Path(path)
    if not p.exists(): return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [tuple(map(int, r)) for r in data]  
    except Exception:
        return []

def ratio(mask):
    a = mask.size
    return 0.0 if a == 0 else float(cv2.countNonZero(mask)) / a

def color_mask_hsv(bgr, low, high):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    k = np.ones((3,3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return m

def largest_roundness(mask, min_area=100):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        p = cv2.arcLength(c, True)
        if p == 0: continue
        best = max(best, 4.0*np.pi*a/(p*p))
    return best

def clamp_box(x,y,w,h,W,H):
    x=max(0,int(x)); y=max(0,int(y)); w=max(1,int(w)); h=max(1,int(h))
    if x+w>W: w=W-x
    if y+h>H: h=H-y
    return x,y,w,h

def process_frame(frame, rois):
    vis = frame.copy()
    H, W = frame.shape[:2]
    durumlar = []
    tam, eksik = 0, 0

    for idx, (x,y,w,h) in enumerate(rois, start=1):
        x,y,w,h = clamp_box(x,y,w,h,W,H)
        person = frame[y:y+h, x:x+w]

        # Baş ve Gövde
        hh = int(h*HEAD_RATIO)
        ty = int(h*TORSO_Y)
        th = int(h*TORSO_H)
        head  = person[0:hh, :]
        torso = person[ty:ty+th, :]

        # Kask
        mY = color_mask_hsv(head, HSV_YO_LOW, HSV_YO_HIGH)
        y_ratio, y_round = ratio(mY), largest_roundness(mY, min_area=max(60,(hh*person.shape[1])//500))
        scoreY = 0.6*y_ratio + 0.4*y_round

        scoreW = 0.0
        if ENABLE_WHITE_HELMET:
            mW = color_mask_hsv(head, HSV_WHITE_LOW, HSV_WHITE_HIGH)
            w_ratio, w_round = ratio(mW), largest_roundness(mW, min_area=max(60,(hh*person.shape[1])//500))
            scoreW = 0.6*w_ratio + 0.4*w_round

        helmet_score = max(scoreY, scoreW)
        has_helmet   = helmet_score >= HELMET_SCORE_TH

        # Yelek
        m = color_mask_hsv(torso, HSV_YO_LOW, HSV_YO_HIGH)
        v_ratio = ratio(m)
        v = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)[:,:,2]
        bright = ratio(cv2.inRange(v, 220, 255)) * VEST_BRIGHT_WEIGHT
        vest_score = v_ratio + bright
        has_vest   = vest_score >= VEST_SCORE_TH

        color = (0,180,0) if (has_helmet and has_vest) else ((0,165,255) if (has_helmet or has_vest) else (0,0,255))
        label = f"Kask {'VAR' if has_helmet else 'YOK'} | Yelek {'VAR' if has_vest else 'YOK'} (k:{helmet_score:.2f}, y:{vest_score:.2f})"

        cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
        dot_y = max(0, y-14)
        cv2.circle(vis, (x+8,  dot_y), 5, (0,180,0) if has_helmet else (0,0,255), -1)
        cv2.circle(vis, (x+24, dot_y), 5, (0,180,0) if has_vest   else (0,0,255), -1)
        cv2.putText(vis, label, (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        tag = f"ROI-{idx}"
        durumlar.append((tag, "VAR" if has_helmet else "YOK", "VAR" if has_vest else "YOK"))
        if has_helmet and has_vest: tam += 1
        else: eksik += 1

    return vis, durumlar, tam, eksik

def ensure_cap():
    if st.session_state.ce_cap is None:
        cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            st.error(f"❌ Video açılamadı: {VIDEO_PATH}")
            st.stop()
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.ce_frame_idx)
        st.session_state.ce_cap = cap
    return st.session_state.ce_cap

def read_next_frame(cap):
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        st.session_state.ce_frame_idx = 0
        ok, frame = cap.read()
    if ok:
        st.session_state.ce_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        st.session_state.ce_last_raw = frame.copy()
    return ok, frame

def render_ui_image_and_grid(img_bgr, durumlar, tam, eksik, rois):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.session_state.ce_video_placeholder.image(rgb, use_container_width=True)
    with st.session_state.ce_grid_placeholder.container():
        st.subheader("🦺 ROI Durumları")
        cols = st.columns(GRID_COLS)
        for idx, tup in enumerate(durumlar):
            # Güvenli açma: 3'lü bekleniyor, yanlış veri gelirse düşmesin
            if len(tup) == 3:
                etiket, kask, yelek = tup
            elif len(tup) == 2:
                etiket, kask = tup
                yelek = "YOK"
            else:
                etiket = str(tup[0]) if tup else f"ROI-{idx+1}"
                kask = "YOK"; yelek = "YOK"

            if kask=="VAR" and yelek=="VAR":
                renk = "#4CAF50"   # tam
            elif kask=="VAR" or yelek=="VAR":
                renk = "#FF9800"   # kısmi
            else:
                renk = "#F44336"   # eksik
            with cols[idx % GRID_COLS]:
                st.markdown(
                    f"<div style='background-color:{renk};padding:10px 8px;"
                    f"border-radius:10px;text-align:center;color:white;"
                    f"font-weight:bold;margin-bottom:8px'>{etiket}<br>"
                    f"Kask: {kask} | Yelek: {yelek}</div>",
                    unsafe_allow_html=True
                )
        toplam = len(rois)
        st.info(f"Tam Güvenlik: **{tam}** | Eksik: **{eksik}** | Toplam ROI: **{toplam}**")

st.set_page_config(page_title="Kask & Yelek Takip", page_icon="🦺", layout="wide")
st.title("🦺 Kask & Yelek Takip Sistemi")

rois = load_rois()
if not rois:
    st.error("❌ Kayıtlı ROI bulunamadı! Önce masaüstü uygulamada (algi.py) ROI çizin ve rois.json’a kaydedin.")
    st.stop()

# --- Kontroller ---
c1, c2, c3, c4, c5 = st.columns([1,1,1,1,2])
if c1.button("▶️ Başlat / Sürdür", type="primary"):
    st.session_state.ce_running = True
if c2.button("⏸️ Durdur"):
    st.session_state.ce_running = False
if c3.button("🖼️ Tek Kare Analizi"):
    if st.session_state.ce_last_raw is not None:
        vis, durumlar, tam, eksik = process_frame(st.session_state.ce_last_raw, rois)
        st.session_state.ce_last_vis = vis
        st.session_state.ce_last_durumlar = durumlar
        st.session_state.ce_last_tam = tam
        st.session_state.ce_last_eksik = eksik
if c4.button("⏭️ +1 Kare İleri"):
    cap = ensure_cap()
    ok, frame = read_next_frame(cap)
    if ok:
        vis, durumlar, tam, eksik = process_frame(frame, rois)
        st.session_state.ce_last_vis = vis
        st.session_state.ce_last_durumlar = durumlar
        st.session_state.ce_last_tam = tam
        st.session_state.ce_last_eksik = eksik
if c5.button("⏮️ Başa Sar"):
    if st.session_state.ce_cap is not None:
        st.session_state.ce_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    st.session_state.ce_frame_idx = 0

# --- Hız ---
speed = st.slider("⏱️ Oynatma Hızı", 0.1, 1.0, 0.3, 0.1)
FRAME_DELAY = 0.03 / max(speed, 1e-6)

# --- Placeholder'lar ---
video_col, grid_col = st.columns([2,1])
if st.session_state.ce_video_placeholder is None:
    st.session_state.ce_video_placeholder = video_col.empty()
    st.session_state.ce_grid_placeholder  = grid_col.empty()

# --- Çalıştırma ---
if st.session_state.ce_running:
    cap = ensure_cap()
    loop_start = time.time()
    for _ in range(1):
        ok, frame = read_next_frame(cap)
        if not ok:
            break
        vis, durumlar, tam, eksik = process_frame(frame, rois)
        st.session_state.ce_last_vis = vis
        st.session_state.ce_last_durumlar = durumlar
        st.session_state.ce_last_tam = tam
        st.session_state.ce_last_eksik = eksik
        render_ui_image_and_grid(vis, durumlar, tam, eksik, rois)
        elapsed = time.time() - loop_start
        wait_more = max(FRAME_DELAY - elapsed, 0)
        time.sleep(wait_more)
        st.rerun()

if not st.session_state.ce_running:
    if st.session_state.ce_last_vis is not None:
        render_ui_image_and_grid(
            st.session_state.ce_last_vis,
            st.session_state.ce_last_durumlar,
            st.session_state.ce_last_tam,
            st.session_state.ce_last_eksik,
            rois
        )
    else:
        cap = ensure_cap()
        ok, frame = read_next_frame(cap)
        if ok:
            vis, durumlar, tam, eksik = process_frame(frame, rois)
            st.session_state.ce_last_vis = vis
            st.session_state.ce_last_durumlar = durumlar
            st.session_state.ce_last_tam = tam
            st.session_state.ce_last_eksik = eksik
            render_ui_image_and_grid(vis, durumlar, tam, eksik, rois)
