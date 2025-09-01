import cv2
import json
import time
import numpy as np
from pathlib import Path

VIDEO_PATH = "Kanalizasyon altyapı işleri.mp4"
ROI_JSON   = "rois.json"


HSV_YO_LOW, HSV_YO_HIGH = (10, 50, 120), (30, 255, 255)  
ENABLE_WHITE_HELMET = False
HSV_WHITE_LOW, HSV_WHITE_HIGH = (0, 0, 200), (180, 50, 255)
HELMET_SCORE_TH    = 0.05
VEST_SCORE_TH      = 0.08
VEST_BRIGHT_WEIGHT = 0.40
HEAD_RATIO, TORSO_Y, TORSO_H = 0.25, 0.25, 0.50
PAD = 0
GRID_COLS = 8

mouse_pos = (0, 0)
paused    = False
rois      = []  


drawing = False
start_x, start_y = -1, -1

def load_rois(path: str = ROI_JSON):
    p = Path(path)
    if not p.exists(): return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [tuple(map(int, r)) for r in data]
    except:
        return []

def save_rois():
    Path(ROI_JSON).write_text(json.dumps(rois), encoding="utf-8")

def delete_rois_file():
    p = Path(ROI_JSON)
    if p.exists(): p.unlink()

def ratio(mask):
    a = mask.size
    return 0.0 if a == 0 else float(cv2.countNonZero(mask)) / a

def color_mask_hsv(bgr, low, high):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, 1)
    return m

def largest_roundness(mask, min_area=100):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        p = cv2.arcLength(c, True)
        if p == 0: continue
        best = max(best, 4.0 * np.pi * a / (p * p))
    return best

def clamp_box(x, y, w, h, W, H):
    x = max(0, int(x)); y = max(0, int(y))
    w = max(1, int(w)); h = max(1, int(h))
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def point_in_box(px, py, box):
    x, y, w, h = box
    return (px >= x and px <= x + w and py >= y and py <= y + h)

def process_person_roi(frame, box):
    H, W = frame.shape[:2]
    x, y, w, h = clamp_box(box[0] - PAD, box[1] - PAD, box[2] + 2*PAD, box[3] + 2*PAD, W, H)
    person = frame[y:y+h, x:x+w]

    hh = int(h*HEAD_RATIO)
    ty = int(h*TORSO_Y)
    th = int(h*TORSO_H)

    head  = person[0:hh, :]
    torso = person[ty:ty+th, :]

    mY = color_mask_hsv(head, HSV_YO_LOW, HSV_YO_HIGH)
    y_ratio, y_round = ratio(mY), largest_roundness(mY, min_area=max(60, (hh*person.shape[1])//500))
    scoreY = 0.6*y_ratio + 0.4*y_round

    scoreW = 0.0
    if ENABLE_WHITE_HELMET:
        mW = color_mask_hsv(head, HSV_WHITE_LOW, HSV_WHITE_HIGH)
        w_ratio, w_round = ratio(mW), largest_roundness(mW, min_area=max(60, (hh*person.shape[1])//500))
        scoreW = 0.6*w_ratio + 0.4*w_round

    helmet_score = max(scoreY, scoreW)
    has_helmet = (helmet_score >= HELMET_SCORE_TH)

    m = color_mask_hsv(torso, HSV_YO_LOW, HSV_YO_HIGH)
    v_ratio = ratio(m)
    v = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)[:, :, 2]
    bright = ratio(cv2.inRange(v, 220, 255)) * VEST_BRIGHT_WEIGHT
    vest_score = v_ratio + bright
    has_vest = (vest_score >= VEST_SCORE_TH)

    color = (0,180,0) if (has_helmet and has_vest) else ((0,165,255) if (has_helmet or has_vest) else (0,0,255))
    label = f"Kask {'VAR' if has_helmet else 'YOK'} | Yelek {'VAR' if has_vest else 'YOK'} (k:{helmet_score:.2f}, y:{vest_score:.2f})"
    return (x, y, w, h), color, label, has_helmet, has_vest

def on_mouse(event, x, y, flags, param):
    global mouse_pos, drawing, start_x, start_y, rois
    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pass  

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w, h = abs(x - start_x), abs(y - start_y)
        if w > 5 and h > 5:  
            x0, y0 = min(start_x, x), min(start_y, y)
            rois.append((x0, y0, w, h))
            save_rois()
            print(f"[+] ROI eklendi: {(x0, y0, w, h)}")

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, b in enumerate(rois):
            if point_in_box(x, y, b):
                removed = rois.pop(i)
                save_rois()
                print(f"[-] ROI silindi: {removed}")
                break

def main():
    global paused, rois, HELMET_SCORE_TH, VEST_SCORE_TH

    cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH); return

    rois = load_rois()

    cv2.namedWindow("Kask & Yelek (Çizerek ROI)")
    cv2.setMouseCallback("Kask & Yelek (Çizerek ROI)", on_mouse)

    fps_s, t0 = None, time.time()

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok: break

        disp = frame.copy()

        
        for idx, box in enumerate(rois, start=1):
            (X, Y, W, H), color, label, hasH, hasV = process_person_roi(frame, box)
            cv2.rectangle(disp, (X, Y), (X+W, Y+H), color, 2)
            row = (idx-1)//GRID_COLS + 1
            col = (idx-1)%GRID_COLS + 1
            tag = f"{row}-{col}"
            cv2.putText(disp, tag, (X+2, Y+14), cv2.FONT_HERSHEY_PLAIN, 1.2, color, 1)
            cv2.putText(disp, label, (X, max(0, Y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        
        if drawing:
            x0, y0 = min(start_x, mouse_pos[0]), min(start_y, mouse_pos[1])
            w, h = abs(mouse_pos[0] - start_x), abs(mouse_pos[1] - start_y)
            cv2.rectangle(disp, (x0, y0), (x0+w, y0+h), (255,255,255), 1)

        
        t1 = time.time(); fps = 1.0/max(1e-6, t1 - t0); t0 = t1
        fps_s = fps if fps_s is None else (0.9*fps_s + 0.1*fps)
        cv2.putText(disp, f"FPS:{fps_s:.1f}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60,220,60), 2)

        cv2.imshow("Kask & Yelek (Çizerek ROI)", disp)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:   
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('s'):
            save_rois()
        elif key == ord('l'):
            rois = load_rois()
        elif key == ord('c'):
            rois = []
            delete_rois_file()
        elif key == ord('1'):
            HELMET_SCORE_TH = max(0.0, HELMET_SCORE_TH - 0.01)
        elif key == ord('2'):
            HELMET_SCORE_TH = min(1.0, HELMET_SCORE_TH + 0.01)
        elif key == ord('3'):
            VEST_SCORE_TH = max(0.0, VEST_SCORE_TH - 0.01)
        elif key == ord('4'):
            VEST_SCORE_TH = min(1.0, VEST_SCORE_TH + 0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
