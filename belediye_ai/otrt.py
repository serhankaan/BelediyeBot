import cv2
import pickle
import os


gorsel_yolu = "ilk_kare.jpg"

ı
kayit_dosyasi = "ParkAlanlari.pkl"


width, height = 28, 16



if os.path.exists(kayit_dosyasi):
    with open(kayit_dosyasi, "rb") as f:
        posList = pickle.load(f)
else:
    posList = []

def mouseClick(event, x, y, flags, params):
    global posList
    degisti = False

    if event == cv2.EVENT_LBUTTONDOWN:
        row = len(posList) // 8 + 1
        col = len(posList) % 8 + 1
        posList.append(((x, y), (row, col)))
        degisti = True

    if event == cv2.EVENT_RBUTTONDOWN:
        for i, (pos, _) in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
                degisti = True
                break

    if degisti:
        with open(kayit_dosyasi, "wb") as f:
            pickle.dump(posList, f)
        print(f"[+] {len(posList)} kutu kaydedildi.")


img = cv2.imread(gorsel_yolu)

while True:
    temp_img = img.copy()

    
    for pos, (row, col) in posList:
        x, y = pos
        cv2.rectangle(temp_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(temp_img, f"{row}-{col}", (x + 2, y + 12),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("Kutu Sec", temp_img)
    cv2.setMouseCallback("Kutu Sec", mouseClick)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("[✓] Kutu secimi tamamlandi. pickle dosyasina kaydedildi.")

import cv2
import numpy as np
import pickle
import os


video_adi = "video.mp4"
kayit_dosyasi = "ParkAlanlari.pkl"  


width, height = 28, 16


if os.path.exists(kayit_dosyasi):
    with open(kayit_dosyasi, "rb") as f:
        posList = pickle.load(f)
else:
    print("[!] Kayıtlı park alanı bulunamadı.")
    posList = []


cap = cv2.VideoCapture(video_adi)

def guncel_durum_yazdir(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    median = cv2.medianBlur(thresh, 5)
    dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

    print("\n📌 Video Karede Tespit Edilen Park Durumları:")
    bos_sayac = 0
    for (x, y), (row, col) in posList:
        crop = dilate[y:y + height, x:x + width]
        count = cv2.countNonZero(crop)
        durum = "BOŞ" if count < 150 else "DOLU"
        simge = "🟩" if durum == "BOŞ" else "🟥"
        print(f" - [{row}-{col}] koordinat: ({x}, {y}) → {simge} {durum} | count = {count}")
        if durum == "BOŞ":
            bos_sayac += 1
    print(f"\n📊 Toplam: {bos_sayac}/{len(posList)} park yeri BOŞ\n")


while True:
    success, frame = cap.read()
    if not success:
        print("🎬 Video bitti.")
        break

    
    guncel_durum_yazdir(frame)

   
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 16)
    median = cv2.medianBlur(thresh, 5)
    img_dilate = cv2.dilate(median, np.ones((3, 3), np.uint8), iterations=1)

    bos_sayac = 0

    for pos, (row, col) in posList:
        x, y = pos
        crop = img_dilate[y:y + height, x:x + width]
        count = cv2.countNonZero(crop)

        if count < 150:
            renk = (0, 255, 0)
            durum = "BOŞ"
            bos_sayac += 1
        else:
            renk = (0, 0, 255)
            durum = "DOLU"

        etiket = f"{row}-{col}"
        cv2.rectangle(img, (x, y), (x + width, y + height), renk, 2)
        cv2.putText(img, etiket, (x + 2, y + 12), cv2.FONT_HERSHEY_PLAIN, 1.2, renk, 1)
        cv2.putText(img, durum, (x + 2, y + height - 2), cv2.FONT_HERSHEY_PLAIN, 1, renk, 1)

    cv2.putText(img, f"Bos: {bos_sayac}/{len(posList)}", (15, 24),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    cv2.imshow("Video - Park Alani Durumu", img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


