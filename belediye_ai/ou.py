import cv2
import pickle
import os


gorsel_yolu = "ilk_kare.jpg"


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

