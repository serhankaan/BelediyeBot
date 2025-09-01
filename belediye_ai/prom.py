import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from keras.saving import save_model


df = pd.read_excel('kentsel_donusum_oncelik_verisi_v2.xlsx')


feature_cols = [
    'Bina_Yasi',
    'Imar_Uyumsuzluk',
    'Sikayet_Orani',
    'Afet_Riski',
    'Zemin_Kotu',
    'Denetim_Eksigi'
]
target_col = 'Oncelik_Skoru'

X = df[feature_cols].astype('float32')
y = df[target_col].astype('float32').values  


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


print('📌 Yapay Zeka Modeli Eğitiliyor...')
model = Sequential([
    Input(shape=(len(feature_cols),)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])


es  = EarlyStopping(patience=20, restore_best_weights=True)
rlr = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)


history = model.fit(
    X_train_sc, y_train,
    validation_data=(X_test_sc, y_test),
    epochs=300,
    batch_size=16,
    callbacks=[es, rlr],
    verbose=1
)


y_pred = model.predict(X_test_sc).flatten()
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")


save_model(model, 'urban_model.keras')
joblib.dump(scaler, "urban_scaler.save")
joblib.dump(feature_cols, "urban_columns.save")
print("✅ Model ve scaler kaydedildi.")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))

while True:
    try:
        bina_yasi = float(input('Bina yaşını giriniz (0–50): '))
        imar      = float(input('İmar uyumsuzluk (0–1): '))
        sikayet   = float(input('Şikayet oranı (0–1): '))
        afet      = float(input('Afet riski (0–1): '))
        zemin     = float(input('Zemin kötü (0–1): '))
        denetim   = float(input('Denetim eksiği (0–1): '))
    except ValueError:
        print("❌ Lütfen sayısal değerler giriniz.")
        continue

    
    bina_yasi = clamp(bina_yasi, 0, 50)
    imar      = clamp(imar, 0.0, 1.0)
    sikayet   = clamp(sikayet, 0.0, 1.0)
    afet      = clamp(afet, 0.0, 1.0)
    zemin     = clamp(zemin, 0.0, 1.0)
    denetim   = clamp(denetim, 0.0, 1.0)

    user_df = pd.DataFrame([[
        bina_yasi, imar, sikayet, afet, zemin, denetim
    ]], columns=feature_cols).astype('float32')

    user_sc = scaler.transform(user_df)
    score   = float(model.predict(user_sc, verbose=0)[0][0])

    if score <= 0.33:
        kategori = "İyi"
    elif score <= 0.66:
        kategori = "Orta"
    else:
        kategori = "Yüksek"

    print(f"🔮 Öncelik Skoru: {score:.3f}  →  {kategori}")
