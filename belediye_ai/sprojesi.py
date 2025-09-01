import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATA_PATH = "sentetik_saglik_risk_verisi_ilce_enlem_boylam_kategorili.xlsx"
OUT_MODEL = "risk_model.keras"
OUT_SCALER = "risk_scaler.save"
OUT_COLUMNS = "risk_columns.save"


df = pd.read_excel(DATA_PATH)


feature_cols = [
    "Nufus",
    "Yas_Ortalamasi",
    "Gelir_Seviyesi",
    "Saglik_Basvuru",
    "Yesil_Alan_Orani",
    "Nufus_Yogunlugu",
]
target_col = "Risk_Skoru"


missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Veride eksik sütun(lar) var: {missing_cols}")

df = df.dropna(subset=feature_cols + [target_col]).copy()

X = df[feature_cols].astype(float)
y = df[target_col].astype(float).clip(0, 1)  


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


tf.random.set_seed(42)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),    
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")       
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse",
              metrics=["mae"])


early = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)
plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5
)


history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=300,
    batch_size=16,
    callbacks=[early, plateau],
    verbose=1
)


y_pred = model.predict(X_test_scaled).ravel()
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2   = r2_score(y_test, y_pred)

print("\n--- Test Performansı ---")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")


model.save(OUT_MODEL)
print(f"\n✅ Model kaydedildi: {OUT_MODEL}")


joblib.dump(scaler, OUT_SCALER)
print(f"✅ Scaler kaydedildi: {OUT_SCALER}")


joblib.dump(feature_cols, OUT_COLUMNS)
print(f"✅ Kolon listesi kaydedildi: {OUT_COLUMNS}")
