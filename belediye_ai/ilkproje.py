import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense,Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore



data = pd.read_excel('birlesik_su_tuketimi_3.xlsx')

df = pd.DataFrame(data)




df = pd.get_dummies(df, columns=['Mevsim'], drop_first=False)







X = df.drop(['SuTuketimi','Yil','Enlem','Boylam','Mahalle','Ilce'], axis=1)
y = df['SuTuketimi']

X_train, X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=42)




print('yapay zekam')
model = Sequential()
model.add(Input(shape=(X.shape[1],)))  
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))





optimizer = Adam(learning_rate=0.005)
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)     
X_test_scaled = scaler.transform(X_test)           


model.fit(X_train_scaled, y_train, epochs=300, verbose=1)

y_pred = model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
from sklearn.metrics import root_mean_squared_error
rmse = root_mean_squared_error(y_test, y_pred)


print(f"RMSE: {rmse:.2f}")
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f}")
print(f"R2 Skoru: {r2:.4f}")

import joblib
from keras.saving import save_model
save_model(model, 'modelim.keras')

joblib.dump(scaler, "scaler.save")
joblib.dump(X.columns.tolist(), "columns.save")

def veri_kontrolu(mevsim, sicaklik):
    sinirlar = {
        "Kış": (0, 20),
        "İlkbahar": (15, 30),
        "Yaz": (30, 45),
        "Sonbahar": (10, 25),
    }

    if mevsim not in sinirlar:
        raise ValueError(f"Geçersiz mevsim: {mevsim}")

    min_s, max_s = sinirlar[mevsim]
    if not (min_s <= sicaklik <= max_s):
        raise ValueError(f"{mevsim} için sıcaklık {min_s}–{max_s} °C arasında olmalı. Girdiğiniz: {sicaklik} °C")

while True:
    user_input_1 = input('Mahalleyi giriniz ')
    user_input_2 = input('Mevsimi giriniz (Kış, İlkbahar, Yaz, Sonbahar): ')
    user_input_3 = float(input('Sıcaklığı giriniz (°C): '))
    user_input_4 = float(input('Nüfusu giriniz: '))

    try:
        veri_kontrolu(user_input_2, user_input_3)
    except ValueError as e:
        print(f"❌ Hata: {e}")
        continue  

    
    user_data = pd.DataFrame({
        'Mevsim_' + user_input_2:[1],
        'Sicaklik':[user_input_3],
        'Nufus':[user_input_4],
        'Sicaklik_Mevsim_Etkisi':[user_input_3 * {'Kış': 0.5, 'İlkbahar': 0.7, 'Yaz': 1.0, 'Sonbahar': 0.6}[user_input_2]]
    })

    user_Data = user_data.reindex(columns=X.columns, fill_value=0)
    user_data_scaled = scaler.transform(user_Data)

    prediction = model.predict(user_data_scaled)
    tahmin_edilen_tuketim = prediction[0][0]
    print(f"🔮 Tahmini Su Tüketimi: {tahmin_edilen_tuketim:.0f} m³ (≈ {tahmin_edilen_tuketim*1000:.0f} litre)")

   

