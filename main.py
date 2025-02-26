import numpy as np 
import pandas as pd 
import os 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

color_data = pd.read_csv('colors.csv')
color_data.head ()
x = color_data [['R','G','B']].values
y = color_data ['color_name'].values
# Normaliasi data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# Split dataset untuk training dan testing
x_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.2,random_state=42 )

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit (x_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
train_preds = knn.predict(x_train)
train_acc = accuracy_score(y_train,train_preds)
print(f"Akurasi pada data latih: {train_acc*100:.2f}%")

import cv2
import numpy as np

# inisialisasi kamera
cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    if not ret :
        break
  # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    pixel_center = frame[height//2, width//2]
    
    # Normalisasi pixel sebelum prediksi
    pixel_center_scaled = scaler.transform([pixel_center])
    
    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    
    # Tampilkan warna pada frame
    cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()   
    