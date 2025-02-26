import numpy as np
import os
import cv2
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Path ke dataset
DATASET_PATH = 'training_dataset'

data = []
labels = []

# Membaca dataset dari folder
for color_name in os.listdir(DATASET_PATH):
    color_path = os.path.join(DATASET_PATH, color_name)
    if os.path.isdir(color_path):
        for img_name in os.listdir(color_path):
            img_path = os.path.join(color_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            avg_color = img.mean(axis=(0, 1))  # Ambil rata-rata warna RGB
            data.append(avg_color)
            labels.append(color_name)

# Konversi ke array numpy
X = np.array(data)
y = np.array(labels)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Melatih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Akurasi pada data latih: {train_acc*100:.2f}%")
print(f"Akurasi pada data uji: {accuracy*100:.2f}%")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definisi rentang warna untuk deteksi HSV
    colors_ranges = {
        'Red': [(0, 120, 70), (10, 255, 255)],
        'Green': [(36, 100, 100), (86, 255, 255)],
        'Blue': [(94, 80, 2), (126, 255, 255)],
        'Yellow': [(15, 100, 100), (35, 255, 255)],
        'Purple': [(129, 50, 70), (158, 255, 255)],
        'White': [(0, 0, 200), (180, 30, 255)],
        'Black': [(0, 0, 0), (180, 255, 50)]
    }

    for color_name, (lower, upper) in colors_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter hanya kontur besar untuk menghindari noise
        contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        if contours:
            # Ambil kontur terbesar agar deteksi lebih stabil
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ambil ROI dan hitung rata-rata warna di dalamnya
            roi = frame[y:y+h, x:x+w]
            avg_color_bgr = cv2.mean(roi)[:3]  # Ambil rata-rata warna dalam format BGR
            avg_color_rgb = avg_color_bgr[::-1]  # Konversi BGR ke RGB

            # Prediksi warna menggunakan model
            avg_color_scaled = scaler.transform([avg_color_rgb])
            predicted_color = model.predict(avg_color_scaled)[0]
            confidence = np.max(model.predict_proba(avg_color_scaled)) * 100  # Probabilitas prediksi

            # Warna bounding box diambil dari hasil rata-rata ROI
            bbox_color = tuple(map(int, avg_color_bgr))  # Gunakan warna asli dari objek

            # Gambar bounding box dengan warna yang sesuai dengan objek
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 3)

            # Tampilkan nama warna + akurasi real-time
            text = f"{predicted_color} ({confidence:.2f}%)"
            text_position = (x, y - 10) if y - 10 > 20 else (x, y + 20)
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)

    # Menampilkan hasil deteksi warna
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
