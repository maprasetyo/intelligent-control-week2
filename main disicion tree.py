import numpy as np 
import pandas as pd 
import cv2
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Membaca dataset warna
color_data = pd.read_csv('colors.csv')

# Memisahkan fitur (R, G, B) dan label (color_name)
X = color_data[['R', 'G', 'B']].values
y = color_data['color_name'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Melatih model Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Evaluasi model
y_pred = dt.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
train_preds = dt.predict(X_train)
train_acc = accuracy_score(y_train, train_preds)
print(f"Akurasi Model pada Data Latih: {train_acc * 100:.2f}%")
print(f"Akurasi Model pada Data Uji: {test_acc * 100:.2f}%")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# **Perbaikan Rentang Warna HSV**
colors_ranges = {
    'Red': [(0, 120, 70), (10, 255, 255)],
    'Green': [(40, 50, 50), (85, 255, 255)],  # Diperbaiki agar tidak bertabrakan dengan biru
    'Blue': [(100, 100, 50), (130, 255, 255)],  # Diperbaiki agar tidak bertabrakan dengan hijau
    'Yellow': [(15, 100, 100), (35, 255, 255)],
    'Purple': [(129, 50, 70), (158, 255, 255)],
    'White': [(0, 0, 200), (180, 30, 255)],
    'Black': [(0, 0, 0), (180, 255, 50)]
}

# Warna bounding box sesuai dengan warna yang terdeteksi
color_bgr_map = {
    'Red': (0, 0, 255),
    'Green': (0, 255, 0),
    'Blue': (255, 0, 0),
    'Yellow': (0, 255, 255),
    'Purple': (128, 0, 128),
    'White': (255, 255, 255),
    'Black': (0, 0, 0)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi ke format HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color_name, (lower, upper) in colors_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # **Filter hanya kontur besar untuk menghindari noise**
        contours = [c for c in contours if cv2.contourArea(c) > 1200]
        
        if contours:
            # Ambil kontur terbesar
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ambil area warna dari ROI
            roi = frame[y:y+h, x:x+w]
            
            # **Gunakan kombinasi mean & median untuk warna yang lebih stabil**
            avg_color_mean = np.mean(roi, axis=(0, 1))
            avg_color_median = np.median(roi, axis=(0, 1))
            avg_color = (avg_color_mean + avg_color_median) / 2  # Gabungan mean dan median
            avg_color_rgb = avg_color[::-1]  # **Konversi ke RGB**

            # **Prediksi warna menggunakan model**
            avg_color_scaled = scaler.transform([avg_color_rgb])
            predicted_color = dt.predict(avg_color_scaled)[0]
            confidence = np.max(dt.predict_proba(avg_color_scaled)) * 100  # Probabilitas prediksi

            # **Jika confidence rendah, abaikan prediksi**
            if confidence < 60:
                predicted_color = "Unknown"

            # **Warna bounding box sesuai deteksi HSV**
            bbox_color = color_bgr_map[color_name]

            # **Gambar bounding box**
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 3)

            # **Tampilkan nama warna + akurasi real-time**
            text = f"{predicted_color} ({confidence:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2)

    # Menampilkan hasil deteksi warna
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
