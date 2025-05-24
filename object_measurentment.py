import cv2

# Rasio konversi: 2.5 cm = 72 px
cm_per_pixel = 8.4 / 230  # ≈ 0.03472

# Inisialisasi kamera (0 adalah default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold untuk mendapatkan objek (bisa juga gunakan Canny edge atau deteksi warna)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Abaikan kontur kecil
        if cv2.contourArea(cnt) < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Ukuran dalam cm
        width_cm = w * cm_per_pixel
        height_cm = h * cm_per_pixel

        # Gambar bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Tampilkan ukuran objek dalam piksel & cm
        label = f"{w}px/{width_cm:.2f}cm x {h}px/{height_cm:.2f}cm"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Object Size Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
