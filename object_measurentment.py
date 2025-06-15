import cv2

# ====== KALIBRASI UKURAN DENGAN JARAK ======
ukuran_objek_cm = 20               # Ukuran real objek kalibrasi (cm)
ukuran_objek_px = 41               # Ukuran objek di kamera saat kalibrasi (pixel)
jarak_kalibrasi_cm = 200           # Jarak kamera ke objek saat kalibrasi (cm)
jarak_kamera_cm = 568.5            # Jarak kamera saat penggunaan (cm) - bisa diubah

# Hitung konstanta kalibrasi
k = ukuran_objek_cm / (ukuran_objek_px * jarak_kalibrasi_cm)
cm_per_pixel = k * jarak_kamera_cm

# ====== INISIALISASI KAMERA ======
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

print(f"Resolusi: {width}x{height}")
print(f"FPS: {fps}")
print(f"FOURCC Codec: {fourcc}")
print(f"Rasio cm/pixel: {cm_per_pixel:.5f} cm/pixel")

# ====== RESIZE UNTUK TAMPILAN WINDOW ======
display_width = 1280
display_height = 720
scale_x = display_width / width
scale_y = display_height / height

# ====== LOOP UTAMA ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)

        width_cm = w * cm_per_pixel
        height_cm = h * cm_per_pixel

        # Ubah warna ke biru (BGR: 255, 0, 0)
        color = (255, 0, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        label = f"{w}px/{width_cm:.2f}cm x {h}px/{height_cm:.2f}cm"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    display_frame = cv2.resize(frame, (display_width, display_height))
    cv2.imshow("Object Size Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
