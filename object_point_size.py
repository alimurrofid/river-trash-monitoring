import cv2

# Rasio konversi dari pixel ke cm
cm_per_pixel = 2.5 / 72  # ≈ 0.03472

# Global list untuk menyimpan titik
points = []
measuring = False

# Fungsi untuk menghitung jarak antar dua titik
def calculate_distance(pt1, pt2):
    return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

# Callback mouse
def mouse_callback(event, x, y, flags, param):
    global points, measuring
    if event == cv2.EVENT_LBUTTONDOWN and measuring:
        if len(points) < 2:
            points.append((x, y))

# Buka kamera
cap = cv2.VideoCapture(0)

cv2.namedWindow("Pixel to CM")
cv2.setMouseCallback("Pixel to CM", mouse_callback)

print("Tekan 's' untuk mengukur jarak antara dua titik.")
print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tampilkan titik yang diklik
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # Jika 2 titik sudah dipilih, hitung jarak
    if len(points) == 2:
        pixel_distance = calculate_distance(points[0], points[1])
        cm_distance = pixel_distance * cm_per_pixel
        cv2.putText(frame, f"{cm_distance:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)

    cv2.imshow("Pixel to CM", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        points = []
        measuring = True

cap.release()
cv2.destroyAllWindows()
