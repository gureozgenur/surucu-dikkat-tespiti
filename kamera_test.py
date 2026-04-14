import cv2
import numpy as np
import winsound
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model mimarisi
model = Sequential([
    Input(shape=(96, 96, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# NPZ agirliklarini yukle
data = np.load(r"C:\Users\ozgen\OneDrive\Masaüstü\surucu-dikkat\eye_weights.npz")
weights = [data[f'arr_{i}'] for i in range(len(data.files))]
model.set_weights(weights)

# Haar cascade dosyalari
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

IMG_SIZE = 96
sleepy_counter = 0
sleepy_threshold = 10
alarm_on = False

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alinamadi.")
        break

    # Ayna etkisini duzelt
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    status_text = "No Face"

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Sadece gerçek göz tespiti
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)

        if len(eyes) > 0:
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2] * e[3])

            eye_region = face_color[ey:ey + eh, ex:ex + ew]
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

            if eye_region is not None and eye_region.size > 0:
                eye_resized = cv2.resize(eye_region, (IMG_SIZE, IMG_SIZE))
                eye_normalized = eye_resized.astype("float32") / 255.0
                eye_input = np.expand_dims(eye_normalized, axis=0)

                pred = model.predict(eye_input, verbose=0)[0][0]

                if pred > 0.7:
                    status_text = f"Sleepy ({pred:.2f}) [Eye]"
                    sleepy_counter += 1

                elif pred < 0.3:
                    status_text = f"Awake ({pred:.2f}) [Eye]"
                    sleepy_counter = 0
                    alarm_on = False
                    winsound.PlaySound(None, winsound.SND_PURGE)

                else:
                    status_text = f"UNSURE ({pred:.2f}) [Eye]"
                    sleepy_counter = 0
                    alarm_on = False
                    winsound.PlaySound(None, winsound.SND_PURGE)

                if sleepy_counter >= sleepy_threshold:
                    status_text = "DROWSINESS ALERT! [Eye]"
                    if not alarm_on:
                        winsound.Beep(1000, 700)
                        alarm_on = True
            else:
                sleepy_counter = 0
                alarm_on = False
                winsound.PlaySound(None, winsound.SND_PURGE)
                status_text = "No Eye Region"
        else:
            sleepy_counter = 0
            alarm_on = False
            winsound.PlaySound(None, winsound.SND_PURGE)
            status_text = "No Eye"
    else:
        sleepy_counter = 0
        alarm_on = False
        winsound.PlaySound(None, winsound.SND_PURGE)

    if "UNSURE" in status_text:
        color = (0, 255, 255)   # sari
    elif "ALERT" in status_text:
        color = (0, 0, 255)     # kirmizi
    elif "No Eye" in status_text or "No Face" in status_text:
        color = (255, 255, 255) # beyaz
    else:
        color = (0, 255, 0)     # yesil

    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()