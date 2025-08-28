import cv2
import os
import json
from deepface import DeepFace
import pyttsx3
from pyzbar.pyzbar import decode
from PIL import Image

# ========== SETTINGS ==========
KNOWN_FOLDER = "known_faces"
QR_FOLDER = "student_qrcodes"
CONFIDENCE_THRESHOLD = 0.5  # Lower = stricter match

# ========== TEXT-TO-SPEECH ==========
def speak(name):
    engine = pyttsx3.init()
    engine.say(f"{name}, please proceed")
    engine.runAndWait()

# ========== LOAD STUDENT DATA FROM QR CODES ==========
def load_students():
    students = {}
    for file in os.listdir(QR_FOLDER):
        if file.lower().endswith(".png"):
            path = os.path.join(QR_FOLDER, file)
            decoded = decode(Image.open(path))
            if decoded:
                data = decoded[0].data.decode("utf-8")
                student = json.loads(data)
                students[student["student_id"]] = student
    return students

# ========== SCAN FACES ==========
def scan_faces():
    students = load_students()
    cap = cv2.VideoCapture(0)
    print("📷 Camera ready. Please show your face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Graduation Scanner", frame)
        cv2.imwrite("temp_face.jpg", frame)

        matched = False
        for file in os.listdir(KNOWN_FOLDER):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                known_path = os.path.join(KNOWN_FOLDER, file)
                try:
                    result = DeepFace.verify("temp_face.jpg", known_path, enforce_detection=False)
                    if result["verified"] and result["distance"] < CONFIDENCE_THRESHOLD:
                        student_id = os.path.splitext(file)[0]
                        student_info = students.get(student_id)

                        if student_info:
                            name = student_info["name"]
                            print(f"✅ Matched: {name} ({student_id})")
                            cv2.putText(frame, f"{name} ({student_id})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            speak(name)
                            speak(student_info["description"])
                        else:
                            print(f"⚠️ No QR data for {student_id}")
                            cv2.putText(frame, f"{student_id}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                        matched = True
                        break
                except Exception as e:
                    print(f"❌ Error comparing: {file} - {e}")

        if matched:
            cv2.imshow("Verified", frame)
            cv2.waitKey(10000)
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ========== RUN ==========
if __name__ == "__main__":
    scan_faces()
