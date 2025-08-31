# student_qr_and_import.py
import os
import json
import qrcode
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore

# ========== Firebase init ==========
def get_db():
    load_dotenv(dotenv_path="./.env")
    raw = os.environ.get("FIREBASE_CREDENTIALS")
    if not raw:
        raise RuntimeError("Missing FIREBASE_CREDENTIALS in your .env file")

    cfg = json.loads(raw)
    cfg["private_key"] = cfg["private_key"].replace("\n", "\n")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cfg)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# ========== Unified student records ==========
STUDENTS = [
    {
        "seat_num": 1,
        "student_id": "24WMR09274",
        "name": "Tan Jee Cheng",
        "course": "Bachelor of Software Engineering",
        "award": "Graduated with Distinction",
        "registered": False,
    },
    {
        "seat_num": 2,
        "student_id": "24WMR09155",
        "name": "Cheong Jau Chun",
        "course": "Bachelor of Computer Science",
        "award": "Distinction and Book Prize Winner",
        "registered": False,
    },
    {
        "seat_num": 3,
        "student_id": "24WMR00559",
        "name": "Hiu Chin Jeng",
        "course": "Bachelor of Software Engineering",
        "award": "Graduated with Distinction",
        "registered": False,
    },
    {
        "seat_num": 4,
        "student_id": "24WMR03717",
        "name": "Ling Wey Xian",
        "course": "Bachelor of Software Development",
        "award": "Graduated with Merit",
        "registered": False,
    },
]

# ========== Generate QR codes (exclude 'registered' in QR payload) ==========
def generate_qr(students):
    output_folder = "student_qrcodes"
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for student in students:
        filename = f"{student['student_id']}.png"
        filepath = os.path.join(output_folder, filename)

        # Ensure image_path is filename only
        student["image_path"] = filename

        # Build QR payload EXCLUDING 'registered'
        qr_payload = {k: v for k, v in student.items() if k != "registered"}
        qr_json = json.dumps(qr_payload)

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_json)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filepath)

        count += 1

    print(f"Saved {count} QR codes.")

# ========== Firestore upsert ==========
def upsert_students(students):
    db = get_db()
    coll = db.collection("students")

    CHUNK = 400
    total = 0
    for i in range(0, len(students), CHUNK):
        batch = db.batch()
        chunk = students[i : i + CHUNK]

        for s in chunk:
            s = dict(s)
            s["seat_num"] = int(s.get("seat_num", 0))
            s.setdefault("registered", False)
            sid = s.get("student_id")
            if not sid:
                raise ValueError("Each student must include 'student_id'.")
            doc_ref = coll.document(sid)
            batch.set(doc_ref, s, merge=True)

        batch.commit()
        total += len(chunk)
        print(f"Committed {total}/{len(students)} records…")

# ========== Main ==========
if __name__ == "__main__":
    generate_qr(STUDENTS)      # prints only total QR count
    upsert_students(STUDENTS)  # Firestore upload
